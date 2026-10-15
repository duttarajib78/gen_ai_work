"""Alert generation (replaces the prototype's ``main.py``).

For admissions the model flags as billing anomalies, regroup the admission with the principal diagnosis
removed (using the internal AR-DRG grouper), quantify the possible recovery and emit one JSON line per alert.

Differences from the prototype script, all deliberate:
  * no runtime ``pip install`` (supply-chain risk; dependencies are pinned and installed up front)
  * no hard-coded paths or business rules: everything is in ``configs/alerts.yaml``
  * no unbound-variable crash when nothing qualifies
  * admissions without a matching claim are skipped and counted instead of crashing on ``int(None)``
  * output is written atomically to a run-specific file (re-runs never duplicate lines) with mode 0600
  * the Excel copy (which duplicates clinical data) is opt-in
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from drg_audit._mapping import dataclass_from_mapping
from drg_audit.config import load_yaml
from drg_audit.data.io import read_table
from drg_audit.errors import ConfigError, DataValidationError
from drg_audit.logging_utils import get_logger, log_event

logger = get_logger("alerts")

ALERT_TEMPLATE = """
The admission was flagged by a model for the below reason, however please review ALL coding in the admission and if recommended, please add a suitable comment for the coding error.<br>
This admission was flagged as having the PDx {pdx} which is poorly supported by the amount of resources utilized to manage the condition and also procedures performed / medical investigations carried out.

This rule is proposing to remove the PDx {pdx} and regroup. However please review the coding and identify if re-sequencing or removal would be more appropriate.

This admission has DRG {drg} (with charged cost weight {drg_rw}).
Removing {pdx} regroups to DRG {regroup_drg} - {regroup_drg_desc} with cost weight {regroup_drg_value} in DRG version {drg_ver_grouper}.
"""

GROUPER_OUTPUT_COLUMNS = (
    "admission_no", "principal_diagnosis_code", "drg", "drg_rw", "regroup_drg", "regroup_drg_desc",
    "regroup_drg_value", "drg_ver_grouper", "possible_recovery", "hospital_claims_seqno_array",
    "hospital_claims_service_class_code_array", "hospital_claims_array",
)


@dataclass(frozen=True)
class RuleConfig:
    alert_instance_id: int
    concept_name: str = "medical_investigation_mismatch_with_drg_benefit"
    target_service_class_codes: list[str] = field(default_factory=lambda: ["DRG_CASE_PAYMENT", "EPM", "CWO"])
    alert_info_prefix: str = ""


@dataclass(frozen=True)
class EligibilityConfig:
    fund_ids: list[int] = field(default_factory=lambda: [102, 103])
    min_diagnosis_count: int = 2
    exclude_principal_dx_prefixes: list[str] = field(default_factory=lambda: ["C"])
    exclude_first_secondary_dx_prefixes: list[str] = field(default_factory=lambda: ["R", "Z"])
    exclude_previously_alerted: bool = True


@dataclass(frozen=True)
class GrouperConfig:
    lorica_stack_name: str
    reference_data_dir: str
    backend: str = "au_grouper"


@dataclass(frozen=True)
class AlertIOConfig:
    admissions_path: str
    past_alerts_path: str
    predictions_path: str
    output_dir: str
    output_filename: str = "{concept_name}_{run_id}_json.txt"
    write_excel: bool = False


@dataclass(frozen=True)
class AlertsConfig:
    rule: RuleConfig
    grouper: GrouperConfig
    io: AlertIOConfig
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlertsConfig:
        return dataclass_from_mapping(cls, load_yaml(path))


class GrouperBackend(Protocol):
    def regroup_without_principal_diagnosis(self, df: pd.DataFrame, reference_data_dir: str) -> pd.DataFrame: ...


class AuGrouperBackend:
    """Adapter for the internal ``au_grouper`` package (not part of this repository)."""

    def __init__(self, lorica_stack_name: str) -> None:
        try:
            from au_grouper import Groupy, run_grouper
        except ImportError as exc:
            raise ConfigError(
                "the internal 'au_grouper' package is not installed; install a pinned version from the "
                "private package index") from exc
        self._groupy = Groupy(lorica_stack_name=lorica_stack_name)
        self._run_grouper = run_grouper

    def regroup_without_principal_diagnosis(self, df: pd.DataFrame, reference_data_dir: str) -> pd.DataFrame:
        return self._run_grouper(df, self._groupy, reference_data_dir)


def build_backend(cfg: GrouperConfig) -> GrouperBackend:
    if cfg.backend != "au_grouper":
        raise ConfigError(f"unsupported grouper backend '{cfg.backend}'")
    return AuGrouperBackend(cfg.lorica_stack_name)


# --------------------------------------------------------------------------------------
def _as_bool(series: pd.Series) -> pd.Series:
    mapped = series.map(lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true")
    return mapped.astype(bool)


def read_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path) if str(path).lower().endswith(".csv") else read_table(path)
    missing = [c for c in ("admission_no", "is_billing_anomaly") if c not in df.columns]
    if missing:
        raise DataValidationError(f"predictions are missing columns: {missing}")
    return df


def select_candidates(
    admissions: pd.DataFrame, past_alerts: pd.DataFrame, predictions: pd.DataFrame, rules: EligibilityConfig
) -> pd.DataFrame:
    """Admissions flagged by the model that also satisfy the business eligibility rules."""
    adm = admissions.copy()
    adm["admission_no"] = adm["admission_no"].astype("int64")
    flagged = predictions.loc[_as_bool(predictions["is_billing_anomaly"]), "admission_no"].astype("int64").unique()

    dx = adm["diagnosis_code_array"]
    keep = dx.apply(lambda codes: len(codes) >= rules.min_diagnosis_count)
    if rules.exclude_previously_alerted:
        keep &= ~adm["admission_no"].isin(past_alerts["admission_no"].astype("int64").unique())
    if rules.exclude_principal_dx_prefixes:
        pdx = adm["principal_diagnosis_code"].fillna("").astype(str)
        keep &= ~pdx.str.startswith(tuple(rules.exclude_principal_dx_prefixes))
    if rules.exclude_first_secondary_dx_prefixes:
        prefixes = tuple(rules.exclude_first_secondary_dx_prefixes)
        keep &= ~dx.apply(lambda codes: len(codes) > 1 and str(codes[1]).startswith(prefixes))
    keep &= adm["fund_id"].isin(rules.fund_ids)
    keep &= adm["admission_no"].isin(flagged)

    candidates = adm.loc[keep].reset_index(drop=True)
    candidates["target_code"] = candidates["principal_diagnosis_code"]
    candidates["grouper_swap"] = "remove"
    return candidates


def pick_claim_seqno(
    seqnos: Sequence[Any], service_classes: Sequence[Any], claims: Sequence[Any],
    target_classes: Sequence[str], target_codes: str = "",
) -> Any | None:
    """First claim whose service class is targeted (and, if ``target_codes`` is given as ``a|b``, whose
    claim text contains one of the codes). Returns ``None`` when no claim matches."""
    wanted = [c for c in target_codes.split("|") if c] if target_codes else []
    for seqno, service_class, claim in zip(seqnos, service_classes, claims):
        if service_class not in target_classes:
            continue
        if not wanted or any(code in claim for code in wanted):
            return seqno
    return None


def render_alert_info(row: Any) -> str:
    return ALERT_TEMPLATE.format(
        pdx=row.principal_diagnosis_code, drg=row.drg, drg_rw=row.drg_rw, regroup_drg=row.regroup_drg,
        regroup_drg_desc=row.regroup_drg_desc, regroup_drg_value=row.regroup_drg_value,
        drg_ver_grouper=row.drg_ver_grouper)


def build_alert_records(quantified: pd.DataFrame, rule: RuleConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    missing = [c for c in GROUPER_OUTPUT_COLUMNS if c not in quantified.columns]
    if missing:
        raise DataValidationError(f"grouper output is missing columns: {missing}")
    unique = quantified.drop_duplicates(subset="admission_no", keep="first").reset_index(drop=True)
    skipped = {"no_matching_claim": 0, "invalid_recovery_amount": 0}
    records: list[dict[str, Any]] = []
    for row in unique.itertuples(index=False):
        seqno = pick_claim_seqno(row.hospital_claims_seqno_array, row.hospital_claims_service_class_code_array,
                                 row.hospital_claims_array, rule.target_service_class_codes)
        if seqno is None:
            skipped["no_matching_claim"] += 1
            continue
        recovery = row.possible_recovery
        if recovery is None or (isinstance(recovery, float) and not math.isfinite(recovery)):
            skipped["invalid_recovery_amount"] += 1
            continue
        records.append({
            "claim_seqno": str(int(seqno)),
            "alert_info": rule.alert_info_prefix + render_alert_info(row),
            "recovery_amount": round(recovery),
            "alert_instance_id": int(rule.alert_instance_id),
        })
    return records, skipped


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        os.chmod(tmp_name, 0o600)  # output contains clinical context: owner-only
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def generate_alerts(
    cfg: AlertsConfig, *, backend: GrouperBackend | None = None, run_id: str | None = None
) -> dict[str, Any]:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    admissions = read_table(cfg.io.admissions_path)
    past_alerts = read_table(cfg.io.past_alerts_path)
    predictions = read_predictions(cfg.io.predictions_path)

    candidates = select_candidates(admissions, past_alerts, predictions, cfg.eligibility)
    summary: dict[str, Any] = {"run_id": run_id, "n_candidates": len(candidates), "n_alerts": 0, "output": None}
    if candidates.empty:
        log_event(logger, "no_candidates", run_id=run_id)
        return summary

    backend = backend or build_backend(cfg.grouper)
    quantified = backend.regroup_without_principal_diagnosis(candidates, cfg.grouper.reference_data_dir)
    quantified = quantified.copy()
    quantified["admission_no"] = quantified["admission_no"].astype("int64")
    records, skipped = build_alert_records(quantified, cfg.rule)

    out_dir = Path(cfg.io.output_dir)
    if records:
        target = out_dir / cfg.io.output_filename.format(concept_name=cfg.rule.concept_name, run_id=run_id)
        write_jsonl_atomic(target, records)
        summary["output"] = str(target)
    if cfg.io.write_excel:
        out_dir.mkdir(parents=True, exist_ok=True)
        excel_path = out_dir / f"{cfg.rule.concept_name}_{run_id}_quantified.xlsx"
        quantified.drop_duplicates(subset="admission_no").to_excel(excel_path, index=False)
        os.chmod(excel_path, 0o600)
    summary.update(n_alerts=len(records), skipped=skipped)
    log_event(logger, "alerts_generated", run_id=run_id, n_candidates=len(candidates), n_alerts=len(records),
              **{f"skipped_{k}": v for k, v in skipped.items()})
    return summary
