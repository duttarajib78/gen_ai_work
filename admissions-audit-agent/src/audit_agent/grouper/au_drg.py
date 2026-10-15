"""Adapter around the internal ``au_grouper`` module and the ``groupy`` client.

``au_grouper`` (and the ``groupy`` package it uses) are internal and are not
part of this repository: place ``au_grouper.py`` in ``vendor/`` before building
the image and install ``groupy`` via ``requirements/grouper.txt``. The adapter
calls them exactly the way the validated prototype did:

    drg_grouper_test(row, groupy, use_alt_diag=True, new_diag_col="new_diag_poa")
    calculate_possible_recovery(one_row_df, drg_df, drg_desc)

Failure mapping: a bad-input style error (KeyError/ValueError/TypeError) becomes
``GrouperInputError`` (this episode cannot be regrouped, HTTP 200 failed); any
other error is treated as the service being unavailable (HTTP 503, retryable).
"""
from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable, Optional

import pandas as pd

from ..config import Settings
from ..deadline import Deadline
from ..errors import ConfigError, GrouperInputError, GrouperUnavailableError
from ..schemas import Episode
from .corrections import apply_diagnosis_corrections, rows_to_diag_poa
from .drg_version import derive_grouper_version
from .interface import RegroupResult

logger = logging.getLogger(__name__)


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None or pd.isna(value) else float(value)


class AuDrgGrouper:
    def __init__(
        self,
        *,
        au_grouper_module: Any,
        groupy: Any,
        drg_df: Any,
        drg_desc: Any,
        serialize_calls: bool = True,
        min_budget_s: float = 5.0,
        version_deriver: Callable = derive_grouper_version,
    ) -> None:
        self._au = au_grouper_module
        self._groupy = groupy
        self._drg_df = drg_df
        self._drg_desc = drg_desc
        self._lock = threading.Lock() if serialize_calls else None
        self._min_budget_s = min_budget_s
        self._derive_version = version_deriver

    @classmethod
    def from_settings(cls, settings: Settings) -> "AuDrgGrouper":
        try:
            au = importlib.import_module("au_grouper")
        except ImportError as exc:
            raise GrouperUnavailableError(
                "the internal 'au_grouper' module is not importable; vendor it into the image "
                "(vendor/au_grouper.py) and install groupy, or set AUDIT_GROUPER_BACKEND=disabled"
            ) from exc
        drg_dir = settings.model_dir / "drg"
        if not drg_dir.is_dir():
            raise ConfigError("the model artifact has no 'drg/' directory (DRG cost-weight reference data)")
        try:
            groupy = au.Groupy(lorica_stack_name=settings.grouper_stack_name)
            drg_df, drg_desc = au.get_metadata(str(drg_dir))
        except Exception as exc:  # noqa: BLE001
            raise GrouperUnavailableError(f"grouper start-up failed ({type(exc).__name__})") from exc
        return cls(
            au_grouper_module=au, groupy=groupy, drg_df=drg_df, drg_desc=drg_desc,
            serialize_calls=settings.grouper_serialize_calls, min_budget_s=settings.min_grouper_budget_s,
        )

    def regroup(
        self,
        episode: Episode,
        *,
        removed_codes: frozenset,
        pdx_resequence: Optional[tuple[str, str]],
        deadline: Deadline,
    ) -> RegroupResult:
        deadline.check(self._min_budget_s, what="the grouper call")
        corrected_rows = apply_diagnosis_corrections(episode, removed_codes, pdx_resequence)

        row = pd.Series(episode.to_row())
        row["new_diag_poa"] = rows_to_diag_poa(corrected_rows)
        for column in ("dte_admitted", "dte_separated"):
            if not hasattr(row[column], "strftime"):
                row[column] = pd.to_datetime(row[column])

        try:
            one_row_df = self._derive_version(pd.DataFrame([row]))
            row = one_row_df.iloc[0]
            result, scored = self._run(row, one_row_df)
            r = scored.iloc[0]
        except (KeyError, ValueError, TypeError) as exc:
            raise GrouperInputError(
                f"the grouper could not process this episode ({type(exc).__name__}); check the grouper input fields"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise GrouperUnavailableError(f"the grouper service call failed ({type(exc).__name__})") from exc

        original_cw = _opt_float(r.get("drg_rw"))
        regrouped_cw = _opt_float(r.get("regroup_drg_value"))
        drop = (original_cw - regrouped_cw) if original_cw is not None and regrouped_cw is not None else None
        original_drg = episode.extras.get("drg")
        regrouped_drg = _opt_str(r.get("regroup_drg"))
        return RegroupResult(
            codes_removed=tuple(sorted(removed_codes)),
            pdx_resequence=pdx_resequence,
            original_drg=original_drg,
            original_drg_desc=_opt_str(r.get("drg_desc")),
            original_cost_weight=original_cw,
            regrouped_drg=regrouped_drg,
            regrouped_drg_desc=_opt_str(r.get("regroup_drg_desc")),
            regrouped_cost_weight=regrouped_cw,
            drg_changed=(original_drg != regrouped_drg),
            cost_weight_drop=drop,
            possible_recovery=_opt_float(r.get("possible_recovery")),
            drg_ver_grouper=_opt_str(r.get("drg_ver_grouper")),
        )

    def _run(self, row: Any, one_row_df: Any):
        def call():
            result = self._au.drg_grouper_test(row, self._groupy, use_alt_diag=True, new_diag_col="new_diag_poa")
            one_row_df["regroup_drg"] = result["regroup_drg"]
            one_row_df["cost_weight"] = result["cost_weight"]
            one_row_df["original_cost_weight"] = result["original_cost_weight"]
            return result, self._au.calculate_possible_recovery(one_row_df, self._drg_df, self._drg_desc)

        if self._lock is None:
            return call()
        with self._lock:
            return call()
