"""Synthetic, PHI-free admissions for smoke tests and demos.

Records carry real structure (codes, procedures, claims, LOS and theatre time depend on the DRG and its
complexity split) so a tiny model can learn it in a few epochs. Never use this to evaluate model quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BASES = (("F", "10"), ("F", "62"), ("G", "20"), ("G", "45"), ("E", "65"))
_SPLITS = ("A", "B", "C", "D")
_SPLIT_PROBS = (0.2, 0.3, 0.3, 0.2)
_SECONDARY_COUNT = {"A": (5, 8), "B": (3, 6), "C": (2, 4), "D": (1, 3)}
_LOS_RANGE = {"A": (8, 20), "B": (4, 9), "C": (2, 5), "D": (1, 3)}


def generate_frame(n_rows: int, seed: int = 0, n_noise_codes: int = 20) -> pd.DataFrame:
    """Return a dataframe that satisfies the data contract (see ``drg_audit.data.io``)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        mdc, base = _BASES[int(rng.integers(len(_BASES)))]
        split = _SPLITS[int(rng.choice(len(_SPLITS), p=_SPLIT_PROBS))]
        low, high = _SECONDARY_COUNT[split]
        pool = [f"{mdc}{base}.{j}" for j in range(8)]
        secondary = [str(c) for c in rng.choice(pool, size=min(int(rng.integers(low, high + 1)), len(pool)),
                                                replace=False)]
        noise = [f"N.{int(k)}" for k in rng.integers(0, n_noise_codes, size=int(rng.integers(0, 3)))]
        procedures = ([f"P{base}-{int(k)}" for k in rng.integers(0, 3, size=int(rng.integers(0, 3)))]
                      if split in ("A", "B") else [])
        claims = [f"M{base}{int(k)}" for k in rng.integers(0, 4, size=int(rng.integers(1, 4)))]
        los_low, los_high = _LOS_RANGE[split]
        has_theatre = split in ("A", "B")
        rows.append({
            "admission_no": str(1_000_000 + i),
            "diagnosis_code_array": [f"{mdc}{base}.P", *secondary, *noise],
            "procedure_code_array": procedures,
            "medical_claims_array": claims,
            "drg_vers": str(rng.choice(["9.0", "10.0"])),
            "drg": f"{mdc}{base}{split}",
            "los": int(rng.integers(los_low, los_high + 1)),
            "minutes_operating_theatre": float(max(0.0, rng.normal(180, 30))) if has_theatre else 0.0,
            "hours_of_mechanical_ventilation": float(rng.integers(0, 48)) if split == "A" and rng.random() < 0.3
            else 0.0,
            "member_age": float(rng.integers(18, 95)),
            "drg_benefit": float(round(rng.lognormal(8.5, 0.5), 2)),
        })
    return pd.DataFrame(rows)


def write_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)  # requires pyarrow
