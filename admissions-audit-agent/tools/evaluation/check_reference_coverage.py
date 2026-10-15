"""How much of the evidence resolves to a real description?

The model sees descriptions, not codes. A code with no description reaches it as
"(description not found)", and a model reasoning over many such lines cannot judge
support well whatever the prompt or model. This checks coverage without any model call.

    python -m tools.evaluation.check_reference_coverage --input episodes.parquet --reference-dir ./reference \
        [--mbs-array-field hospital_claims_array] [--min-resolution 0.95]

Exits non-zero when any category is below --min-resolution, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from audit_agent.reference_data import NOT_FOUND, ReferenceData
from tools.tabular import coerce_to_list, load_table


class Tally:
    def __init__(self) -> None:
        self.total = 0
        self.resolved = 0
        self.missing: Counter = Counter()
        self.episodes_with_missing: set = set()

    def add(self, code: str, ok: bool, episode: object) -> None:
        self.total += 1
        if ok:
            self.resolved += 1
        else:
            self.missing[code] += 1
            self.episodes_with_missing.add(episode)

    @property
    def rate(self) -> Optional[float]:
        return self.resolved / self.total if self.total else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--mbs-array-field", default="medical_claims_array", choices=("medical_claims_array", "hospital_claims_array"))
    p.add_argument("--n-samples", type=int)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--min-resolution", type=float, default=0.95)
    a = p.parse_args(argv)

    ref = ReferenceData.from_directory(a.reference_dir)
    df = load_table(a.input)
    if a.n_samples:
        df = df.head(a.n_samples)
    tallies = {"Diagnosis (PDx+ADx)": Tally(), "ACHI": Tally(), f"MBS ({a.mbs_array_field})": Tally()}
    sources = [("diagnosis_code_array", ref.resolve_diagnosis), ("procedure_code_array", ref.resolve_achi),
               (a.mbs_array_field, ref.resolve_mbs)]
    for _, row in df.iterrows():
        for (column, resolve), tally in zip(sources, tallies.values()):
            for code in coerce_to_list(row.get(column, [])):
                code = str(code).strip()
                tally.add(code, resolve(code) != NOT_FOUND, row.get("admission_no"))

    print(f"Reference data: {ref.counts()} (MBS file {ref.mbs_source_file}); {len(df)} episode(s) checked.\n")
    failed = False
    for label, t in tallies.items():
        if not t.total:
            print(f"{label}: no codes seen")
            continue
        print(f"{label}: {t.resolved}/{t.total} resolved ({t.rate:.1%}); {len(t.episodes_with_missing)} episode(s) affected")
        for code, n in t.missing.most_common(a.top):
            print(f"    {code!r}: {n}")
        failed = failed or t.rate < a.min_resolution
    if failed:
        print(f"\nBelow {a.min_resolution:.0%} in at least one category. If MBS is low, try the other --mbs-array-field.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
