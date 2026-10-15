"""A tiny, hand-built, NON-CLINICAL stand-in for the four licensed reference files.

It exists so the whole system (loaders, indexes, pipeline, service, compile step) can be exercised end to end with no licensed data and no
credentials: unit tests, `dx-support make-sample-data`, local demos, and CI. The wording is invented for plumbing checks; nothing here is a
real ICD-10-AM / ACHI / MBS / prostheses extract and nothing about it is clinically meaningful.
"""
from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape

from dx_support.indexing.families import build_achi_index, build_dx_index, build_mbs_index, build_prosthesis_index
from dx_support.indexing.universe import ACHI_FILE, DIAGNOSIS_FILE, PROSTHESES_FILE, Universe
from dx_support.reference.categories import build_categories
from dx_support.reference.loaders import MbsItem, ProsRecord

DIAGNOSES = {
    "H250": "Senile incipient cataract", "H251": "Senile nuclear cataract", "H252": "Senile cataract, morgagnian type",
    "H258": "Other senile cataract", "H259": "Senile cataract, unspecified",
    "I480": "Paroxysmal atrial fibrillation", "I481": "Persistent atrial fibrillation",
    "I489": "Atrial fibrillation and atrial flutter, unspecified",
    "K350": "Acute appendicitis with generalised peritonitis", "K358": "Acute appendicitis, other and unspecified",
    "K37": "Unspecified acute appendicitis",
    "V010": "Pedestrian injured in collision with pedal cycle",
    **{f"M25{i:02d}": f"Joint pain, site {i}" for i in range(25)},
}
ACHI = {
    "4270100": "Insertion of intraocular lens", "4270400": "Removal of intraocular lens",
    "4260000": "Extracapsular extraction of lens by phacoemulsification", "3880000": "Insertion of cardiac pacemaker",
    "3880100": "Electrocardiographic monitoring of atrial fibrillation", "3890000": "Cardioversion of atrial fibrillation",
    "5500000": "Joint pain site assessment of joint", "5500100": "Joint pain site injection of joint",
    "5510000": "Ultrasound of joint pain site", "6300000": "Sole child chain test procedure",
    "8800005": "Provisional use of 88000-05 [8888]",
}
MBS = [
    MbsItem("3", "1", "A1", "", "1", "Professional attendance at consulting rooms by a general practitioner", False),
    MbsItem("23010", "3", "T10", "", "", "Anaesthesia, perfusion or assistance, if the service time is not more than 15 minutes (1 basic units)", False),
    MbsItem("42702", "3", "T8", "9", "", "Lens extraction and insertion of intraocular lens", False),
    MbsItem("42710", "3", "T8", "9", "", "Intraocular lens, removal of, and replacement", False),
    MbsItem("38200", "3", "T8", "5", "", "Cardioversion of atrial fibrillation by direct current", False),
    MbsItem("11700", "2", "D1", "", "", "Electrocardiogram monitoring of atrial fibrillation for 12 hours", False),
    MbsItem("55114", "5", "I1", "", "", "Ultrasound scan of joint pain site", False),
    MbsItem("99999", "3", "T8", "9", "", "Ended service", True),
]
_LENS = ("01 - Ophthalmic", "01.02 - POSTERIOR CHAMBER INTRAOCULAR LENSES", "01.02.02 - Foldable", "01.02.02.01 - Hydrophobic")
_PM = ("08 - Cardiac", "08.04 - Single Chamber Pacemakers", "08.04.01 - Standard", "")
PROSTHESES = {
    "EG009": ProsRecord("EG009", _LENS, ("ifc",), True, True, 1), "EG010": ProsRecord("EG010", _LENS, (), True, False, 1),
    "AB123": ProsRecord("AB123", _PM, ("m",), True, True, 1),
    "QHV01": ProsRecord("QHV01", ("01 - Cardio-thoracic", "", "", ""), (), False, False, 1),
}


def synthetic_universe() -> Universe:
    """The same data, built in memory (no files)."""
    cats = build_categories(DIAGNOSES)
    indices = {"achi": build_achi_index(ACHI), "mbs": build_mbs_index(MBS), "prosthesis": build_prosthesis_index(PROSTHESES),
               "dx": build_dx_index(cats)}
    for idx in indices.values():
        idx.bm25()
    return Universe(categories=cats, indices=indices, report={}, digest="synthetic")


def write_sample_data(data_dir: Path) -> dict[str, Path]:
    """Writes the four files in the exact formats the real loaders expect. Returns their paths."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    def tsv(name: str, code_type: str, rows: dict[str, str]) -> Path:
        p = data_dir / name
        with open(p, "w", encoding="utf-8", newline="") as fh:
            for code, desc in rows.items():
                fh.write("\t".join(["ALL", code_type, code, "13", "", "", desc, desc]) + "\n")
        return p

    out = {"diagnosis": tsv(DIAGNOSIS_FILE, "ICD10_DIAGNOSIS", DIAGNOSES), "achi": tsv(ACHI_FILE, "ICD10_PROCEDURE", ACHI)}
    pros = data_dir / PROSTHESES_FILE
    with open(pros, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Product Category", "Sub Category", "Product Group", "Product Sub Group", "Suffix", "Billing Code", "Benefit"])
        for code, rec in PROSTHESES.items():
            w.writerow([*rec.path, rec.suffixes[0] if rec.suffixes else "No Suffix", code, "$100.00" if rec.in_latest_extract else "100"])
    out["prostheses"] = pros
    mbs = data_dir / "MBS-XML-20260801.XML"
    with open(mbs, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n<MBS_XML>\n')
        for it in MBS:
            fh.write(f"  <Data><ItemNum>{escape(it.code)}</ItemNum><Category>{escape(it.category)}</Category><Group>{escape(it.group)}</Group>"
                     f"<SubGroup>{escape(it.subgroup)}</SubGroup><SubHeading>{escape(it.subheading)}</SubHeading>"
                     f"<Description>{escape(it.description)}</Description><ItemEndDate>{'20200101' if it.ended else ''}</ItemEndDate></Data>\n")
        fh.write("</MBS_XML>\n")
    out["mbs"] = mbs
    return out
