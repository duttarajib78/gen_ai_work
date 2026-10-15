"""ICD-10-AM chapters (an approximate grouping aid, not an authoritative reference)."""
from __future__ import annotations

import argparse
from typing import Optional

# Boundaries follow the standard ICD-10 chapter ranges. Two deliberate differences from a naive table, both found by checking every real
# category: V starts at 00 (category V00 exists in diagnosis_codes.tsv), and chapter II runs to D49.
_ICD10AM_CHAPTERS: list[tuple[str, list[tuple[str, int, int]]]] = [
    ("I", [("A", 0, 99), ("B", 0, 99)]),
    ("II", [("C", 0, 99), ("D", 0, 49)]),
    ("III", [("D", 50, 89)]),
    ("IV", [("E", 0, 90)]),
    ("V", [("F", 0, 99)]),
    ("VI", [("G", 0, 99)]),
    ("VII", [("H", 0, 59)]),
    ("VIII", [("H", 60, 95)]),
    ("IX", [("I", 0, 99)]),
    ("X", [("J", 0, 99)]),
    ("XI", [("K", 0, 93)]),
    ("XII", [("L", 0, 99)]),
    ("XIII", [("M", 0, 99)]),
    ("XIV", [("N", 0, 99)]),
    ("XV", [("O", 0, 99)]),
    ("XVI", [("P", 0, 96)]),
    ("XVII", [("Q", 0, 99)]),
    ("XVIII", [("R", 0, 99)]),
    ("XIX", [("S", 0, 99), ("T", 0, 98)]),
    ("XX", [("V", 0, 99), ("W", 0, 99), ("X", 0, 99), ("Y", 0, 98)]),
    ("XXI", [("Z", 0, 99)]),
    ("XXII", [("U", 0, 99)]),
]
VALID_CHAPTER_IDS = [c for c, _ in _ICD10AM_CHAPTERS]

# Chapters whose categories describe circumstances/activities/places rather than clinical conditions. A procedure cannot "indicate that the
# condition was treated, investigated or examined" for these, so they are skipped by default.
CONTEXT_ONLY_CHAPTERS = ("XX", "XXII")

CHAPTER_TITLES = {
    "I": "Certain infectious and parasitic diseases", "II": "Neoplasms",
    "III": "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    "IV": "Endocrine, nutritional and metabolic diseases", "V": "Mental and behavioural disorders",
    "VI": "Diseases of the nervous system", "VII": "Diseases of the eye and adnexa", "VIII": "Diseases of the ear and mastoid process",
    "IX": "Diseases of the circulatory system", "X": "Diseases of the respiratory system", "XI": "Diseases of the digestive system",
    "XII": "Diseases of the skin and subcutaneous tissue", "XIII": "Diseases of the musculoskeletal system and connective tissue",
    "XIV": "Diseases of the genitourinary system", "XV": "Pregnancy, childbirth and the puerperium",
    "XVI": "Certain conditions originating in the perinatal period",
    "XVII": "Congenital malformations, deformations and chromosomal abnormalities",
    "XVIII": "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    "XIX": "Injury, poisoning and certain other consequences of external causes",
    "XX": "External causes of morbidity and mortality", "XXI": "Factors influencing health status and contact with health services",
    "XXII": "Codes for special purposes",
}


def parse_chapter(value: str) -> str:
    """argparse `type` for one chapter: accepts 1-22 or a Roman numeral (any case, optional word "chapter") and returns the Roman id."""
    v = str(value).strip().upper().replace("CHAPTER", "").strip()
    if v.isdigit() and 1 <= int(v) <= len(VALID_CHAPTER_IDS):
        return VALID_CHAPTER_IDS[int(v) - 1]
    if v in VALID_CHAPTER_IDS:
        return v
    raise argparse.ArgumentTypeError(f"{value!r} is not a chapter: use 1-{len(VALID_CHAPTER_IDS)} or a Roman numeral I-XXII "
                                     "(see `dx-support estimate --list-chapters`)")


def parse_chapters(value: str) -> tuple[str, ...]:
    """One or more chapters as a comma-separated list and/or ranges -- "9", "9,10", "IX,X", "9-11", "9-11,13" -- or "all".
    Returns the Roman ids in chapter order, without duplicates."""
    picked: set[str] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() == "all":
            picked.update(VALID_CHAPTER_IDS)
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            i, j = VALID_CHAPTER_IDS.index(parse_chapter(a)), VALID_CHAPTER_IDS.index(parse_chapter(b))
            if i > j:
                raise argparse.ArgumentTypeError(f"{part!r} is a backwards range")
            picked.update(VALID_CHAPTER_IDS[i: j + 1])
        else:
            picked.add(parse_chapter(part))
    if not picked:
        raise argparse.ArgumentTypeError(f"{value!r} names no chapter: use e.g. 9, 9,10, IX,X or 9-11")
    return tuple(c for c in VALID_CHAPTER_IDS if c in picked)


def icd10am_chapter(category: str) -> Optional[str]:
    """Map a 3-character ICD-10-AM category to its chapter id, or None."""
    if not category or len(category) < 3 or not category[0].isalpha() or not category[1:3].isdigit():
        return None
    letter, num = category[0].upper(), int(category[1:3])
    for chapter_id, ranges in _ICD10AM_CHAPTERS:
        for lt, lo, hi in ranges:
            if letter == lt and lo <= num <= hi:
                return chapter_id
    return None
