"""Shared test helpers: synthetic PDFs/XML/workbooks and in-memory fakes. No real licensed content is used."""
from __future__ import annotations

from pathlib import Path


def make_pdf(path: Path, pages: list[str]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()


ACS_TOC = [
    ("0001", "Admission", 1), ("0002", "Clinical documentation", 1), ("0010", "Principal diagnosis", 2),
    ("0011", "Additional diagnoses", 3), ("0110", "Bacterial, viral and other infectious agents", 4),
    ("0111", "Sepsis and related conditions", 5),
]


def acs_pages() -> list[str]:
    toc = "Contents\n" + "\n".join(f"{c} {t} {'.' * 12} {p}" for c, t, p in ACS_TOC)
    body = {
        3: "0001 Admission\nAn admission is the process by which a patient commences an episode of care.\n"
           "0002 Clinical documentation\nCoding must be based on clinical documentation.\n1",
        4: "0010 Principal diagnosis\nThe principal diagnosis is the diagnosis established after study.\n2",
        5: "0011 Additional diagnoses\nAdditional diagnoses are conditions that coexist.\n3",
        6: "0110 Bacterial, viral and other infectious agents\nB95-B97 codes must not duplicate an organism named in the code.\n"
           "Example 4: A41.0 with B95.6 is redundant.\n4",
        7: "0111 Sepsis and related conditions\nSepsis is coded per the Tabular List.\n5",
    }
    return ["Australian Coding Standards\nThirteenth Edition", toc] + [body[i] for i in range(3, 8)]


def write_source_set(source_dir: Path) -> None:
    from rule_extractor.ingestion.sources import DEFAULT_SOURCES

    source_dir.mkdir(parents=True, exist_ok=True)
    make_pdf(source_dir / DEFAULT_SOURCES["acs"].filename, acs_pages())
    make_pdf(source_dir / DEFAULT_SOURCES["icd10am_tabular"].filename, [
        "Chapter 1\nA41.0 Sepsis due to Staphylococcus aureus\nA41.1 Sepsis due to other specified staphylococcus\n1",
        "B95.6 Staphylococcus aureus as the cause of diseases classified to other chapters\nB95-B97 Excludes: organism named\n2",
        "E11.9 Type 2 diabetes mellitus without complication\n3",
    ])
    make_pdf(source_dir / DEFAULT_SOURCES["achi_tabular"].filename, [
        "40903-00 Neuroendoscopy\n40903-01 Neuroendoscopy with biopsy\n1",
        "13882-00 Management of continuous ventilatory support, 24 hours\n2",
    ])
    make_pdf(source_dir / DEFAULT_SOURCES["icd10am_alphabetic"].filename,
             [f"Index entry {i}\nSepsis - see also Infection\n{i}" for i in range(1, 4)])
    make_pdf(source_dir / DEFAULT_SOURCES["achi_alphabetic"].filename, ["Biopsy - see also Excision\n1", "Neuroendoscopy\n2"])
    (source_dir / DEFAULT_SOURCES["mbs_xml"].filename).write_text(MBS_XML, encoding="utf-8")


MBS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MBS_XML>
  <Data><ItemNum>49318</ItemNum><Category>3</Category><Group>T8</Group><ScheduleFee>812.50</ScheduleFee>
        <DerivedFee></DerivedFee><EMSNCap>NC</EMSNCap><Description>Knee arthroscopy item</Description></Data>
  <Data><ItemNum>10990</ItemNum><Category>1</Category><Group>A1</Group><ScheduleFee>40.00</ScheduleFee>
        <DerivedFee>55.10</DerivedFee><DerivedFeeStartDate>01/07/2025</DerivedFeeStartDate>
        <EMSNCap>CA</EMSNCap><EMSNFixedCapAmount>200.00</EMSNFixedCapAmount><Description>Consultation</Description></Data>
  <Data><ItemNum>23</ItemNum><Category>1</Category><Group>A1</Group><ScheduleFee>20.00</ScheduleFee>
        <DerivedFee></DerivedFee><Description>Plain item</Description></Data>
</MBS_XML>
"""


def make_env_dict(tmp, **extra_env):
    """Write synthetic sources + a template workbook and return the matching environment mapping."""
    from .test_excel_writer import make_template

    tmp = Path(tmp)
    write_source_set(tmp / "source_docs")
    make_template(tmp / "template.xlsx")
    env = {
        "AWS_REGION": "ap-southeast-2",
        "RULE_EXTRACTOR_SOURCE_DOCS_DIR": str(tmp / "source_docs"),
        "RULE_EXTRACTOR_CORPUS_DIR": str(tmp / "corpus"),
        "RULE_EXTRACTOR_MODEL_DIR": str(tmp / "model"),
        "RULE_EXTRACTOR_WORK_DIR": str(tmp / "work"),
        "RULE_EXTRACTOR_OUTPUT_DIR": str(tmp / "out"),
        "RULE_EXTRACTOR_STATE_DIR": str(tmp / "state"),
        "RULE_EXTRACTOR_TEMPLATE_PATH": str(tmp / "template.xlsx"),
        "RULE_EXTRACTOR_MIN_ACS_STANDARDS": "5",
        "RULE_EXTRACTOR_RETRY_BASE_DELAY_SECONDS": "0",
        "RULE_EXTRACTOR_MAX_RETRIES": "2",
        "RULE_EXTRACTOR_CONCURRENCY": "2",
    }
    env.update(extra_env)
    return env


def make_env(tmp, **extra_env):
    """Synthetic source docs -> built corpus -> Settings pointing at it (+ a template workbook). Returns Settings."""
    from rule_extractor.ingestion.corpus import build_corpus
    from rule_extractor.settings import Settings

    settings = Settings.load(env=make_env_dict(tmp, **extra_env))
    build_corpus(settings)
    return settings
