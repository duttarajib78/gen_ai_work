import json
import tempfile
import unittest
from pathlib import Path

from rule_extractor.errors import IngestionError
from rule_extractor.ingestion import mbs
from rule_extractor.ingestion.chunking import sliding_windows, standard_chunks
from rule_extractor.ingestion.corpus import CorpusIndex, build_corpus, canon_code, verify_corpus
from rule_extractor.ingestion.pdf import detect_printed_page
from rule_extractor.ingestion.sources import resolve_sources
from rule_extractor.ingestion.toc import parse_toc
from rule_extractor.settings import Settings

from .helpers import write_source_set


class ChunkingTests(unittest.TestCase):
    def test_window_counts_match_measured_plan(self):
        self.assertEqual(len(sliding_windows(614, 6, 1)), 123)
        self.assertEqual(len(sliding_windows(418, 6, 1)), 84)
        self.assertEqual(len(sliding_windows(598, 10, 1)), 67)
        self.assertEqual(len(sliding_windows(286, 10, 1)), 32)

    def test_windows_cover_every_page_with_overlap(self):
        for total, ppc, ov in [(614, 6, 1), (5, 6, 1), (6, 6, 1), (7, 6, 1), (13, 4, 0)]:
            windows = sliding_windows(total, ppc, ov)
            covered = set()
            for s, e in windows:
                covered.update(range(s, e + 1))
            self.assertEqual(covered, set(range(1, total + 1)), (total, ppc, ov))
            for (s1, e1), (s2, _) in zip(windows, windows[1:]):
                self.assertEqual(e1 - s2 + 1, ov)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            sliding_windows(10, 3, 3)
        self.assertEqual(sliding_windows(0, 3, 1), [])

    def test_standard_chunks_share_boundary_page_and_cap(self):
        anchors = [{"code": "0001", "title": "A", "start_page": 3}, {"code": "0002", "title": "B", "start_page": 3},
                   {"code": "0010", "title": "C", "start_page": 10}, {"code": "0099", "title": "Z", "start_page": None}]
        chunks = standard_chunks(anchors, total_pages=200, max_pages=5)
        self.assertEqual([(c.code, c.start_page, c.end_page) for c in chunks],
                         [("0001", 3, 3), ("0002", 3, 7), ("0010", 10, 14)])


class TocTests(unittest.TestCase):
    def test_dot_leaders_and_plain(self):
        text = "0001 Admission .......... 1\nACS 0010 Principal diagnosis 12\n0300 Diabetes mellitus type 2 45\n"
        entries = parse_toc(text)
        self.assertEqual([(e.code, e.title, e.printed_page) for e in entries],
                         [("0001", "Admission", 1), ("0010", "Principal diagnosis", 12), ("0300", "Diabetes mellitus type 2", 45)])

    def test_wrapped_titles_and_duplicates(self):
        text = "0110 Bacterial, viral and other\ninfectious agents ........ 101\n0110 Bacterial dup ... 9\n0111 Sepsis 102\n"
        entries = parse_toc(text)
        self.assertEqual([(e.code, e.printed_page) for e in entries], [("0110", 101), ("0111", 102)])
        self.assertEqual(entries[0].title, "Bacterial, viral and other infectious agents")

    def test_printed_page_detection(self):
        self.assertEqual(detect_printed_page("body\nmore body\n27"), "27")
        self.assertEqual(detect_printed_page("xii\nbody"), "xii")
        self.assertEqual(detect_printed_page("body text that ends with a number 12"), "")


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_source_set(self.root / "source_docs")
        self.settings = Settings.load(env={
            "RULE_EXTRACTOR_SOURCE_DOCS_DIR": str(self.root / "source_docs"),
            "RULE_EXTRACTOR_CORPUS_DIR": str(self.root / "corpus"),
            "RULE_EXTRACTOR_MIN_ACS_STANDARDS": "5",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_and_read_back(self):
        manifest = build_corpus(self.settings)
        self.assertEqual(manifest["documents"]["acs"]["pages"], 7)
        self.assertEqual(manifest["mbs"], {"items": 3, "facts": 2})
        index = CorpusIndex(self.settings.paths.corpus_dir)
        self.assertEqual(index.page_count("icd10am_tabular"), 3)
        self.assertTrue(index.page_text("acs", 6).startswith("=== acs | pdf_page 6 | printed_page"))
        standards = {c.code: c for c in index.standards()}
        self.assertEqual(len(standards), 6)
        self.assertEqual((standards["0110"].start_page, standards["0110"].end_page), (6, 7))
        self.assertEqual(standards["0001"].start_page, 3)
        self.assertIn("Example 4", index.window_text("acs", standards["0110"].start_page, standards["0110"].end_page))
        codes = index.known_codes()
        self.assertIn(canon_code("A41.0"), codes.icd10am)
        self.assertIn(canon_code("B95.6"), codes.icd10am)
        self.assertIn(canon_code("40903-01"), codes.achi)
        self.assertIn("49318", codes.mbs)
        self.assertEqual(verify_corpus(self.settings.paths.corpus_dir, min_standards=5), [])
        self.assertTrue(index.mbs_facts_path().is_file())
        self.assertEqual(len(index.digest()), 64)

    def test_incremental_rebuild_skips_unchanged_documents(self):
        build_corpus(self.settings)
        marker = self.settings.paths.corpus_dir / "acs" / "marker.txt"
        marker.write_text("kept")
        build_corpus(self.settings)
        self.assertTrue(marker.exists())            # acs untouched
        build_corpus(self.settings, force=True)
        self.assertFalse(marker.exists())           # rebuilt

    def test_min_standards_guard_fails_closed(self):
        strict = Settings.load(env={
            "RULE_EXTRACTOR_SOURCE_DOCS_DIR": str(self.root / "source_docs"),
            "RULE_EXTRACTOR_CORPUS_DIR": str(self.root / "corpus2"),
            "RULE_EXTRACTOR_MIN_ACS_STANDARDS": "50",
        })
        with self.assertRaises(IngestionError):
            build_corpus(strict)

    def test_missing_source_is_reported(self):
        (self.root / "source_docs" / "ACHI_Tabular_List_Thirteenth_Edition__10_Mar_2025_.pdf").unlink()
        with self.assertRaises(IngestionError):
            build_corpus(self.settings)

    def test_page_bounds_enforced(self):
        build_corpus(self.settings)
        index = CorpusIndex(self.settings.paths.corpus_dir)
        for bad in (0, 99, -1):
            with self.assertRaises(IngestionError):
                index.page_text("acs", bad)

    def test_filename_overrides_cannot_traverse(self):
        with self.assertRaises(ValueError):
            resolve_sources({"acs": {"filename": "../../etc/passwd"}})
        with self.assertRaises(ValueError):
            resolve_sources({"nope": {"filename": "x.pdf"}})


class MbsTests(unittest.TestCase):
    def test_facts_and_lookup(self):
        from .helpers import MBS_XML

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.xml"
            p.write_text(MBS_XML, encoding="utf-8")
            items = mbs.parse_mbs_xml(p)
            self.assertEqual([i.item_num for i in items], ["49318", "10990", "23"])
            facts = list(mbs.billing_facts(items))
            self.assertEqual([(f["fact_type"], f["item_num"]) for f in facts], [("emsn_cap", "10990"), ("derived_fee", "10990")])
            lines = list(mbs.lookup_lines(items))
            self.assertTrue(lines[0].startswith("ITEM 49318 | category 3 | group T8 | fee 812.50"))

    def test_xxe_and_entity_expansion_are_not_resolved(self):
        xml = """<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY xxe SYSTEM "file:///etc/passwd"><!ENTITY a "AAAA"><!ENTITY b "&a;&a;&a;&a;">]>
<r><Data><ItemNum>1</ItemNum><Description>&xxe;</Description><Category>&b;</Category></Data></r>"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "evil.xml"
            p.write_text(xml, encoding="utf-8")
            try:
                items = mbs.parse_mbs_xml(p)
            except IngestionError:
                return  # refusing the document is also an acceptable outcome
            self.assertNotIn("root:", items[0].description)
            self.assertNotIn("AAAA", items[0].category)

    def test_malformed_and_empty_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.xml"
            bad.write_text("<a><b>", encoding="utf-8")
            with self.assertRaises(IngestionError):
                mbs.parse_mbs_xml(bad)
            empty = Path(d) / "empty.xml"
            empty.write_text("<a><b><c/></b></a>", encoding="utf-8")
            with self.assertRaises(IngestionError):
                mbs.parse_mbs_xml(empty)
            with self.assertRaises(IngestionError):
                mbs.parse_mbs_xml(Path(d) / "missing.xml")


if __name__ == "__main__":
    unittest.main()
