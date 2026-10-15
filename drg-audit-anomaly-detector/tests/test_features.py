import unittest

from tests.helpers import make_record, make_vocab  # noqa: I001  (must precede drg_audit imports)

import numpy as np

from drg_audit.features import (
    CLS, MASK, PAD, SEP, UNK, encode_records, encode_tokens, meta_features, record_tokens, split_drg,
    stay_type_bucket,
)
from drg_audit.schemas import TokenizationSpec


class TokenizationTests(unittest.TestCase):
    def test_pdx_is_masked_and_never_present(self):
        tokens = record_tokens(make_record(), TokenizationSpec.standard())
        self.assertEqual(tokens[0], MASK)
        self.assertNotIn("I21.4", tokens)  # principal diagnosis must not leak

    def test_standard_sorts_every_group(self):
        record = make_record(diagnosis_code_array=["P", "Z9", "A1"], procedure_code_array=["9", "1"],
                             medical_claims_array=["55114", "38200"])
        self.assertEqual(record_tokens(record, TokenizationSpec.standard()),
                         [MASK, "A1", "Z9", "1", "9", "38200", "55114"])

    def test_legacy_training_keeps_claim_order(self):
        record = make_record(medical_claims_array=["55114", "38200"])
        tokens = record_tokens(record, TokenizationSpec.legacy_training())
        self.assertEqual(tokens[-2:], ["55114", "38200"])

    def test_truncation_semantics_differ_by_spec(self):
        vocab = make_vocab(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        tokens = list("abcdefghij")
        std = encode_tokens(tokens, vocab, TokenizationSpec(max_len=8))
        legacy = encode_tokens(tokens, vocab, TokenizationSpec(max_len=8, keep_sep_on_truncation=False))
        self.assertEqual(len(std), 8)
        self.assertEqual(std[0], vocab[CLS])
        self.assertEqual(std[-1], vocab[SEP])
        self.assertEqual(len(legacy), 8)
        self.assertNotIn(vocab[SEP], legacy)

    def test_unknown_tokens_map_to_unk(self):
        vocab = make_vocab(["known"])
        ids = encode_tokens(["known", "mystery"], vocab, TokenizationSpec())
        self.assertEqual(ids, [vocab[CLS], vocab["known"], vocab[UNK], vocab[SEP]])


class MetaFeatureTests(unittest.TestCase):
    def test_stay_buckets(self):
        expected = {1: 0.0, 0: 0.25, 2: 0.25, 3: 0.25, 4: 0.5, 7: 0.5, 8: 0.75, 14: 0.75, 15: 1.0}
        for los, bucket in expected.items():
            self.assertEqual(stay_type_bucket(los), bucket, msg=f"los={los}")

    def test_meta_vector(self):
        meta = meta_features(make_record(los=1, minutes_operating_theatre=60.0,
                                         hours_of_mechanical_ventilation=0.0, member_age=40.0))
        self.assertAlmostEqual(meta[0], 0.0)
        self.assertAlmostEqual(meta[1], np.log1p(60.0))
        self.assertAlmostEqual(meta[2], 0.0)
        self.assertAlmostEqual(meta[3], 0.4)

    def test_negative_and_nonfinite_inputs_are_clamped(self):
        meta = meta_features(make_record(minutes_operating_theatre=-5.0,
                                         hours_of_mechanical_ventilation=float("nan"),
                                         member_age=float("inf")))
        self.assertEqual(meta[1], 0.0)
        self.assertEqual(meta[2], 0.0)
        self.assertAlmostEqual(meta[3], 0.5)


class DrgSplitTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_drg("F62A"), ("F", "62", "A"))
        self.assertEqual(split_drg(" F62 "), ("F", "62", "D"))  # missing split defaults to D
        self.assertIsNone(split_drg("F6"))
        self.assertIsNone(split_drg(""))
        self.assertIsNone(split_drg(None))


class EncodeRecordsTests(unittest.TestCase):
    def setUp(self):
        self.vocab = make_vocab(["E11.9", "I10", "38497-00", "38200", "55114"])
        self.versions = {"10.0": 0, "11.0": 1}
        self.spec = TokenizationSpec(max_len=16)

    def test_shapes_and_unknown_version(self):
        records = [make_record(), make_record(drg_vers="99.0")]
        batch = encode_records(records, self.vocab, self.versions, self.spec)
        self.assertEqual(batch.input_ids.shape, (2, 16))
        self.assertEqual(batch.meta.shape, (2, 4))
        self.assertEqual(batch.unknown_version.tolist(), [False, True])
        self.assertEqual(batch.version_ids.tolist(), [0, 0])  # unknown falls back to index 0

    def test_dynamic_padding_is_equivalent(self):
        records = [make_record(), make_record(diagnosis_code_array=["I21.4"], procedure_code_array=[],
                                              medical_claims_array=[])]
        full = encode_records(records, self.vocab, self.versions, self.spec)
        trimmed = encode_records(records, self.vocab, self.versions, self.spec, pad_to_longest=True)
        width = trimmed.input_ids.shape[1]
        self.assertLess(width, 16)
        np.testing.assert_array_equal(full.input_ids[:, :width], trimmed.input_ids)
        np.testing.assert_array_equal(full.attention_mask.sum(axis=1), trimmed.attention_mask.sum(axis=1))
        self.assertTrue((full.input_ids[:, width:] == self.vocab[PAD]).all())


if __name__ == "__main__":
    unittest.main()
