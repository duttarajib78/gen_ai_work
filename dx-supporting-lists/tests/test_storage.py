"""Object stores (local + S3 with a fake client), the checkpoint, and the run lock."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from dx_support.storage.checkpoint import Checkpoint, latest_by_key, parse_jsonl
from dx_support.storage.locks import RunLock
from dx_support.storage.objectstore import LocalObjectStore, S3ObjectStore, StoreError, namespace, open_store, validate_key


class AwsError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """Just enough of the S3 client: keeps objects in a dict, honours If-None-Match, paginates listings in pages of 2."""

    def __init__(self):
        self.objects, self.put_kwargs = {}, []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise AwsError("NoSuchKey")
        data = self.objects[Key]
        return {"Body": type("B", (), {"read": lambda s: data})()}

    def put_object(self, Bucket, Key, Body, **kw):
        self.put_kwargs.append(kw)
        if kw.get("IfNoneMatch") == "*" and Key in self.objects:
            raise AwsError("PreconditionFailed")
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                for i in range(0, max(len(keys), 1), 2):
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 2]]}
        return P()


class KeyTests(unittest.TestCase):
    def test_keys_cannot_escape_or_be_malformed(self):
        for ok in ("a", "briefs/H25.json", "verdicts/H25/achi/ab12.jsonl", ".run.lock", "routes/H25/achi/0f.json"):
            self.assertEqual(validate_key(ok), ok)
        for bad in ("", "/abs", "../x", "a/../b", "a//b", "a b", "a\\b", "a/./b", "x" * 600, "ünï", "a;b"):
            with self.assertRaises(StoreError, msg=bad):
                validate_key(bad)


class StoreContract:
    """Behaviour every store must have; mixed into the concrete test cases."""

    def make(self):
        raise NotImplementedError

    def test_roundtrip_overwrite_delete(self):
        s = self.make()
        self.assertIsNone(s.get("a/b.json"))
        s.put("a/b.json", b"1")
        s.put("a/b.json", b"2")
        self.assertEqual(s.get("a/b.json"), b"2")
        s.delete("a/b.json")
        self.assertIsNone(s.get("a/b.json"))
        s.delete("a/b.json")                                   # deleting a missing key is not an error

    def test_put_if_absent_is_exclusive(self):
        s = self.make()
        self.assertTrue(s.put_if_absent(".run.lock", b"me"))
        self.assertFalse(s.put_if_absent(".run.lock", b"you"))
        self.assertEqual(s.get(".run.lock"), b"me")

    def test_list_by_prefix(self):
        s = self.make()
        for k in ("verdicts/H25/achi/1.jsonl", "verdicts/H25/mbs/2.jsonl", "verdicts/H250/achi/3.jsonl", "briefs/H25.json", "status/H25.json"):
            s.put(k, b"x")
        self.assertEqual(s.list("verdicts/H25/"), ["verdicts/H25/achi/1.jsonl", "verdicts/H25/mbs/2.jsonl"])
        self.assertEqual(s.list("verdicts/H25/achi/"), ["verdicts/H25/achi/1.jsonl"])
        self.assertEqual(len(s.list("verdicts/")), 3)
        self.assertEqual(s.list("nothing/"), [])


class LocalStoreTests(StoreContract, unittest.TestCase):
    def make(self):
        return LocalObjectStore(Path(tempfile.mkdtemp()))

    def test_no_temp_files_are_listed_and_traversal_is_refused(self):
        s = self.make()
        s.put("a/b.json", b"x")
        (s.root / "a" / ".tmp-leftover").write_bytes(b"junk")
        self.assertEqual(s.list("a/"), ["a/b.json"])
        with self.assertRaises(StoreError):
            s.get("../outside")


class S3StoreTests(StoreContract, unittest.TestCase):
    def make(self):
        self.fake = FakeS3()
        return S3ObjectStore("my-bucket", "pfx/ns", client=self.fake)

    def test_prefix_mapping_and_pagination(self):
        s = self.make()
        for i in range(5):
            s.put(f"x/{i}.json", b"v")
        self.assertIn("pfx/ns/x/0.json", self.fake.objects)
        self.assertEqual(s.list("x/"), [f"x/{i}.json" for i in range(5)])          # five keys over three pages, prefix stripped

    def test_server_side_encryption_is_explicit_only_with_a_kms_key(self):
        s = self.make()
        s.put("k.json", b"v")
        self.assertNotIn("ServerSideEncryption", self.fake.put_kwargs[-1])         # relies on the bucket default encryption
        k = S3ObjectStore("my-bucket", "", client=self.fake, kms_key_id="arn:aws:kms:ap-southeast-2:111122223333:key/abc")
        k.put("k2.json", b"v")
        self.assertEqual(self.fake.put_kwargs[-1]["ServerSideEncryption"], "aws:kms")
        self.assertEqual(self.fake.put_kwargs[-1]["SSEKMSKeyId"], "arn:aws:kms:ap-southeast-2:111122223333:key/abc")

    def test_conditional_write_header_and_unexpected_errors_propagate(self):
        s = self.make()
        s.put_if_absent("l", b"1")
        self.assertEqual(self.fake.put_kwargs[-1]["IfNoneMatch"], "*")

        class Broken(FakeS3):
            def put_object(self, **kw):
                raise AwsError("AccessDenied")
        with self.assertRaises(AwsError):
            S3ObjectStore("my-bucket", "", client=Broken()).put_if_absent("l", b"1")

    def test_bucket_name_is_validated(self):
        with self.assertRaises(StoreError):
            S3ObjectStore("Bad_Bucket!", "", client=FakeS3())


class PrefixAndOpenTests(unittest.TestCase):
    def test_namespaced_view_is_isolated(self):
        base = LocalObjectStore(Path(tempfile.mkdtemp()))
        a, b = namespace(base, "serve", "fp1"), namespace(base, "serve", "fp2")
        a.put("status/H25.json", b"A")
        b.put("status/H25.json", b"B")
        self.assertEqual((a.get("status/H25.json"), b.get("status/H25.json")), (b"A", b"B"))
        self.assertEqual(a.list("status/"), ["status/H25.json"])
        self.assertTrue(base.get("serve/fp1/status/H25.json"))
        self.assertIs(namespace(base), base)

    def test_open_store_uris(self):
        d = tempfile.mkdtemp()
        self.assertIsInstance(open_store(f"file://{d}"), LocalObjectStore)
        self.assertIsInstance(open_store(d), LocalObjectStore)
        s3 = open_store("s3://my-bucket/some/prefix")
        self.assertIsInstance(s3, S3ObjectStore)
        self.assertEqual((s3.bucket, s3.prefix), ("my-bucket", "some/prefix"))
        for bad in ("", "ftp://x/y", "gs://b/p"):
            with self.assertRaises(StoreError):
                open_store(bad)


def run(coro):
    return asyncio.run(coro)


def verdict(cat, fam, uid, tier="core", ts="2026-01-01T00:00:00"):
    return {"category": cat, "family": fam, "unit_id": uid, "support_tier": tier, "evidence_type": "treated", "rationale": "r",
            "member_scope": "all", "source": "route", "model": "m", "ts": ts}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.store = LocalObjectStore(Path(tempfile.mkdtemp()))

    def test_everything_written_is_found_by_a_fresh_instance(self):
        c1 = Checkpoint(self.store)

        async def write():
            await c1.write_brief("H25", {"clinical_summary": "s"})
            await c1.write_route("H25", "achi", 0, "k1", ["achi:2:42"])
            await c1.write_verdicts([verdict("H25", "achi", "4270100"), verdict("H25", "achi", "4270400", "broad"), verdict("H25", "mbs", "42702")])
            await c1.write_status({"category": "H25", "status": "complete", "families_complete": ["achi", "mbs"]})
        run(write())
        c1.close()
        c2 = Checkpoint(self.store)
        run(c2.ahydrate("H25"))
        self.assertEqual(c2.briefs["H25"], {"clinical_summary": "s"})
        self.assertEqual(c2.routes[("H25", "achi", "k1")], ["achi:2:42"])
        self.assertEqual(c2.verdict_keys, {("H25", "achi", "4270100"), ("H25", "achi", "4270400"), ("H25", "mbs", "42702")})
        self.assertEqual(c2.status["H25"]["status"], "complete")
        self.assertEqual(set(c2.read_verdicts("H25", "achi")), {("H25", "achi", "4270100"), ("H25", "achi", "4270400")})
        c2.close()

    def test_hydration_is_per_category_and_cached(self):
        c = Checkpoint(self.store)
        run(c.write_brief("H25", {"clinical_summary": "s"}))
        run(c.write_brief("I48", {"clinical_summary": "t"}))
        c.close()
        c2 = Checkpoint(self.store)
        c2.hydrate("H25")
        self.assertEqual(set(c2.briefs), {"H25"})                  # I48 was not loaded
        self.assertEqual(c2.cached_brief_ids(), {"H25", "I48"})
        c2.close()

    def test_batch_objects_are_deterministic_so_rewrites_do_not_duplicate(self):
        c = Checkpoint(self.store)
        recs = [verdict("H25", "achi", "b"), verdict("H25", "achi", "a")]
        run(c.write_verdicts(recs))
        run(c.write_verdicts(list(reversed(recs))))
        self.assertEqual(len(self.store.list("verdicts/H25/achi/")), 1)
        self.assertEqual(len(list(c.iter_verdicts(["H25"]))), 2)
        c.close()

    def test_newest_verdict_wins_by_timestamp_not_by_object_order(self):
        c = Checkpoint(self.store)
        run(c.write_verdicts([verdict("H25", "achi", "u1", "not_supporting", "2026-01-01T00:00:00")]))
        run(c.write_verdicts([verdict("H25", "achi", "u1", "core", "2026-02-01T00:00:00"), verdict("H25", "achi", "u2", "broad")]))
        got = c.read_verdicts("H25")
        self.assertEqual(got[("H25", "achi", "u1")]["support_tier"], "core")
        self.assertEqual(latest_by_key([verdict("A", "f", "u", "x", "2"), verdict("A", "f", "u", "y", "1")])[("A", "f", "u")]["support_tier"], "x")
        c.close()

    def test_corrupt_lines_are_skipped_and_status_loads_in_bulk(self):
        self.store.put("verdicts/H25/achi/bad.jsonl", b'{"category":"H25","family":"achi","unit_id":"u1"}\n{truncated\n\n')
        self.assertEqual(len(list(parse_jsonl(self.store.get("verdicts/H25/achi/bad.jsonl")))), 1)
        c = Checkpoint(self.store)
        run(c.write_status({"category": "H25", "status": "complete"}))
        run(c.write_status({"category": "I48", "status": "partial"}))
        c.close()
        c2 = Checkpoint(self.store)
        self.assertEqual({k: v["status"] for k, v in c2.load_status_all().items()}, {"H25": "complete", "I48": "partial"})
        self.assertEqual(set(c2.load_status_all(["I48"])), {"I48"})
        self.assertEqual(c2.refresh_status("H25")["status"], "complete")
        c2.close()

    def test_run_log_is_buffered_flushed_and_carries_no_prompts(self):
        c = Checkpoint(self.store, run_log_flush_every=3)
        for i in range(7):
            c.log_run({"stage": "judge", "ok": True, "cost_usd": 0.01, "n": i})
        c.close()                                                   # flushes the remainder
        recs = list(Checkpoint(self.store).iter_run_log())
        self.assertEqual(sorted(r["n"] for r in recs), list(range(7)))
        self.assertTrue(all("prompt" not in r for r in recs))
        self.assertGreaterEqual(len(self.store.list("runlog/")), 3)

    def test_a_failing_store_cannot_fail_the_run_log(self):
        class Boom(LocalObjectStore):
            def put(self, key, data):
                raise OSError("disk full")
        c = Checkpoint(Boom(Path(tempfile.mkdtemp())))
        c.log_run({"a": 1})
        c.flush_run_log()                                           # swallowed: telemetry never fails a run
        c._io.shutdown(wait=True)

    def test_writes_that_fail_surface_and_leave_memory_untouched(self):
        class Boom(LocalObjectStore):
            def put(self, key, data):
                raise OSError("disk full")
        c = Checkpoint(Boom(Path(tempfile.mkdtemp())))
        with self.assertRaises(OSError):
            run(c.write_brief("H25", {"clinical_summary": "s"}))
        self.assertNotIn("H25", c.briefs)
        c._io.shutdown(wait=True)


class LockTests(unittest.TestCase):
    def test_exclusive_and_released(self):
        s = LocalObjectStore(Path(tempfile.mkdtemp()))
        with RunLock(s):
            holder = RunLock.read(s)
            self.assertTrue(holder["pid"].isdigit() and "host" in holder and "started" in holder)
            with self.assertRaises(SystemExit) as e:
                RunLock(s).__enter__()
            self.assertIn("already exists", str(e.exception))
        self.assertIsNone(RunLock.read(s))
        with RunLock(s):
            pass


if __name__ == "__main__":
    unittest.main()
