import tempfile
import threading
import unittest
from pathlib import Path

from rule_extractor.domain.jobs import JobRecord, JobStatus, UnitRecord, UnitStatus
from rule_extractor.errors import ConflictError, NotFoundError
from rule_extractor.storage.aws import DynamoStateStore, S3ArtifactStore, SqsWorkQueue
from rule_extractor.storage.local import LocalArtifactStore, LocalStateStore
from rule_extractor.storage.memory import InMemoryArtifactStore, InMemoryStateStore, InMemoryWorkQueue

from .fakes import FakeDynamoClient, FakeS3Client, FakeSqsClient


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def new_job(job_id="job-aaaaaaaaaaaa", status=JobStatus.RUNNING.value, created="2026-10-15T00:00:00Z"):
    return JobRecord(job_id=job_id, status=status, created_at=created)


class StateStoreContract:
    """Behaviour every StateStore implementation must satisfy."""

    def make(self):
        raise NotImplementedError

    def test_job_roundtrip_and_conflict(self):
        s = self.make()
        s.put_job(new_job())
        self.assertEqual(s.get_job("job-aaaaaaaaaaaa").status, "RUNNING")
        with self.assertRaises(ConflictError):
            s.put_job(new_job())
        self.assertIsNone(s.get_job("job-missing"))

    def test_update_job_applies_or_declines(self):
        s = self.make()
        s.put_job(new_job())
        applied, rec = s.update_job("job-aaaaaaaaaaaa", lambda j: setattr(j, "cost_usd", 2.5) or True)
        self.assertTrue(applied)
        self.assertEqual(s.get_job("job-aaaaaaaaaaaa").cost_usd, 2.5)
        applied, rec = s.update_job("job-aaaaaaaaaaaa", lambda j: False)
        self.assertFalse(applied)
        self.assertEqual(s.update_job("job-nope", lambda j: True), (False, None))

    def test_units_listing_is_ordered_and_update_is_cas(self):
        s = self.make()
        s.put_job(new_job())
        s.put_units([UnitRecord("job-aaaaaaaaaaaa", f"u{i}", "acs_topic", order=10 - i) for i in range(30)])
        units = s.list_units("job-aaaaaaaaaaaa")
        self.assertEqual(len(units), 30)
        self.assertEqual([u.order for u in units], sorted(u.order for u in units))
        applied, u = s.update_unit("job-aaaaaaaaaaaa", "u3", lambda x: setattr(x, "status", UnitStatus.RUNNING.value) or True)
        self.assertTrue(applied)
        self.assertEqual(s.get_unit("job-aaaaaaaaaaaa", "u3").status, "RUNNING")
        self.assertIsNone(s.get_unit("job-aaaaaaaaaaaa", "nope"))

    def test_concurrent_updates_do_not_lose_writes(self):
        s = self.make()
        s.put_job(new_job())

        def bump():
            for _ in range(20):
                s.update_job("job-aaaaaaaaaaaa", lambda j: setattr(j, "cost_usd", j.cost_usd + 1) or True)

        threads = [threading.Thread(target=bump) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(s.get_job("job-aaaaaaaaaaaa").cost_usd, 120)

    def test_idempotency_reservation(self):
        s = self.make()
        self.assertIsNone(s.reserve_idempotency("key-0001", "job-1", "digest-a"))
        self.assertEqual(s.reserve_idempotency("key-0001", "job-2", "digest-b"), ("job-1", "digest-a"))

    def test_list_jobs_by_status_tracks_transitions(self):
        s = self.make()
        s.put_job(new_job("job-aaaaaaaaaaaa", created="2026-10-15T00:00:01Z"))
        s.put_job(new_job("job-bbbbbbbbbbbb", created="2026-10-15T00:00:02Z"))
        s.update_job("job-aaaaaaaaaaaa", lambda j: setattr(j, "status", "SUCCEEDED") or True)
        self.assertEqual([j.job_id for j in s.list_jobs_by_status("RUNNING")], ["job-bbbbbbbbbbbb"])
        self.assertEqual([j.job_id for j in s.list_jobs_by_status("SUCCEEDED")], ["job-aaaaaaaaaaaa"])


class InMemoryStateTests(StateStoreContract, unittest.TestCase):
    def make(self):
        return InMemoryStateStore()


class DynamoStateTests(StateStoreContract, unittest.TestCase):
    def make(self):
        return DynamoStateStore(FakeDynamoClient(page_size=7), "tbl")

    def test_batch_write_retries_unprocessed_items(self):
        client = FakeDynamoClient()
        client.unprocessed_once = True
        s = DynamoStateStore(client, "tbl")
        s.put_job(new_job())
        s.put_units([UnitRecord("job-aaaaaaaaaaaa", f"u{i}", "x", i) for i in range(10)])
        self.assertEqual(len(s.list_units("job-aaaaaaaaaaaa")), 10)

    def test_items_carry_ttl_and_sparse_gsi_keys(self):
        client = FakeDynamoClient()
        s = DynamoStateStore(client, "tbl", retention_days=1, clock=lambda: 100.0)
        s.put_job(new_job())
        s.put_units([UnitRecord("job-aaaaaaaaaaaa", "u1", "x", 0)])
        job_item = client.items[("JOB#job-aaaaaaaaaaaa", "META")]
        unit_item = client.items[("JOB#job-aaaaaaaaaaaa", "UNIT#u1")]
        self.assertEqual(job_item["ttl"]["N"], str(100 + 86400))
        self.assertIn("gsi1pk", job_item)
        self.assertNotIn("gsi1pk", unit_item)


class LocalStateTests(StateStoreContract, unittest.TestCase):
    def make(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return LocalStateStore(self._tmp.name)

    def test_state_survives_restart(self):
        s = self.make()
        s.put_job(new_job())
        s.put_units([UnitRecord("job-aaaaaaaaaaaa", "u1", "x", 0)])
        s.reserve_idempotency("key-0001", "job-aaaaaaaaaaaa", "d")
        again = LocalStateStore(self._tmp.name)
        self.assertEqual(again.get_job("job-aaaaaaaaaaaa").job_id, "job-aaaaaaaaaaaa")
        self.assertEqual(len(again.list_units("job-aaaaaaaaaaaa")), 1)
        self.assertEqual(again.reserve_idempotency("key-0001", "x", "y"), ("job-aaaaaaaaaaaa", "d"))


class ArtifactContract:
    def make(self):
        raise NotImplementedError

    def test_roundtrip_list_exists(self):
        a = self.make()
        uri = a.put_bytes("jobs/j1/units/a.jsonl", b"hello")
        self.assertTrue(uri.endswith("jobs/j1/units/a.jsonl"))
        a.put_bytes("jobs/j1/units/b.jsonl", b"x")
        a.put_bytes("jobs/j2/units/c.jsonl", b"y")
        self.assertEqual(a.get_bytes("jobs/j1/units/a.jsonl"), b"hello")
        self.assertTrue(a.exists("jobs/j1/units/a.jsonl"))
        self.assertFalse(a.exists("jobs/j1/units/zzz.jsonl"))
        self.assertEqual(a.list_keys("jobs/j1/"), ["jobs/j1/units/a.jsonl", "jobs/j1/units/b.jsonl"])
        with self.assertRaises(NotFoundError):
            a.get_bytes("jobs/j1/none")

    def test_unsafe_keys_rejected(self):
        a = self.make()
        for bad in ("../x", "/abs", "a//b", "a/../b", "a\\b", ""):
            with self.assertRaises(ValueError):
                a.put_bytes(bad, b"x")


class InMemoryArtifactTests(ArtifactContract, unittest.TestCase):
    def make(self):
        return InMemoryArtifactStore()


class LocalArtifactTests(ArtifactContract, unittest.TestCase):
    def make(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return LocalArtifactStore(self._tmp.name)


class S3ArtifactTests(ArtifactContract, unittest.TestCase):
    def make(self):
        self.client = FakeS3Client(page_size=2)
        return S3ArtifactStore(self.client, "bkt", "pfx", kms_key_id="key-id")

    def test_prefix_and_kms_are_applied(self):
        a = self.make()
        a.put_bytes("k/a", b"1")
        (bucket, key), obj = next(iter(self.client.objects.items()))
        self.assertEqual((bucket, key), ("bkt", "pfx/k/a"))
        self.assertEqual(obj["extra"], {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "key-id"})
        self.assertEqual(a.uri("k/a"), "s3://bkt/pfx/k/a")


class QueueContract:
    def make(self, clock):
        raise NotImplementedError

    def test_visibility_ack_extend_and_release(self):
        clock = Clock()
        q = self.make(clock)
        q.send_batch([f"m{i}" for i in range(12)])
        got = q.receive(10, 0, 60)
        self.assertEqual(len(got), 10)
        self.assertEqual(q.depth(), (2, 10))
        self.assertEqual(len(q.receive(10, 0, 60)), 2)     # the remaining two
        self.assertEqual(q.receive(10, 0, 60), [])         # everything in flight
        q.ack(got[0])
        q.extend(got[1], 0)                                 # release immediately
        again = q.receive(10, 0, 60)
        self.assertEqual([m.body for m in again], [got[1].body])
        self.assertEqual(again[0].receive_count, 2)
        clock.t += 61                                       # visibility timeout expires -> redelivery
        redelivered = q.receive(10, 0, 60)
        self.assertEqual(len(redelivered), 10)              # 12 sent - 1 acked = 11 left; 10 per receive
        q.extend(redelivered[0], 600)                       # heartbeat keeps one hidden
        clock.t += 120
        bodies = {m.body for m in q.receive(10, 0, 60)}
        self.assertNotIn(redelivered[0].body, bodies)


class InMemoryQueueTests(QueueContract, unittest.TestCase):
    def make(self, clock):
        return InMemoryWorkQueue(clock)


class SqsQueueTests(QueueContract, unittest.TestCase):
    def make(self, clock):
        self.client = FakeSqsClient(clock)
        return SqsWorkQueue(self.client, "https://sqs/queue")

    def test_partial_batch_failure_is_retried(self):
        clock = Clock()
        q = self.make(clock)
        self.client.fail_first_batch_entry = True
        q.send_batch(["a", "b", "c"])
        self.assertEqual(len(self.client.messages), 3)

    def test_extend_on_stale_handle_is_ignored(self):
        clock = Clock()
        q = self.make(clock)
        q.send_batch(["a"])
        msg = q.receive(1, 0, 10)[0]
        q.ack(msg)
        q.extend(msg, 10)  # message already gone: must not raise


if __name__ == "__main__":
    unittest.main()
