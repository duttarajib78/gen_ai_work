"""AWS implementations: DynamoDB (state), S3 (artifacts), SQS (work queue).

Design notes
* Low-level boto3 *clients* only (thread-safe), no resources, no ``Decimal`` handling.
* DynamoDB items store each record as one JSON string plus a numeric ``version``; every update is a
  compare-and-swap on ``version`` (``ConditionExpression``), so correctness never relies on expression subtleties.
* Single table: ``PK``/``SK`` keys, sparse GSI ``gsi1`` (job status index), TTL attribute ``ttl``.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .. import __version__
from ..domain.jobs import JobRecord, UnitRecord
from ..errors import ConflictError, NotFoundError
from ..settings import Settings
from ..util import validate_object_key, validate_object_prefix
from .interfaces import Backends, QueueMessage

GSI_NAME = "gsi1"


def error_code(exc: BaseException) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


class DynamoStateStore:
    def __init__(self, client: Any, table: str, *, retention_days: int = 90, clock: Callable[[], float] = time.time,
                 max_cas_attempts: int = 12) -> None:
        self._c, self._table, self._clock = client, table, clock
        self._ttl_seconds = retention_days * 86400
        self._max_cas = max_cas_attempts

    # ------------------------------------------------------------------ low level
    def _item(self, pk: str, sk: str, data: dict, version: int, *, gsi: tuple[str, str] | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "PK": {"S": pk}, "SK": {"S": sk}, "version": {"N": str(version)},
            "data": {"S": json.dumps(data, separators=(",", ":"))},
            "ttl": {"N": str(int(self._clock() + self._ttl_seconds))},
        }
        if gsi:
            item["gsi1pk"], item["gsi1sk"] = {"S": gsi[0]}, {"S": gsi[1]}
        return item

    def _get(self, pk: str, sk: str) -> tuple[dict | None, int]:
        resp = self._c.get_item(TableName=self._table, Key={"PK": {"S": pk}, "SK": {"S": sk}}, ConsistentRead=True)
        item = resp.get("Item")
        if not item:
            return None, 0
        return json.loads(item["data"]["S"]), int(item["version"]["N"])

    def _put_new(self, item: dict[str, Any]) -> bool:
        try:
            self._c.put_item(TableName=self._table, Item=item, ConditionExpression="attribute_not_exists(PK)")
            return True
        except Exception as exc:
            if error_code(exc) == "ConditionalCheckFailedException":
                return False
            raise

    def _cas(self, pk: str, sk: str, loader: Callable[[dict], Any], mutate: Callable[[Any], bool],
             gsi: Callable[[Any], tuple[str, str] | None] | None = None) -> tuple[bool, Any]:
        for attempt in range(self._max_cas):
            data, version = self._get(pk, sk)
            if data is None:
                return False, None
            record = loader(data)
            if not mutate(record):
                return False, record
            item = self._item(pk, sk, record.to_dict(), version + 1, gsi=gsi(record) if gsi else None)
            try:
                self._c.put_item(TableName=self._table, Item=item, ConditionExpression="#v = :v",
                                 ExpressionAttributeNames={"#v": "version"},
                                 ExpressionAttributeValues={":v": {"N": str(version)}})
                return True, record
            except Exception as exc:
                if error_code(exc) != "ConditionalCheckFailedException":
                    raise
                time.sleep(min(0.5, 0.02 * (2 ** attempt)) * random.random())
        raise ConflictError(f"too much contention updating {pk}/{sk}")

    @staticmethod
    def _job_gsi(job: JobRecord) -> tuple[str, str]:
        return f"JOBSTATUS#{job.status}", job.created_at

    # ------------------------------------------------------------------- protocol
    def reserve_idempotency(self, key: str, job_id: str, options_digest: str) -> tuple[str, str] | None:
        pk = f"IDEM#{key}"
        if self._put_new(self._item(pk, "META", {"job_id": job_id, "options_digest": options_digest}, 1)):
            return None
        data, _ = self._get(pk, "META")
        return (data["job_id"], data["options_digest"]) if data else None

    def put_job(self, job: JobRecord) -> None:
        if not self._put_new(self._item(f"JOB#{job.job_id}", "META", job.to_dict(), 1, gsi=self._job_gsi(job))):
            raise ConflictError(f"job {job.job_id} already exists")

    def get_job(self, job_id: str) -> JobRecord | None:
        data, _ = self._get(f"JOB#{job_id}", "META")
        return JobRecord.from_dict(data) if data else None

    def update_job(self, job_id: str, mutate: Callable[[JobRecord], bool]) -> tuple[bool, JobRecord | None]:
        return self._cas(f"JOB#{job_id}", "META", JobRecord.from_dict, mutate, self._job_gsi)

    def put_units(self, units: Sequence[UnitRecord]) -> None:
        pending = [{"PutRequest": {"Item": self._item(f"JOB#{u.job_id}", f"UNIT#{u.unit_id}", u.to_dict(), 1)}} for u in units]
        for start in range(0, len(pending), 25):
            batch = pending[start:start + 25]
            for attempt in range(8):
                resp = self._c.batch_write_item(RequestItems={self._table: batch})
                batch = resp.get("UnprocessedItems", {}).get(self._table, [])
                if not batch:
                    break
                time.sleep(min(2.0, 0.05 * (2 ** attempt)) * (0.5 + random.random()))
            else:
                raise RuntimeError("DynamoDB batch write left unprocessed items")

    def get_unit(self, job_id: str, unit_id: str) -> UnitRecord | None:
        data, _ = self._get(f"JOB#{job_id}", f"UNIT#{unit_id}")
        return UnitRecord.from_dict(data) if data else None

    def update_unit(self, job_id: str, unit_id: str, mutate: Callable[[UnitRecord], bool]) -> tuple[bool, UnitRecord | None]:
        return self._cas(f"JOB#{job_id}", f"UNIT#{unit_id}", UnitRecord.from_dict, mutate)

    def list_units(self, job_id: str) -> list[UnitRecord]:
        units: list[UnitRecord] = []
        kwargs: dict[str, Any] = dict(
            TableName=self._table, ConsistentRead=True,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
            ExpressionAttributeValues={":pk": {"S": f"JOB#{job_id}"}, ":sk": {"S": "UNIT#"}})
        while True:
            resp = self._c.query(**kwargs)
            units.extend(UnitRecord.from_dict(json.loads(i["data"]["S"])) for i in resp.get("Items", []))
            if not resp.get("LastEvaluatedKey"):
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return sorted(units, key=lambda u: (u.order, u.unit_id))

    def list_jobs_by_status(self, status: str, limit: int = 100) -> list[JobRecord]:
        resp = self._c.query(TableName=self._table, IndexName=GSI_NAME, Limit=limit,
                             KeyConditionExpression="gsi1pk = :p",
                             ExpressionAttributeValues={":p": {"S": f"JOBSTATUS#{status}"}})
        return [JobRecord.from_dict(json.loads(i["data"]["S"])) for i in resp.get("Items", [])]


class S3ArtifactStore:
    def __init__(self, client: Any, bucket: str, prefix: str = "", kms_key_id: str | None = None) -> None:
        self._c, self._bucket, self._prefix, self._kms = client, bucket, prefix.strip("/"), kms_key_id

    def _key(self, key: str) -> str:
        validate_object_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def _sse(self) -> dict[str, str]:
        return {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self._kms} if self._kms else {}

    def put_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        self._c.put_object(Bucket=self._bucket, Key=self._key(key), Body=data, ContentType=content_type, **self._sse())
        return self.uri(key)

    def put_file(self, key: str, path: Path, *, content_type: str = "application/octet-stream") -> str:
        return self.put_bytes(key, Path(path).read_bytes(), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._c.get_object(Bucket=self._bucket, Key=self._key(key))["Body"].read()
        except Exception as exc:
            if error_code(exc) in ("NoSuchKey", "404"):
                raise NotFoundError(f"artifact not found: {key}") from exc
            raise

    def exists(self, key: str) -> bool:
        try:
            self._c.head_object(Bucket=self._bucket, Key=self._key(key))
            return True
        except Exception as exc:
            if error_code(exc) in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        validate_object_prefix(prefix)
        full_prefix = f"{self._prefix}/{prefix}" if self._prefix else prefix
        strip = len(self._prefix) + 1 if self._prefix else 0
        keys: list[str] = []
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": full_prefix}
        while True:
            resp = self._c.list_objects_v2(**kwargs)
            keys.extend(o["Key"][strip:] for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        return sorted(keys)

    def uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{self._key(key)}"


class SqsWorkQueue:
    def __init__(self, client: Any, queue_url: str) -> None:
        self._c, self._url = client, queue_url

    def send_batch(self, bodies: Sequence[str]) -> None:
        for start in range(0, len(bodies), 10):
            entries = [{"Id": str(i), "MessageBody": b} for i, b in enumerate(bodies[start:start + 10])]
            for attempt in range(5):
                resp = self._c.send_message_batch(QueueUrl=self._url, Entries=entries)
                failed = {f["Id"] for f in resp.get("Failed", [])}
                if not failed:
                    break
                entries = [e for e in entries if e["Id"] in failed]
                time.sleep(min(2.0, 0.1 * (2 ** attempt)))
            else:
                raise RuntimeError(f"SQS rejected {len(entries)} message(s) after retries")

    def receive(self, max_messages: int, wait_seconds: int, visibility_timeout: int) -> list[QueueMessage]:
        resp = self._c.receive_message(
            QueueUrl=self._url, MaxNumberOfMessages=max(1, min(10, max_messages)),
            WaitTimeSeconds=max(0, min(20, wait_seconds)), VisibilityTimeout=visibility_timeout,
            AttributeNames=["ApproximateReceiveCount"])
        return [QueueMessage(handle=m["ReceiptHandle"], body=m["Body"], message_id=m.get("MessageId", ""),
                             receive_count=int(m.get("Attributes", {}).get("ApproximateReceiveCount", "1")))
                for m in resp.get("Messages", [])]

    def ack(self, message: QueueMessage) -> None:
        self._c.delete_message(QueueUrl=self._url, ReceiptHandle=message.handle)

    def extend(self, message: QueueMessage, visibility_timeout: int) -> None:
        try:
            self._c.change_message_visibility(QueueUrl=self._url, ReceiptHandle=message.handle,
                                              VisibilityTimeout=max(0, min(43200, visibility_timeout)))
        except Exception as exc:
            if error_code(exc) in ("ReceiptHandleIsInvalid", "MessageNotInflight", "InvalidParameterValue"):
                return  # message already redelivered/deleted: nothing to extend
            raise

    def depth(self) -> tuple[int, int]:
        attrs = self._c.get_queue_attributes(
            QueueUrl=self._url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
        return int(attrs.get("ApproximateNumberOfMessages", 0)), int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))


def build_aws_backends(settings: Settings) -> Backends:
    """Create thread-safe boto3 clients using the default credential chain (the endpoint's execution role)."""
    settings.validate_for_aws_backend()
    import boto3
    from botocore.config import Config

    base = dict(region_name=settings.bedrock.region, user_agent_extra=f"rule-extractor/{__version__}",
                retries={"max_attempts": 10, "mode": "adaptive"})
    ddb = boto3.client("dynamodb", config=Config(connect_timeout=3, read_timeout=10, **base))
    s3 = boto3.client("s3", config=Config(connect_timeout=5, read_timeout=60, **base))
    sqs = boto3.client("sqs", config=Config(connect_timeout=3, read_timeout=30, **base))
    return Backends(
        state=DynamoStateStore(ddb, settings.aws.state_table or "", retention_days=settings.service.job_retention_days),
        artifacts=S3ArtifactStore(s3, settings.aws.artifact_bucket or "", settings.aws.artifact_prefix, settings.aws.kms_key_id),
        queue=SqsWorkQueue(sqs, settings.aws.work_queue_url or ""),
    )
