"""Hand-written fakes that mimic the subset of the boto3 client API used by the AWS adapters."""
from __future__ import annotations

import io
import threading
import uuid


class FakeClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeDynamoClient:
    def __init__(self, page_size: int = 1000):
        self.items: dict[tuple[str, str], dict] = {}
        self.lock = threading.Lock()
        self.page_size = page_size
        self.unprocessed_once = False
        self.calls: list[str] = []

    def get_item(self, TableName, Key, ConsistentRead=False):
        with self.lock:
            self.calls.append("get_item")
            item = self.items.get((Key["PK"]["S"], Key["SK"]["S"]))
            return {"Item": dict(item)} if item else {}

    def put_item(self, TableName, Item, ConditionExpression=None, ExpressionAttributeNames=None, ExpressionAttributeValues=None):
        with self.lock:
            self.calls.append("put_item")
            key = (Item["PK"]["S"], Item["SK"]["S"])
            existing = self.items.get(key)
            if ConditionExpression == "attribute_not_exists(PK)":
                if existing is not None:
                    raise FakeClientError("ConditionalCheckFailedException")
            elif ConditionExpression == "#v = :v":
                expected = ExpressionAttributeValues[":v"]["N"]
                if existing is None or existing["version"]["N"] != expected:
                    raise FakeClientError("ConditionalCheckFailedException")
            elif ConditionExpression is not None:
                raise AssertionError(f"unsupported condition in fake: {ConditionExpression}")
            self.items[key] = dict(Item)
            return {}

    def batch_write_item(self, RequestItems):
        with self.lock:
            self.calls.append("batch_write_item")
            (table, requests), = RequestItems.items()
            assert len(requests) <= 25
            if self.unprocessed_once and len(requests) > 1:
                self.unprocessed_once = False
                half = len(requests) // 2
                for r in requests[:half]:
                    item = r["PutRequest"]["Item"]
                    self.items[(item["PK"]["S"], item["SK"]["S"])] = dict(item)
                return {"UnprocessedItems": {table: requests[half:]}}
            for r in requests:
                item = r["PutRequest"]["Item"]
                self.items[(item["PK"]["S"], item["SK"]["S"])] = dict(item)
            return {"UnprocessedItems": {}}

    def query(self, TableName, KeyConditionExpression, ExpressionAttributeValues, IndexName=None, ConsistentRead=False,
              Limit=None, ExclusiveStartKey=None):
        with self.lock:
            self.calls.append("query")
            vals = {k: v["S"] for k, v in ExpressionAttributeValues.items()}
            if IndexName:
                rows = sorted((i for i in self.items.values() if i.get("gsi1pk", {}).get("S") == vals[":p"]),
                              key=lambda i: i["gsi1sk"]["S"])
            else:
                assert KeyConditionExpression == "PK = :pk AND begins_with(SK, :sk)"
                rows = sorted((i for (pk, sk), i in self.items.items() if pk == vals[":pk"] and sk.startswith(vals[":sk"])),
                              key=lambda i: i["SK"]["S"])
            start = int(ExclusiveStartKey["n"]["N"]) if ExclusiveStartKey else 0
            size = min(self.page_size, Limit or self.page_size)
            page = rows[start:start + size]
            out = {"Items": [dict(i) for i in page]}
            if start + size < len(rows) and not Limit:
                out["LastEvaluatedKey"] = {"n": {"N": str(start + size)}}
            return out


class FakeS3Client:
    def __init__(self, page_size: int = 1000):
        self.objects: dict[tuple[str, str], dict] = {}
        self.page_size = page_size

    def put_object(self, Bucket, Key, Body, ContentType=None, **extra):
        self.objects[(Bucket, Key)] = {"Body": bytes(Body), "ContentType": ContentType, "extra": extra}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)]["Body"])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        keys = sorted(k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        out = {"Contents": [{"Key": k} for k in page], "IsTruncated": start + self.page_size < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + self.page_size)
        return out


class FakeSqsClient:
    def __init__(self, clock):
        self.clock = clock
        self.messages: dict[str, dict] = {}
        self.fail_first_batch_entry = False

    def send_message_batch(self, QueueUrl, Entries):
        assert len(Entries) <= 10
        failed = []
        for e in Entries:
            if self.fail_first_batch_entry and e["Id"] == "0":
                self.fail_first_batch_entry = False
                failed.append({"Id": e["Id"], "Code": "ServiceUnavailable"})
                continue
            self.messages[uuid.uuid4().hex] = {"body": e["MessageBody"], "visible_at": 0.0, "receipt": None, "receives": 0}
        return {"Failed": failed} if failed else {}

    def receive_message(self, QueueUrl, MaxNumberOfMessages, WaitTimeSeconds, VisibilityTimeout, AttributeNames):
        assert 1 <= MaxNumberOfMessages <= 10 and 0 <= WaitTimeSeconds <= 20
        out, now = [], self.clock()
        for mid, m in self.messages.items():
            if m["visible_at"] <= now and len(out) < MaxNumberOfMessages:
                m["visible_at"], m["receipt"] = now + VisibilityTimeout, uuid.uuid4().hex
                m["receives"] += 1
                out.append({"MessageId": mid, "ReceiptHandle": m["receipt"], "Body": m["body"],
                            "Attributes": {"ApproximateReceiveCount": str(m["receives"])}})
        return {"Messages": out} if out else {}

    def _find(self, receipt):
        for mid, m in self.messages.items():
            if m["receipt"] == receipt:
                return mid, m
        return None, None

    def delete_message(self, QueueUrl, ReceiptHandle):
        mid, _ = self._find(ReceiptHandle)
        if mid:
            del self.messages[mid]

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        _, m = self._find(ReceiptHandle)
        if m is None:
            raise FakeClientError("ReceiptHandleIsInvalid")
        m["visible_at"] = self.clock() + VisibilityTimeout

    def get_queue_attributes(self, QueueUrl, AttributeNames):
        now = self.clock()
        visible = sum(1 for m in self.messages.values() if m["visible_at"] <= now)
        return {"Attributes": {"ApproximateNumberOfMessages": str(visible),
                               "ApproximateNumberOfMessagesNotVisible": str(len(self.messages) - visible)}}


# ----------------------------------------------------------------------------- agent runner fake
import asyncio as _asyncio
import inspect as _inspect

from rule_extractor.agents.types import AgentRunStats


def default_rule_args(unit_id: str, i: int = 0, logic: str | None = None) -> dict:
    return dict(
        rule_name=f"Rule from {unit_id} #{i}",
        rule_logic_plain_english=logic or f"Logic for {unit_id} #{i}: A41.0 coded with B95.6",
        upcoding_mechanism="Adds a comorbidity-like code with no independent meaning",
        recommended_action="remove_code", source_reference="icd10am_tabular printed p.27; acs printed p.101",
        rule_shape="redundancy", icd10am_codes="A41.0, B95.6", materiality="high",
        flag_condition={"all": [{"has_diagnosis": "A41.0"}, {"has_diagnosis": "B95.6"}]},
        correction_action={"action": "remove_code", "target": "diagnosis", "match": "B95.6"})


def emit_default(task, count: int = 1, logic: str | None = None) -> AgentRunStats:
    handler = task.tools[0].handler
    for i in range(count):
        result = handler(default_rule_args(task.unit_id, i, None if logic is None else (logic if i == 0 else f"{logic} #{i}")))
        assert not result.is_error, result.text
    return AgentRunStats(turns=3, cost_usd=0.05, usage={"input_tokens": 100, "output_tokens": 20}, duration_s=1.0)


class FakeRunner:
    """Implements the AgentRunner protocol. ``script(task, attempt_no)`` may return stats or raise."""

    def __init__(self, script=None):
        self.script = script
        self.tasks = []
        self.attempts: dict[str, int] = {}

    async def run(self, task):
        self.tasks.append(task)
        n = self.attempts[task.unit_id] = self.attempts.get(task.unit_id, 0) + 1
        if self.script is None:
            return emit_default(task)
        out = self.script(task, n)
        if _inspect.isawaitable(out):
            out = await out
        return out
