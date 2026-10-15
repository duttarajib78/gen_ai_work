"""Client for the SageMaker real-time endpoint: batching, retries (botocore standard mode) and bounded
concurrency. Requires ``boto3`` (an ops dependency, not needed inside the endpoint container)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from drg_audit.errors import DrgAuditError
from drg_audit.schemas import InpatientRecord

MAX_SERVER_RECORDS = 256


def record_to_payload(record: InpatientRecord) -> dict[str, Any]:
    return {
        "admission_no": record.admission_no,
        "diagnosis_code_array": list(record.diagnosis_code_array),
        "procedure_code_array": list(record.procedure_code_array),
        "medical_claims_array": list(record.medical_claims_array),
        "drg_vers": record.drg_vers,
        "drg": record.drg,
        "los": int(record.los),
        "minutes_operating_theatre": float(record.minutes_operating_theatre),
        "hours_of_mechanical_ventilation": float(record.hours_of_mechanical_ventilation),
        "member_age": float(record.member_age),
        "drg_benefit": float(record.drg_benefit),
    }


class EndpointClient:
    def __init__(
        self,
        endpoint_name: str,
        region_name: str | None = None,
        *,
        timeout_seconds: int = 60,
        max_attempts: int = 5,
        runtime_client: Any = None,
    ) -> None:
        if runtime_client is None:
            import boto3
            from botocore.config import Config

            runtime_client = boto3.client(
                "sagemaker-runtime",
                region_name=region_name,
                config=Config(retries={"max_attempts": max_attempts, "mode": "standard"},
                              read_timeout=timeout_seconds, connect_timeout=5),
            )
        self.endpoint_name = endpoint_name
        self._client = runtime_client

    def score(self, records: Sequence[dict[str, Any]], confidence_threshold: float | None = None) -> dict[str, Any]:
        if not 0 < len(records) <= MAX_SERVER_RECORDS:
            raise DrgAuditError(f"a request must contain between 1 and {MAX_SERVER_RECORDS} records")
        payload: dict[str, Any] = {"records": list(records)}
        if confidence_threshold is not None:
            payload["options"] = {"confidence_threshold": confidence_threshold}
        response = self._client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload, allow_nan=False).encode("utf-8"),
        )
        return json.loads(response["Body"].read())

    def score_stream(
        self,
        records: Iterable[dict[str, Any]],
        *,
        batch_size: int = 128,
        max_workers: int = 4,
        confidence_threshold: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield per-record results in input order. Concurrency is bounded to avoid throttling the endpoint."""
        if not 1 <= batch_size <= MAX_SERVER_RECORDS:
            raise ValueError(f"batch_size must be between 1 and {MAX_SERVER_RECORDS}")
        items = list(records)
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for response in pool.map(lambda b: self.score(b, confidence_threshold), batches):
                yield from response["results"]
