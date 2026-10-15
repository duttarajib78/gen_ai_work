"""Operational metrics. CloudWatch ``PutMetricData`` (batched, failure-tolerant) in AWS, no-ops elsewhere."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Protocol

log = logging.getLogger(__name__)


class MetricsSink(Protocol):
    def emit(self, name: str, value: float, unit: str = "Count") -> None: ...
    def flush(self) -> None: ...


class NullMetrics:
    def emit(self, name: str, value: float, unit: str = "Count") -> None:
        pass

    def flush(self) -> None:
        pass


class ListMetrics:
    """Collects points in memory (tests)."""

    def __init__(self) -> None:
        self.points: list[tuple[str, float, str]] = []

    def emit(self, name: str, value: float, unit: str = "Count") -> None:
        self.points.append((name, float(value), unit))

    def flush(self) -> None:
        pass

    def names(self) -> list[str]:
        return [p[0] for p in self.points]


class CloudWatchMetrics:
    """Low-cardinality metrics (one ``EndpointName`` dimension). Never raises into the worker."""

    def __init__(self, client: Any, namespace: str, endpoint_name: str, *, max_buffer: int = 5000) -> None:
        self._c, self._ns, self._endpoint, self._max = client, namespace, endpoint_name, max_buffer
        self._buf: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, name: str, value: float, unit: str = "Count") -> None:
        point = {"MetricName": name, "Value": float(value), "Unit": unit, "Timestamp": datetime.now(timezone.utc),
                 "Dimensions": [{"Name": "EndpointName", "Value": self._endpoint}]}
        with self._lock:
            if len(self._buf) < self._max:
                self._buf.append(point)

    def flush(self) -> None:
        with self._lock:
            batch, self._buf = self._buf, []
        for start in range(0, len(batch), 100):
            try:
                self._c.put_metric_data(Namespace=self._ns, MetricData=batch[start:start + 100])
            except Exception as exc:
                log.warning("metric publish failed (%s); dropping %d points", type(exc).__name__, len(batch[start:start + 100]))
