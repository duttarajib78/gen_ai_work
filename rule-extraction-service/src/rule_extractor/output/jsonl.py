from __future__ import annotations

import json
from typing import Any, Iterable, Iterator


def dumps_line(record: Any) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_jsonl_bytes(records: Iterable[Any]) -> bytes:
    return "".join(dumps_line(r) + "\n" for r in records).encode("utf-8")


def iter_jsonl(data: bytes | str) -> Iterator[Any]:
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
