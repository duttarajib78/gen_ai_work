"""Run the real HTTP app locally with a stub model and stub grouper (no AWS, no network).

For checking the request/response contract and building clients. Verdicts come from a
simple policy, not from any model: every diagnosis is "clear" unless you list codes to fail.
Lives next to the test fakes and is not shipped in the image.

    python -m tools.dev_server --fail-code N179 [--port 8080]
    curl -s localhost:8080/invocations -H 'content-type: application/json' -d @tools/deploy/smoke_payload.json
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from audit_agent.engine import AuditEngine
from audit_agent.reference_data import ReferenceData
from audit_agent.serving.app import create_app
from audit_agent.service import AuditService
from tests.fakes import PolicyLlm, StubGrouper, make_reference_dir, make_settings


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fail-code", action="append", default=[], help="diagnosis code the stub model judges unsupported")
    p.add_argument("--reference-dir", type=Path, help="real reference data (default: a tiny built-in fixture)")
    a = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = a.reference_dir or make_reference_dir(Path(tmp) / "reference")
        settings = make_settings()
        reference = ReferenceData.from_directory(ref_dir)
        llm = PolicyLlm({code: ("none", 0.9) for code in a.fail_code})
        engine = AuditEngine(reference=reference, llm=llm, grouper=StubGrouper(), settings=settings)
        app = create_app(settings=settings, service=AuditService(settings=settings, engine=engine, reference=reference))
        print(f"stub server on http://127.0.0.1:{a.port}  (failing codes: {a.fail_code or 'none'})")
        app.run(host="127.0.0.1", port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
