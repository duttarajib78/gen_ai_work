"""Diagnose one episode that comes back with errors or odd verdicts.

Builds the exact prompt the service would send, calls Bedrock directly, and reports what
came back: stop reason, content block types, token usage, and whether the answer passes the
service's validation. Use it when an episode keeps failing with llm_output_invalid or
llm_output_truncated. The model's text is hidden unless --show-text is given, because it is
derived from clinical codes.

    python -m tools.diagnostics.diagnose_episode --payload episode.json --reference-dir ./reference [--show-text]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from audit_agent.config import Settings
from audit_agent.episode_text import build_episode_text
from audit_agent.llm.bedrock import BedrockClaudeClient, create_bedrock_runtime
from audit_agent.llm.verdicts import InvalidAnswer, verdicts_validator
from audit_agent.prompts import SYSTEM_PROMPT, VERDICTS_TOOL, verdicts_user_message
from audit_agent.reference_data import ReferenceData
from audit_agent.schemas import parse_request


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--max-output-tokens", type=int, help="override the output budget for this probe")
    p.add_argument("--show-text", action="store_true")
    a = p.parse_args(argv)

    settings = Settings.from_env()
    request = parse_request(json.loads(a.payload.read_text(encoding="utf-8")), settings)
    ref = ReferenceData.from_directory(a.reference_dir)
    formatted = build_episode_text(request.episode, ref, request.mbs_array_field)
    user_text = verdicts_user_message(formatted.text, formatted.pdx_code, formatted.adx_codes)
    print(f"prompt: {len(SYSTEM_PROMPT) + len(user_text)} characters (~{(len(SYSTEM_PROMPT) + len(user_text)) // 4} tokens); "
          f"{1 + len(formatted.adx_codes)} diagnosis code(s)")

    client = BedrockClaudeClient.from_settings(settings, runtime_client=create_bedrock_runtime(settings))
    req = client.build_request(system_prompt=SYSTEM_PROMPT, user_text=user_text, tool=VERDICTS_TOOL,
                               max_output_tokens=a.max_output_tokens)
    print(f"maxTokens: {req['inferenceConfig']['maxTokens']}  extra fields: {req['additionalModelRequestFields']}")
    response = create_bedrock_runtime(settings).converse(**req)

    blocks = (response.get("output") or {}).get("message", {}).get("content", [])
    print(f"stopReason: {response.get('stopReason')}")
    print(f"usage: {response.get('usage')}  latencyMs: {(response.get('metrics') or {}).get('latencyMs')}")
    print(f"content blocks: {[next(iter(b)) for b in blocks]}")
    if response.get("stopReason") == "max_tokens":
        print("-> the answer was cut off: thinking plus the tool call exceeded maxTokens. "
              "Raise AUDIT_LLM_MAX_OUTPUT_TOKENS (or lower AUDIT_LLM_EFFORT).")
    calls = [b["toolUse"] for b in blocks if "toolUse" in b]
    if not calls:
        print("-> no tool call: the model answered in text. The service would retry with feedback.")
        if a.show_text:
            print([b.get("text") for b in blocks if "text" in b])
        return 1
    args = calls[0].get("input")
    try:
        verdicts = verdicts_validator(formatted.pdx_code, formatted.adx_codes)(args if isinstance(args, dict) else {})
        print(f"-> valid answer with {len(verdicts)} verdict(s): "
              f"{[(v.code, v.support_level, round(v.confidence, 2)) for v in verdicts]}" if a.show_text else
              f"-> valid answer with {len(verdicts)} verdict(s)")
        return 0
    except InvalidAnswer as exc:
        print(f"-> would be rejected ({exc.category}): {exc}" if a.show_text else f"-> would be rejected ({exc.category})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
