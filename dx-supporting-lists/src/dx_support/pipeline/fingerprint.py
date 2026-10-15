"""The cache fingerprint: results are only reused when EVERYTHING that determines them is unchanged.

New reference data, an edited prompt, a different model or thinking setting, or a changed tunable all produce a new fingerprint, so a cached
verdict can never be served for a configuration that did not produce it. The service keeps its results under `serve/<fingerprint>/`; the batch
CLI stores the fingerprint in the run's `run_config.json` and refuses to continue a run whose fingerprint has changed.
"""
from __future__ import annotations

import hashlib
import json

from dx_support.indexing.universe import Universe
from dx_support.llm.catalog import load_catalog, resolve_model
from dx_support.pipeline import constants as K
from dx_support.pipeline import prompts
from dx_support.settings import Settings

PIPELINE_VERSION = "2.0"


def tunables() -> dict:
    return {k: getattr(K, k) for k in ("JUDGE_BATCH_SIZE", "ROUTE_MAX_NODES_PER_CALL", "LEXICAL_TOP_K", "MAX_UNITS_PER_LEAF_NODE", "MAX_UNITS_PER_FAMILY",
                                       "JUDGE_RETRY_BATCHES")}


def resolved_models(settings: Settings, backend: str) -> dict[str, str]:
    """The exact model identifiers in force (what Bedrock is asked for), so an alias that changes meaning changes the fingerprint."""
    models = settings.stage_models()
    if backend != "bedrock":
        return models
    catalog = load_catalog(settings.model_catalog_path)
    return {stage: resolve_model(name, settings.inference_profile_prefix, catalog).model_id for stage, name in models.items()}


def run_fingerprint(settings: Settings, universe: Universe, *, mode: str = "routed", backend: str = "bedrock") -> str:
    payload = {
        "pipeline": PIPELINE_VERSION, "prompts": prompts.prompt_digest(), "tunables": tunables(), "data": universe.digest, "mode": mode,
        "models": resolved_models(settings, backend), "thinking": settings.thinking, "lexical_rescue": settings.lexical_rescue,
        "coarse_prostheses": universe.include_coarse_prostheses, "backend": backend, "salt": settings.cache_salt,
        "max_tokens": dict(settings.max_tokens),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
