"""The model catalog: which Bedrock identifier an alias means, what it costs, and how each model wants "thinking" to be requested.

Why this exists: Claude 5.x models on Bedrock differ in ways that break a one-size request. Some reject ``thinking.type=disabled``, some reject
budget-token thinking, none accept sampling parameters, and newer ones can only be invoked through an inference profile (``au.``, ``us.``,
``eu.``, ``global.`` + base id). The catalog is data (``data/models.json``, overridable with ``DX_MODEL_CATALOG``), so adding a model or
correcting a price never needs a code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

THINKING_STYLES = ("adaptive_effort", "budget")
OFF_MODES = ("disabled", "between_tools", "unsupported")
KNOWN_PREFIXES = ("au", "apac", "us", "eu", "jp", "global")
_BUDGETS = {"low": 1024, "medium": 4096, "high": 10000, "adaptive": 4096}      # budget-style models (1024 is the documented minimum)
_EFFORTS = ("low", "medium", "high")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    base_id: str
    price_in: float                 # USD per million input tokens (global / standard tier)
    price_out: float                # USD per million output tokens (thinking tokens are output tokens)
    thinking_style: str
    off_mode: str
    profile_prefixes: tuple[str, ...] = ()      # prefixes KNOWN to exist for this model; empty = not verified
    notes: str = ""
    known: bool = True                          # False: an explicit id the catalog does not list (priced as the default model)


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    model_id: str                   # exactly what is sent as `modelId`
    prefix: str                     # "au" | "us" | ... | "global" | "" (explicit id or in-region)
    price_multiplier: float         # geo / in-region premium over the global price
    warning: str = ""


@dataclass(frozen=True)
class Catalog:
    models: dict[str, ModelSpec]
    default_alias: str
    geo_price_multiplier: float = 1.10
    cache_read_multiplier: float = 0.10
    cache_write_multiplier: float = 1.25
    version: str = ""

    def default_spec(self) -> ModelSpec:
        return self.models[self.default_alias]

    def spec_for_base(self, base_id: str) -> Optional[ModelSpec]:
        return next((s for s in self.models.values() if s.base_id == base_id), None)


def _parse(data: dict) -> Catalog:
    models: dict[str, ModelSpec] = {}
    for alias, m in (data.get("models") or {}).items():
        try:
            spec = ModelSpec(alias=alias, base_id=str(m["base_id"]), price_in=float(m["price_in"]), price_out=float(m["price_out"]),
                             thinking_style=str(m["thinking_style"]), off_mode=str(m["off_mode"]),
                             profile_prefixes=tuple(m.get("profile_prefixes") or ()), notes=str(m.get("notes", "")))
        except (KeyError, TypeError, ValueError) as e:
            raise CatalogError(f"catalog entry {alias!r} is invalid: {e}") from None
        if spec.thinking_style not in THINKING_STYLES or spec.off_mode not in OFF_MODES:
            raise CatalogError(f"catalog entry {alias!r}: thinking_style must be one of {THINKING_STYLES}, off_mode one of {OFF_MODES}")
        if spec.price_in < 0 or spec.price_out < 0:
            raise CatalogError(f"catalog entry {alias!r}: prices must not be negative")
        models[alias] = spec
    default_alias = str(data.get("default_alias") or "")
    if default_alias not in models:
        raise CatalogError("catalog default_alias must name an entry in `models`")
    return Catalog(models=models, default_alias=default_alias, geo_price_multiplier=float(data.get("geo_price_multiplier", 1.10)),
                   cache_read_multiplier=float(data.get("cache_read_multiplier", 0.10)),
                   cache_write_multiplier=float(data.get("cache_write_multiplier", 1.25)), version=str(data.get("catalog_version", "")))


_CACHE: dict[Optional[str], Catalog] = {}


def load_catalog(path: Optional[Path] = None) -> Catalog:
    """The packaged catalog, or the one at `path` (DX_MODEL_CATALOG). Parsed once per path."""
    key = str(path) if path else None
    if key in _CACHE:
        return _CACHE[key]
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("dx_support").joinpath("data/models.json").read_text(encoding="utf-8")
    try:
        cat = _parse(json.loads(text))
    except json.JSONDecodeError as e:
        raise CatalogError(f"model catalog is not valid JSON: {e}") from None
    _CACHE[key] = cat
    return cat


def _split_prefix(model_id: str) -> tuple[str, str]:
    head, _, rest = model_id.partition(".")
    return (head, rest) if head in KNOWN_PREFIXES and rest.startswith("anthropic.") else ("", model_id)


def resolve_model(name: str, prefix: str, catalog: Catalog) -> ResolvedModel:
    """Alias -> exact Bedrock id.

      * a catalog alias ("claude-sonnet-5") gets the configured inference-profile prefix ("au" -> "au.anthropic.claude-sonnet-5");
        prefix "none" means the base id is used as-is (in-region on-demand, only valid for models that allow it);
      * an explicit Bedrock model id / inference-profile id ("global.anthropic.claude-sonnet-5-5") or ARN is used exactly as given;
      * anything else is an error: a typo must never silently become a different model.
    """
    name = (name or "").strip()
    if not name:
        raise CatalogError("model name is empty")
    if name in catalog.models:
        spec = catalog.models[name]
        pfx = "" if prefix == "none" else prefix
        model_id = f"{pfx}.{spec.base_id}" if pfx else spec.base_id
        warning = ""
        if pfx and spec.profile_prefixes and pfx not in spec.profile_prefixes:
            warning = (f"{name}: the catalog does not list an {pfx!r} inference profile for this model (known: {list(spec.profile_prefixes)}); "
                       "calls fail if it does not exist. Confirm with `aws bedrock list-inference-profiles`.")
        return ResolvedModel(spec, model_id, pfx, 1.0 if pfx == "global" else catalog.geo_price_multiplier, warning)
    if name.startswith("arn:") or "anthropic." in name:
        ident = name.rsplit("/", 1)[-1] if name.startswith("arn:") else name
        pfx, base = _split_prefix(ident)
        spec = catalog.spec_for_base(base)
        warning = ""
        if spec is None:
            dflt = catalog.default_spec()
            spec = ModelSpec(alias=ident, base_id=base, price_in=dflt.price_in, price_out=dflt.price_out, thinking_style=dflt.thinking_style,
                             off_mode=dflt.off_mode, notes="not in the catalog", known=False)
            warning = f"{ident}: not in the model catalog; priced and configured like {dflt.alias}. Add it to DX_MODEL_CATALOG."
        return ResolvedModel(spec, name, pfx, 1.0 if pfx == "global" else catalog.geo_price_multiplier, warning)
    raise CatalogError(f"unknown model {name!r}: use a catalog alias ({sorted(catalog.models)}) or an explicit Bedrock model / inference-profile id")


def thinking_fields(spec: ModelSpec, preset: str) -> tuple[dict, str]:
    """(additionalModelRequestFields, warning) for a thinking preset.

    adaptive_effort models: low / medium / high -> thinking.type=adaptive + output_config.effort; adaptive -> thinking.type=adaptive at the
    model's default effort; disabled -> whatever the model accepts as "off" (disabled, between_tools, or -- for adaptive-only models --
    adaptive at low effort, with a warning).
    budget models: low / medium / high / adaptive -> thinking.type=enabled with a token budget; disabled -> the field is omitted.
    """
    if preset not in ("disabled", "low", "medium", "high", "adaptive"):
        raise CatalogError(f"unknown thinking preset {preset!r}")
    if spec.thinking_style == "budget":
        return ({} if preset == "disabled" else {"thinking": {"type": "enabled", "budget_tokens": _BUDGETS[preset]}}), ""
    if preset in _EFFORTS:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": preset}}, ""
    if preset == "adaptive":
        return {"thinking": {"type": "adaptive"}}, ""
    if spec.off_mode == "disabled":
        return {"thinking": {"type": "disabled"}}, ""
    if spec.off_mode == "between_tools":
        return {"thinking": {"type": "between_tools"}}, ""
    return ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            f"{spec.alias} cannot turn thinking off; using adaptive thinking at low effort instead of 'disabled'.")


def min_output_tokens(spec: ModelSpec, preset: str) -> int:
    """Budget-style thinking needs max_tokens above the budget; everything else has no hard floor."""
    return _BUDGETS[preset] + 1024 if spec.thinking_style == "budget" and preset != "disabled" else 0


def cost_from_usage(resolved: ResolvedModel, usage: dict, catalog: Catalog) -> float:
    """USD for one call from the Converse `usage` block. Cache reads / writes are separate token classes priced as multiples of input."""
    spec = resolved.spec
    inp = float(usage.get("inputTokens") or 0)
    out = float(usage.get("outputTokens") or 0)
    c_read = float(usage.get("cacheReadInputTokens") or 0)
    c_write = float(usage.get("cacheWriteInputTokens") or 0)
    in_cost = (inp + c_read * catalog.cache_read_multiplier + c_write * catalog.cache_write_multiplier) * spec.price_in / 1e6
    return (in_cost + out * spec.price_out / 1e6) * resolved.price_multiplier


def worst_case_cost(resolved: ResolvedModel, input_chars: int, max_tokens: int) -> float:
    """Upper bound used to refuse a call BEFORE it is sent: estimated input (4 chars/token, +30% for the tokenizer) plus the full output cap."""
    spec = resolved.spec
    est_in = input_chars / 4.0 * 1.3
    return (est_in * spec.price_in / 1e6 + max_tokens * spec.price_out / 1e6) * resolved.price_multiplier
