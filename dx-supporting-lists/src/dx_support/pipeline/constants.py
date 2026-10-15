"""Pipeline tunables. Changing any of them changes results, so they are part of the cache fingerprint (see fingerprint.py)."""
from __future__ import annotations

from dx_support.constants import DEFAULT_FAMILIES

# Judge batch sizes differ by family because unit text length does (MBS items are long; ACHI codes short; prosthesis labels in between).
JUDGE_BATCH_SIZE = {"achi": 150, "mbs": 100, "prosthesis": 100, "dx": 150}
ROUTE_MAX_NODES_PER_CALL = 80
LEXICAL_TOP_K = {"achi": 40, "mbs": 40, "prosthesis": 25, "dx": 25}
MAX_UNITS_PER_LEAF_NODE = 150            # a branch bigger than this is ranked by BM25 and cut (flagged, never silent)
MAX_UNITS_PER_FAMILY = {"achi": 600, "mbs": 600, "prosthesis": 400, "dx": 300}   # circuit breaker on one category's judge workload
MAX_RATIONALE_CHARS = 240
VOCAB_FAMILIES = DEFAULT_FAMILIES        # the brief's vocabulary hint covers the code tables only, never the diagnosis family
# A long answer can skip many candidates (seen: 66-88 of ~100 prosthesis device types in five categories -- and when re-judged, 70% of them were
# core or broad, so skipped units are NOT safe to assume irrelevant). Re-sending the whole skipped list in one call tends to skip them again,
# so skipped units are re-judged in SMALLER batches -- first 25, then 10 -- each with an explicit note, before the category is left `partial`.
JUDGE_RETRY_BATCHES = (25, 10)
DEFAULT_CATEGORY_CONCURRENCY = 8

# Rough cost-model constants (see projection.py). Point-in-time, order-of-magnitude; `dx-support estimate --calibrate` replaces them with measurements.
CHARS_PER_TOKEN = 4.0
TOKENIZER_ADJUST = 1.3                   # Claude 4.7+ tokenizers produce ~30% more tokens for the same text
EST_OUTPUT_TOKENS_PER_VERDICT = 50
EST_THINKING_TOKENS = {"disabled": 0, "low": 800, "medium": 2500, "high": 6000, "adaptive": 4000}
SECONDS_PER_CALL_ESTIMATE = 45           # PROVISIONAL: measured 42-45 s per call with the earlier Agent SDK build; not yet measured on Bedrock
# The earlier build carried ~20,000 input tokens of wrapper overhead per call (the Claude Code process). Bedrock Converse has none.
CALL_OVERHEAD_INPUT_TOKENS = 0
ASSUMED_SELECT_FRACTION = (0.10, 0.40)   # share of a routing frontier the model keeps (low, high)
COST_UNCERTAINTY = (0.7, 1.5)
