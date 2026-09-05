"""Token usage and cost tracking for OpenRouter API calls.

OpenRouter can report the actual per-request cost it billed in
``response.usage.cost`` (see ``common.llm_client.extract_usage``); when a
call carries that figure, ``CostTracker.record`` uses it directly. The
pricing table below is only a fallback estimate for the rare case where a
model/provider combination doesn't return usage-based cost accounting — kept
roughly aligned with each model's published list price at the time this
project was written. Rates change over time and can vary by account/region/
provider routing, so treat the fallback figures as an estimate, not a bill.
Override ``OPENROUTER_MODEL`` / ``ILS_PER_USD`` in your ``.env`` if you need
different assumptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# USD price per 1,000,000 tokens, keyed by OpenRouter model id. Models with a
# "tiers" entry charge a higher rate once the prompt crosses the given token
# threshold (Gemini 2.5 Pro's long-context tier); models without "tiers"
# charge a flat rate. Only used when OpenRouter doesn't report actual cost.
# Covers both the original Gemini defaults and the diverse random-selection
# pool in common/llm_client.py's MODEL_POOL. ``openrouter/auto`` has no
# single price (it routes to a different concrete model per request) so it
# intentionally has no entry here and falls back to DEFAULT_PRICING.
PRICING_USD_PER_MILLION_TOKENS: dict[str, dict] = {
    "google/gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
    "google/gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
    },
    "google/gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "tiers": {
            "threshold_tokens": 200_000,
            "input_above": 2.50,
            "output_above": 15.00,
        },
    },
    "anthropic/claude-3.5-sonnet": {
        "input": 3.00,
        "output": 15.00,
    },
    "openai/gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "input": 0.12,
        "output": 0.30,
    },
    "mistralai/mistral-large-2407": {
        "input": 2.00,
        "output": 6.00,
    },
}

# Fallback used for any model name not found above (e.g. openrouter/auto, or
# a future model the user points OPENROUTER_MODEL at). Kept deliberately
# close to Gemini Flash pricing.
DEFAULT_PRICING = {"input": 0.30, "output": 2.50}

DEFAULT_ILS_PER_USD = 3.65


def get_ils_exchange_rate() -> float:
    """USD -> ILS exchange rate, overridable via the ILS_PER_USD env var."""
    try:
        return float(os.getenv("ILS_PER_USD", DEFAULT_ILS_PER_USD))
    except ValueError:
        return DEFAULT_ILS_PER_USD


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for one call given its token counts."""
    pricing = PRICING_USD_PER_MILLION_TOKENS.get(model, DEFAULT_PRICING)

    input_rate = pricing["input"]
    output_rate = pricing["output"]

    tiers = pricing.get("tiers")
    if tiers and prompt_tokens > tiers["threshold_tokens"]:
        input_rate = tiers["input_above"]
        output_rate = tiers["output_above"]

    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


@dataclass
class CallUsage:
    """Token usage and cost for a single OpenRouter API call."""

    label: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    executed_model: str = ""


@dataclass
class CostTracker:
    """Accumulates token usage and cost across one or more OpenRouter calls.

    The single-agent architecture calls ``record`` once; the multi-agent
    architecture calls it once per agent (7 times) and the tracker sums
    everything for the final budget report.
    """

    model: str
    calls: list[CallUsage] = field(default_factory=list)

    def record(
        self,
        label: str,
        prompt_tokens: int,
        completion_tokens: int,
        actual_cost_usd: float | None = None,
        executed_model: str | None = None,
    ) -> CallUsage:
        """Record one API call's usage and return its individual cost breakdown.

        Pass ``actual_cost_usd`` (from ``common.llm_client.extract_usage``)
        when OpenRouter reported real cost accounting for this call; it is
        used as-is instead of the estimated pricing table. Pass
        ``executed_model`` (also from ``extract_usage``) when known — the
        concrete model that actually served this call, which can differ from
        ``self.model`` (the requested model) when that's a meta-router like
        ``openrouter/auto``.
        """
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        cost_usd = (
            float(actual_cost_usd)
            if actual_cost_usd is not None
            else estimate_cost_usd(self.model, prompt_tokens, completion_tokens)
        )
        usage = CallUsage(
            label=label,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            executed_model=executed_model or self.model,
        )
        self.calls.append(usage)
        return usage

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def cost_ils(self) -> float:
        return self.cost_usd * get_ils_exchange_rate()

    @property
    def executed_model_summary(self) -> str:
        """The exact model(s) OpenRouter actually executed this run's calls on.

        A single concrete requested model normally resolves to itself for
        every call. ``openrouter/auto`` (or "Random per run" landing on it)
        can resolve different calls to different underlying models, in which
        case this reports every distinct one — see the per-call breakdown for
        which call ran on which model.
        """
        executed = sorted({c.executed_model for c in self.calls if c.executed_model})
        if not executed:
            return self.model
        if len(executed) == 1:
            return executed[0]
        return ", ".join(executed)
