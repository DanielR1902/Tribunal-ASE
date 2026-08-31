"""Token usage and cost tracking for Gemini API calls.

Pricing is Google's published pay-as-you-go per-million-token rates for the
Gemini API (https://ai.google.dev/gemini-api/docs/pricing). Rates change
over time and can vary by account/region, so treat the figures below as an
estimate, not a bill. Override ``GEMINI_MODEL`` / ``ILS_PER_USD`` in your
``.env`` if you need different assumptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# USD price per 1,000,000 tokens. Models with a "tiers" entry charge a
# higher rate once the prompt crosses the given token threshold (Gemini 2.5
# Pro's long-context tier); models without "tiers" charge a flat rate.
PRICING_USD_PER_MILLION_TOKENS: dict[str, dict] = {
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "tiers": {
            "threshold_tokens": 200_000,
            "input_above": 2.50,
            "output_above": 15.00,
        },
    },
}

# Fallback used for any model name not found above (e.g. a future model the
# user points GEMINI_MODEL at). Kept deliberately close to Flash pricing.
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
    """Token usage and cost for a single Gemini API call."""

    label: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates token usage and cost across one or more Gemini calls.

    The single-agent architecture calls ``record`` once; the multi-agent
    architecture calls it once per agent (7 times) and the tracker sums
    everything for the final budget report.
    """

    model: str
    calls: list[CallUsage] = field(default_factory=list)

    def record(self, label: str, prompt_tokens: int, completion_tokens: int) -> CallUsage:
        """Record one API call's usage and return its individual cost breakdown."""
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        cost_usd = estimate_cost_usd(self.model, prompt_tokens, completion_tokens)
        usage = CallUsage(
            label=label,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
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
