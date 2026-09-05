"""OpenRouter-backed LLM client shared by both tribunal architectures.

OpenRouter exposes an OpenAI-compatible Chat Completions API
(https://openrouter.ai/docs), so both architectures reuse the official
``openai`` Python SDK and simply repoint its ``base_url`` at OpenRouter,
authenticate with ``OPENROUTER_API_KEY``, and send the two attribution
headers OpenRouter recommends (``HTTP-Referer``, ``X-Title``).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"

# A deliberately diverse pool spanning several providers/families, used by
# the Streamlit dashboard's "Random per run" mode (see pick_random_model())
# and offered as the fixed-model choices too. ``openrouter/auto`` is
# OpenRouter's own meta-router: it picks a concrete underlying model per
# request server-side, which the response then reports back (see
# extract_usage()'s executed_model).
MODEL_POOL: list[str] = [
    "openrouter/auto",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2407",
    "google/gemini-2.5-flash",
]

# Shown to OpenRouter for attribution/analytics only; override in .env if
# you want your own app identified instead of these placeholders.
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/westeros-tribunal")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Westeros Tribunal Simulation")


def get_model() -> str:
    """Resolve the OpenRouter model id, overridable via OPENROUTER_MODEL."""
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def pick_random_model(pool: list[str] | None = None) -> str:
    """Pick one model at random from the diverse pool for a single trial run.

    Called once per Tribunal run (not once per agent call), so every advocate
    and judge within that run is asked via the same requested model — though
    if that model is ``openrouter/auto``, OpenRouter itself may still route
    individual calls to different concrete models under the hood.
    """
    return random.choice(pool or MODEL_POOL)


def get_client(api_key: str) -> OpenAI:
    """Build an OpenAI-SDK client pointed at OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        },
    )


# Ask OpenRouter to report actual per-request cost accounting in
# ``response.usage.cost`` (USD), in addition to token counts. Falls back to
# the estimated pricing table in common/cost_tracker.py when a model/provider
# combination does not return it.
USAGE_ACCOUNTING_EXTRA_BODY: dict[str, Any] = {"usage": {"include": True}}


def build_schema_instructions(schema: type[BaseModel]) -> str:
    """Render a Pydantic model as a JSON-mode instruction block.

    OpenRouter's JSON mode (``response_format={"type": "json_object"}``) only
    guarantees *valid JSON*, not conformance to a particular shape, and not
    every model routed through OpenRouter supports strict JSON-schema
    response formats. Embedding the schema directly in the prompt and
    validating the result with Pydantic on the way back out works uniformly
    across whatever model OPENROUTER_MODEL is set to.
    """
    schema_json = json.dumps(schema.model_json_schema())
    return (
        "\n\nRespond with ONLY a single valid JSON object — no markdown code "
        "fences, no commentary before or after it — that conforms exactly to "
        f"this JSON Schema:\n{schema_json}"
    )


def extract_usage(response: Any) -> tuple[int, int, float | None, str | None]:
    """Pull (prompt_tokens, completion_tokens, actual_cost_usd, executed_model)
    off a chat-completion response.

    ``actual_cost_usd`` is OpenRouter's own accounted cost for the call (see
    ``USAGE_ACCOUNTING_EXTRA_BODY``) and is ``None`` when the provider behind
    the model didn't return one, in which case callers should fall back to
    the estimated pricing table.

    ``executed_model`` is ``response.model`` — the concrete model OpenRouter
    actually ran the request on. For a concrete requested model id this just
    echoes it back; for a meta-router like ``openrouter/auto`` it is the only
    way to know which underlying model really answered, so this (not the
    requested model string) is what should be logged/displayed as the model
    "executed" for a call.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        prompt_tokens = completion_tokens = 0
        actual_cost = None
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        actual_cost = getattr(usage, "cost", None)
    executed_model = getattr(response, "model", None)
    return prompt_tokens, completion_tokens, actual_cost, executed_model
