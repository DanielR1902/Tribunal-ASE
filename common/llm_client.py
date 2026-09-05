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
import re
import warnings
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The dynamic/primary pool: a broad, experimental spread of options,
# ``openrouter/auto`` first as the primary dynamic choice — OpenRouter's own
# meta-router, which picks a concrete underlying model per request
# server-side (see extract_usage()'s executed_model for how to see which one
# actually answered). Used by the Streamlit dashboard's "Random per run"
# mode (see pick_random_model()) and offered as the fixed-model choices too.
# Not every entry here is guaranteed to answer on every request — that's
# what VERIFIED_STABLE_POOL below is for.
MODEL_POOL: list[str] = [
    "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2407",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-chat",
]
DEFAULT_MODEL = MODEL_POOL[0]  # "openrouter/auto"

# The second tier of the resilient-fallback design: plain, undated,
# confirmed-stable model slugs — no meta-router, no experimental/newer
# entries — used by create_chat_completion() when the primary attempt (an
# entry from MODEL_POOL, or whatever a user set OPENROUTER_MODEL to) hits a
# 404, rate limit, timeout, or connection error. Deliberately a separate,
# narrower list from MODEL_POOL: MODEL_POOL optimizes for dynamic variety,
# this one optimizes for "this will answer."
VERIFIED_STABLE_POOL: list[str] = [
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2407",
    "anthropic/claude-3.5-haiku",
]

# Every OpenRouter request gets this long to respond before it's treated as
# failed and handed to the fallback logic below — long enough for a judge's
# ~1500-token completion on a slower model, short enough that a genuinely
# stuck request doesn't hang the simulation indefinitely.
REQUEST_TIMEOUT_SECONDS = 90

# Failure modes that trigger create_chat_completion()'s fallback: an invalid/
# unroutable model id (404), the provider throttling this key (429), the
# request exceeding REQUEST_TIMEOUT_SECONDS, or a network-level failure
# reaching OpenRouter at all. Deliberately NOT catching e.g.
# AuthenticationError or BadRequestError — those indicate a problem no model
# swap will fix (bad API key, malformed request), so they should still
# surface immediately rather than burning through the whole fallback pool.
RETRYABLE_API_ERRORS: tuple[type[Exception], ...] = (
    NotFoundError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)

# Shown to OpenRouter for attribution/analytics only; override in .env if
# you want your own app identified instead of these placeholders.
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://github.com/tribunal-simulation")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Tribunal Simulation")

# Per-call completion token caps, enforced on every OpenRouter chat call so
# briefs/opinions stay short and rapid regardless of which model in
# MODEL_POOL answers — without these, some models will happily produce
# multi-page wall-of-text generations for what should be a tight legal brief.
# Both advocates and judges get the same headroom: their JSON/text payload
# has to fit comfortably within the cap with room to spare, since cutting a
# judge off mid-string (finish_reason="length") produces truncated,
# unparseable JSON, not just a shorter opinion.
ADVOCATE_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 1500
# The single-agent architecture makes ONE call that must produce everything
# two advocate calls and three judge calls would separately (2 summaries +
# 3 full judge verdicts) inside a single JSON object — around 5 fields of
# prose plus JSON structural overhead. Too low a cap here truncates the
# response mid-field (typically partway through a later judge's reasoning),
# which is exactly what produces invalid/incomplete JSON and a parse
# failure — so this needs real headroom, not just the per-role caps above.
SINGLE_AGENT_MAX_TOKENS = 3000


def get_model() -> str:
    """Resolve the OpenRouter model id, overridable via OPENROUTER_MODEL."""
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def pick_random_model(pool: list[str] | None = None) -> str:
    """Pick one model at random from the diverse pool for a single trial run.

    Called once per Tribunal run (not once per agent call), so every advocate
    and judge within that run is asked via the same requested model.
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


def _fallback_candidates(primary_model: str, pool: list[str] | None) -> list[str]:
    """VERIFIED_STABLE_POOL (or a caller-supplied pool), minus whichever
    model just failed — no point retrying the one that already didn't work."""
    stable_pool = pool if pool is not None else VERIFIED_STABLE_POOL
    return [m for m in stable_pool if m != primary_model]


def create_chat_completion(
    client: OpenAI,
    *,
    model: str,
    fallback_pool: list[str] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> Any:
    """``client.chat.completions.create(model=model, timeout=timeout, **kwargs)``,
    with resilient two-tier fallback.

    Primary attempt: whatever ``model`` was requested — a MODEL_POOL entry
    (including the ``openrouter/auto`` meta-router), a manual
    ``OPENROUTER_MODEL`` override, or anything else. If that raises a 404
    (invalid/unroutable model), a 429 (rate limited), a timeout, or a
    connection error (see RETRYABLE_API_ERRORS), this catches it and retries
    against each model in ``fallback_pool`` (default: VERIFIED_STABLE_POOL,
    minus ``model`` itself) in order, returning on the first one that
    succeeds. Only if every candidate fails does the last error propagate.

    Every chat-completion call in this project — advocates, judges, and the
    single-agent monolithic call — goes through this instead of calling the
    SDK directly, so an unreliable/invalid model id degrades to a working
    one instead of raising and crashing the Streamlit app or a CLI run.
    """
    try:
        return client.chat.completions.create(model=model, timeout=timeout, **kwargs)
    except RETRYABLE_API_ERRORS as exc:
        last_error: Exception = exc
        last_model = model
        for fallback_model in _fallback_candidates(model, fallback_pool):
            warnings.warn(
                f"OpenRouter model {last_model!r} failed "
                f"({type(last_error).__name__}: {last_error}); falling back to "
                f"{fallback_model!r} for this call.",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                return client.chat.completions.create(
                    model=fallback_model, timeout=timeout, **kwargs
                )
            except RETRYABLE_API_ERRORS as fallback_exc:
                last_error = fallback_exc
                last_model = fallback_model
                continue
        raise last_error


def create_structured_completion(
    client: OpenAI,
    *,
    model: str,
    schema: type[ModelT],
    fallback_pool: list[str] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> tuple[ModelT, Any]:
    """Like ``create_chat_completion``, but for a JSON-mode call whose reply
    must validate against ``schema`` — and extends the same two-tier
    fallback to cover a *parsing* failure too, not just an API-level one.

    A model can answer successfully (no 404/timeout/rate limit) and still
    return garbled JSON that ``parse_json_response`` can't recover — that's
    not something ``create_chat_completion`` alone can catch, since from the
    API's perspective the call succeeded. This tries the full
    call-then-parse cycle against ``model`` first, then each remaining
    candidate in ``fallback_pool`` (default: VERIFIED_STABLE_POOL, minus
    ``model``) in order, until one both answers AND parses. Returns
    ``(parsed_model, raw_response)`` so callers can still pull
    usage/cost/executed-model off ``raw_response`` exactly as with a plain
    ``create_chat_completion`` call.
    """
    candidates = [model] + _fallback_candidates(model, fallback_pool)
    last_error: Exception | None = None
    for attempt, candidate in enumerate(candidates):
        if attempt > 0:
            warnings.warn(
                f"Retrying with model {candidate!r} after the previous "
                f"attempt failed ({type(last_error).__name__}: {last_error}).",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            response = client.chat.completions.create(
                model=candidate, timeout=timeout, **kwargs
            )
        except RETRYABLE_API_ERRORS as exc:
            last_error = exc
            continue
        content = response.choices[0].message.content or ""
        try:
            parsed = parse_json_response(
                content, schema, finish_reason=extract_finish_reason(response)
            )
        except ValueError as exc:
            last_error = exc
            continue
        return parsed, response

    assert last_error is not None  # candidates is never empty (model + pool)
    raise last_error


# Ask OpenRouter to report actual per-request cost accounting in
# ``response.usage.cost`` (USD), in addition to token counts. Falls back to
# the estimated pricing table in common/cost_tracker.py when a model/provider
# combination does not return it.
USAGE_ACCOUNTING_EXTRA_BODY: dict[str, Any] = {"usage": {"include": True}}

# Requested on every call that expects a structured (JudgeOpinion /
# TribunalVerdict) reply. OpenRouter forwards this to the underlying
# provider and, for a model/provider pair that doesn't support strict JSON
# mode, simply passes it through as a no-op rather than erroring — so it's
# safe to request unconditionally rather than trying to pre-detect per-model
# support (OpenRouter doesn't expose a cheap way to check that per request).
JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


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
        "\n\nOutput ONLY raw RFC 8259 compliant JSON — a single JSON object, "
        "no markdown code fences, no commentary before or after it — that "
        f"conforms exactly to this JSON Schema:\n{schema_json}\n\n"
        "JSON FORMATTING RULES (critical): every string value must itself be "
        "valid JSON. Escape any line break inside a string as \\n and any "
        "double quote inside a string as \\\" — do not write unescaped "
        "quotes inside string values, and never place a raw/literal "
        "newline, tab, or other control character inside a string value. "
        "Use straight double quotes (\") for all JSON keys and string "
        "delimiters, never curly/smart quotes. Keep every string field "
        "within its stated length limit so the JSON is never cut off "
        "mid-value."
    )


def _strip_markdown_fence(text: str) -> str | None:
    """Pull the JSON body out of a ```json ... ``` or ``` ... ``` fence."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def _extract_outer_braces(text: str) -> str | None:
    """Grab the outermost {...} block, in case the model added stray text
    (a preamble, an apology, trailing commentary) around the JSON object."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def _drop_trailing_commas(text: str) -> str:
    """Remove a trailing comma right before a closing ``}``/``]`` — a common
    small mistake some models make that otherwise breaks JSON parsing."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


_SMART_QUOTES = {
    "“": '"', "”": '"',  # “ ”
    "‘": "'", "’": "'",  # ‘ ’
}


def _normalize_smart_quotes(text: str) -> str:
    """Last-resort repair for a model that used typographic quotes as JSON
    structural delimiters (e.g. {“key”: “value”}) instead of straight ones."""
    for smart, straight in _SMART_QUOTES.items():
        text = text.replace(smart, straight)
    return text


def _from_first_brace(text: str) -> str | None:
    """Everything from the first ``{`` onward — unlike ``_extract_outer_braces``
    this does NOT require a closing ``}`` to already be present, so it also
    covers a response that got cut off mid-generation before any closing
    brace was ever emitted."""
    idx = text.find("{")
    return text[idx:] if idx != -1 else None


def _escape_unescaped_inner_quotes(text: str) -> str:
    """Heuristic repair for a model that wrote a raw, unescaped ``"`` inside
    a string value (e.g. ``"reasoning": "He said "no" here."``), which
    otherwise ends the string early and leaves the parser expecting a ``,``
    or ``}`` where it instead finds more text.

    Walks the text tracking string state; a ``"`` encountered while inside a
    string is treated as legitimately closing that string only if the next
    non-whitespace character is a JSON structural character (``,`` ``:``
    ``}`` ``]``) or end-of-text — otherwise it's re-escaped as an embedded
    quote the model forgot to escape.
    """
    result: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < n:
                result.append(ch)
                result.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ",:}]":
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')
                i += 1
                continue
            result.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = True
        result.append(ch)
        i += 1
    return "".join(result)


def _attempt_bracket_repair(text: str) -> str | None:
    """Best-effort repair for JSON truncated mid-generation
    (``finish_reason="length"``): closes any string left open at the end,
    then appends whatever closing ``}``/``]`` are needed to balance the
    structure, in the correct nested order.

    Returns ``None`` if the text wasn't actually left unbalanced (nothing to
    repair), so callers can skip adding a redundant candidate.
    """
    in_string = False
    escape = False
    stack: list[str] = []
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if not in_string and not stack:
        return None  # already balanced — no repair needed

    repaired = text
    if in_string:
        repaired += '"'
    repaired += "".join(reversed(stack))
    return repaired


def parse_json_response(
    text: str, schema: type[ModelT], finish_reason: str | None = None
) -> ModelT:
    """Robustly parse a model's JSON-mode reply into a validated Pydantic model.

    Some models routed through OpenRouter don't reliably emit clean JSON:
    they wrap it in markdown code fences, add stray commentary before/after
    it, leave a trailing comma, use smart quotes, embed an unescaped quote
    inside a string value, emit literal unescaped control characters (raw
    newlines) inside a string value, or get cut off mid-generation
    (``finish_reason="length"``) leaving an unterminated string and
    unbalanced braces. Both Python's default strict JSON parser and
    Pydantic's own ``model_validate_json`` reject the control-character case
    outright with "control character (\\u0000-\\u001F) found while parsing a
    string", and none of them recover from truncation or stray quotes.

    This tries a series of increasingly aggressive candidate repairs of the
    raw text, parsing each with ``json.loads(..., strict=False)`` (which
    tolerates raw control characters inside strings) and validating the
    resulting dict with ``schema.model_validate(...)`` rather than handing
    raw text to Pydantic's stricter JSON parser.
    """
    stripped = text.strip()

    raw_candidates: list[str] = [stripped]

    fenced = _strip_markdown_fence(stripped)
    if fenced:
        raw_candidates.append(fenced)

    braces = _extract_outer_braces(stripped)
    if braces:
        raw_candidates.append(braces)

    # Run the same repair chain (drop trailing commas, escape stray inner
    # quotes, normalize smart quotes, close unterminated strings/brackets)
    # against TWO different bases, since each handles a different failure
    # shape:
    #   - `braces`: the first-`{`-to-last-`}` span. Already excludes any
    #     trailing garbage after the object (a closing markdown fence,
    #     trailing commentary), so this is the right base when the object
    #     itself closed but has an internal defect like a trailing comma.
    #   - `_from_first_brace(stripped)`: first-`{`-to-end-of-text. `braces`'s
    #     regex requires an actual `}` and silently drops any incomplete
    #     tail after the LAST `}` it can find — exactly the content needed
    #     to recover a response that got cut off mid-generation
    #     (finish_reason="length") before any closing brace for the final
    #     field was ever emitted.
    # Trying both means a defect-plus-truncation combination, or a defect
    # with trailing garbage after it, are each still recoverable.
    bases = [b for b in (braces, _from_first_brace(stripped)) if b is not None]
    for base in dict.fromkeys(bases):  # de-dupe while preserving order
        no_trailing_commas = _drop_trailing_commas(base)
        raw_candidates.append(no_trailing_commas)

        quotes_escaped = _escape_unescaped_inner_quotes(no_trailing_commas)
        raw_candidates.append(quotes_escaped)

        smart_quotes_fixed = _normalize_smart_quotes(quotes_escaped)
        raw_candidates.append(smart_quotes_fixed)

        # Truncation repair, tried against both the quote-escaped candidate
        # and the plainer comma-cleaned one (in case quote-escaping over-
        # corrected something), each also with trailing commas re-cleaned
        # since closing an unterminated string can newly expose one.
        for candidate in (smart_quotes_fixed, no_trailing_commas):
            repaired = _attempt_bracket_repair(candidate)
            if repaired is not None:
                raw_candidates.append(_drop_trailing_commas(repaired))

    # De-duplicate while preserving attempt order.
    seen: set[str] = set()
    candidates = []
    for candidate in raw_candidates:
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            last_error = exc
            continue

    snippet = stripped[:300] + ("…" if len(stripped) > 300 else "")
    truncation_note = (
        " The response was cut off by the token limit (finish_reason="
        f"{finish_reason!r}) before it finished — raise the relevant "
        "max_tokens cap if this keeps happening."
        if finish_reason == "length"
        else ""
    )
    raise ValueError(
        f"Could not parse a valid {schema.__name__} from the model's response "
        f"after {len(candidates)} attempt(s). Last error: {last_error}."
        f"{truncation_note} Raw response started with: {snippet!r}"
    ) from last_error


def extract_usage(response: Any) -> tuple[int, int, float | None, str | None]:
    """Pull (prompt_tokens, completion_tokens, actual_cost_usd, executed_model)
    off a chat-completion response.

    ``actual_cost_usd`` is OpenRouter's own accounted cost for the call (see
    ``USAGE_ACCOUNTING_EXTRA_BODY``) and is ``None`` when the provider behind
    the model didn't return one, in which case callers should fall back to
    the estimated pricing table.

    ``executed_model`` is ``response.model`` — the concrete model OpenRouter
    actually ran the request on. Normally this just echoes back the
    requested model id, but it's also what a fallback retry (see
    ``create_chat_completion``) reports as actually having answered, so this
    (not the originally requested model string) is what should be
    logged/displayed as the model "executed" for a call.
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


def extract_finish_reason(response: Any) -> str | None:
    """Pull ``choices[0].finish_reason`` off a chat-completion response.

    ``"length"`` means the model was cut off by ``max_tokens`` before it
    finished — the response is very likely truncated, invalid JSON. Pass
    this straight into ``parse_json_response`` so a resulting parse failure
    can say so explicitly instead of leaving the cause a mystery.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)
