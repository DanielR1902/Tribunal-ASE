# CLAUDE.md — `common/`

Scoped to this package. Read alongside the root [`../CLAUDE.md`](../CLAUDE.md)
(global architecture, Streamlit UI, root commands) and
[`../spec.md`](../spec.md) §3–5 (OpenRouter integration, malformed/truncated
output recovery, persistence) — this file only covers what's specific to
working inside `common/`.

## Scope

`common/` is the shared foundation both architectures build on. Nothing in
here knows about Streamlit, `st.session_state`, or which architecture is
calling it — that separation is load-bearing, not incidental (see
Guardrails below). Five modules:

| File | Responsibility |
|---|---|
| `case_data.py` | Canonical case file — the single source of truth for the tribunal's facts, charge, and issue. |
| `personas.py` | Advocate and judge persona/system prompts, plus the ordering constants (`ADVOCATE_ORDER`, `PROSECUTION_KEYS`, `DEFENSE_KEYS`, `JUDGE_ORDER`) both architectures iterate over. |
| `llm_client.py` | **OpenRouter API client resilience.** Model pools, token/timeout constants, the two-tier fallback (`create_chat_completion`, `create_structured_completion`), and the JSON repair chain (`parse_json_response`). |
| `cost_tracker.py` | **Cost tracking.** `CostTracker`/`CallUsage` accumulate token counts and cost (actual, from OpenRouter, when reported; otherwise `PRICING_USD_PER_MILLION_TOKENS` as a fallback estimate) across one or more calls in a run. |
| `database.py` | **DB models.** SQLite schema for `court_runs.db`, self-migrating `init_db()`, `TrialRunRecord`, `log_trial_run()`, `fetch_run()`. |

## Guardrails

### 1. Strict RFC 8259 JSON parsing
- Every JSON-mode prompt is built via `build_schema_instructions(schema)`,
  never hand-written inline. It embeds the target Pydantic model's
  `model_json_schema()` and explicitly instructs: raw JSON only (no
  fences, no commentary), escaped newlines/quotes inside string values,
  straight quotes only, and — critically — the schema's own field order,
  which is why `verdict` is declared before `reasoning` on every judge
  schema (truncation-safety; see `project_multi_agent/CLAUDE.md` and
  `project_single_agent/CLAUDE.md`).
- Parse every structured reply through `parse_json_response(text, schema,
  finish_reason=...)`, never `schema.model_validate_json(text)` directly.
  `model_validate_json` uses Pydantic's stricter parser, which rejects
  raw control characters outright — `parse_json_response` uses
  `json.loads(..., strict=False)` plus a repair chain (strip markdown
  fences → extract outer braces → drop trailing commas → escape
  unescaped inner quotes → normalize smart quotes → close an unterminated
  string and rebalance brackets) specifically so it survives the ways
  real models violate RFC 8259.
- If you find a new malformed-JSON shape a model produces, add a repair
  step to this chain in `llm_client.py` — don't work around it at the
  call site, and don't loosen a call site to skip `parse_json_response`.

### 2. Two-tier model fallback rules
- **Never call `client.chat.completions.create(...)` directly, from
  anywhere in this package or outside it.** Free-form (non-JSON) calls go
  through `create_chat_completion`; JSON-mode calls needing a validated
  Pydantic result go through `create_structured_completion`. Both apply
  `REQUEST_TIMEOUT_SECONDS` (90s) and the same fallback logic; the
  structured variant additionally retries the whole call-then-parse cycle
  if the response answers successfully but fails to parse (a failure
  invisible to API-level error handling alone).
- **Tier 1 (primary):** whatever `model` the caller passed — a
  `MODEL_POOL` entry (including `"openrouter/auto"`), or a manual
  `OPENROUTER_MODEL` override.
- **Tier 2 (fallback):** `VERIFIED_STABLE_POOL` — plain, undated,
  confirmed-routable slugs only. `_fallback_candidates()` walks it in
  order (skipping whatever just failed) until one succeeds.
- **Trigger set is deliberate — don't widen or narrow it casually.**
  `RETRYABLE_API_ERRORS` = `NotFoundError` (404), `RateLimitError` (429),
  `APITimeoutError`, `APIConnectionError`, PLUS (for the structured
  variant) a `ValueError` from `parse_json_response`. `AuthenticationError`
  / `BadRequestError` are intentionally excluded — no model swap fixes a
  bad key or a malformed request, so those must keep surfacing
  immediately rather than burning through the fallback pool.
- **`openrouter/auto` never appears in `VERIFIED_STABLE_POOL` or in the
  Fixed-mode UI dropdown.** It's a meta-router, not a specific engine a
  user can deliberately pick, and it's not something you'd want a
  *fallback* to land on (the whole point of tier 2 is a known-concrete
  model). Adding it to either would violate spec.md §3.2/§6.
- If you add a new call site anywhere in the repo, wire it through one of
  these two functions. Don't write a new try/except around a raw SDK
  call — that's exactly the drift this package exists to prevent.

### 3. No direct UI dependencies
- `common/` must not import `streamlit`, read `st.session_state`, or
  otherwise assume a Streamlit run context. Both `app.py` and the two
  `project_*/main.py` CLI entry points depend on `common/`; `common/`
  depends on neither. If a function needs to behave differently for the
  UI vs. the CLI, that branching belongs in the caller, not here.
- Similarly, don't hardcode CLI-only concerns (e.g. `print()` for
  progress) into `common/` — `case_data.py`, `personas.py`,
  `llm_client.py`, `cost_tracker.py`, and `database.py` are pure
  logic/data modules. Human-readable output formatting lives in
  `app.py` or the `project_*/main.py` files that already do it.

## Working here

- Adding a model to the pool? Update `MODEL_POOL` (dynamic) or
  `VERIFIED_STABLE_POOL` (fallback-eligible) in `llm_client.py`, then add
  a matching entry to `PRICING_USD_PER_MILLION_TOKENS` in
  `cost_tracker.py` so cost estimation doesn't silently fall back to
  `DEFAULT_PRICING` for it.
- Changing a token cap or the timeout? It's a single constant in
  `llm_client.py` (`ADVOCATE_MAX_TOKENS`, `JUDGE_MAX_TOKENS`,
  `SINGLE_AGENT_MAX_TOKENS`, `REQUEST_TIMEOUT_SECONDS`) — never a literal
  at a call site.
- Changing the DB schema? Add the column to `SCHEMA` in `database.py`
  *and* add an `ALTER TABLE` step to `_ensure_model_column`-style
  migration logic so an existing `court_runs.db` upgrades in place. See
  root `CLAUDE.md` rule 4 for the full preserve-database-state policy.
