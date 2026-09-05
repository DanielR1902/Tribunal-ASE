# SPEC — Step 2: System Specifications

Derived directly from [`intent.txt`](./intent.txt). Where this document
gives a concrete value (a constant, a column name, a token cap) it reflects
the current implementation; if the code and this file ever disagree, treat
that as a spec drift bug to fix, not a reason to trust one over the other
silently.

## 1. Case & Roles (fixed, canonical)

**Case T-001: The Realm v. Jon Snow.** Jon Snow stands accused of the
intentional killing of Daenerys Targaryen. The agreed facts, the charge,
and the tribunal issue are the single source of truth in
`common/case_data.py` — every architecture and every role reasons over
exactly this record, no exceptions.

- **Advocates** (`common/personas.py`): two per side.
  - Prosecution: Daenerys Targaryen, Grey Worm.
  - Defense: Jon Snow, Tyrion Lannister.
- **Judges** — three independent decision-makers, each applying a
  distinct jurisprudential philosophy to the *same* record and the *same*
  tribunal issue. Their verdicts are never merged, averaged, or forced
  into agreement:
  - **Judge Barak** — Purposive Interpretation. Reads a situation not just
    by its literal facts but through the legal values and institutional
    purposes at stake, applying strict four-fold scrutiny (lawful
    authority, proper purpose, rational connection, proportionality).
  - **Judge Elon** — Jewish Law / Historic Justice. Treats law as an
    inherited, ongoing conversation grounded in traditional Jewish legal
    concepts (the pursuer/*rodef*, duty to save a life, human dignity),
    with judicial modesty against turning open-textured standards into a
    license for personal political judgment.
  - **Judge Shamgar** — Rule of Law & Institutional Authority. Insists
    that public ends require formal, lawful means; fact-heavy and
    chronology-driven, distinguishing strictly between lawful institutional
    process and extrajudicial private action.

## 2. Architecture Modes

### Single-Agent (`project_single_agent/main.py`)
One monolithic OpenRouter call plays all four advocates and all three
judges at once, returning a single structured JSON payload
(`TribunalVerdict`): `prosecution_summary`, `defense_summary`, and three
`{verdict, reasoning}` judge objects. `verdict` is declared *before*
`reasoning` in the schema so a verdict already exists even if the
response gets cut off before the reasoning text finishes.

### Multi-Agent (`project_multi_agent/`)
Seven independent OpenRouter calls, orchestrated by
`TribunalOrchestrator`:
1. Two prosecution advocates run concurrently (free-form legal brief, no
   JSON).
2. Two defense advocates run concurrently.
3. All three judges run concurrently and independently against the
   combined prosecution/defense briefs, each returning a structured
   `JudgeOpinion` (`{verdict, reasoning}`, same field order rationale as
   above).

Both architectures log to the identical database schema (Section 5) so
they are directly comparable on cost, tokens, timing, and verdicts.

## 3. OpenRouter Integration

All completion calls route through `common/llm_client.py`, never the
OpenAI SDK's default endpoint — `base_url` is repointed at
`https://openrouter.ai/api/v1`.

### 3.1 Two-tier resilience model
- **Tier 1 — dynamic pool** (`MODEL_POOL`): the primary attempt, led by
  `"openrouter/auto"` (OpenRouter's own meta-router, which resolves to a
  concrete underlying model server-side per request) plus a diverse spread
  of providers: `meta-llama/llama-3.3-70b-instruct`,
  `mistralai/mistral-large-2407`, `google/gemini-2.5-flash`,
  `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`,
  `deepseek/deepseek-chat`.
- **Tier 2 — verified-stable pool** (`VERIFIED_STABLE_POOL`): plain,
  undated, confirmed-routable slugs only — no meta-router, no
  experimental entries: `google/gemini-2.5-flash`, `openai/gpt-4o-mini`,
  `meta-llama/llama-3.3-70b-instruct`, `mistralai/mistral-large-2407`,
  `anthropic/claude-3.5-haiku`.
- **Fallback trigger**: any of `NotFoundError` (404 — invalid/unroutable
  model), `RateLimitError` (429), `APITimeoutError`, or
  `APIConnectionError`. `AuthenticationError` / `BadRequestError` are
  deliberately **not** retried — no model swap fixes a bad key or a
  malformed request, so those surface immediately.
- **Sequential degradation**: on a retryable failure, `create_chat_completion`
  walks the verified-stable pool in order (skipping whichever model just
  failed) until one succeeds, or raises the last error if every candidate
  fails.
- **Parse-aware fallback**: `create_structured_completion` extends the
  same walk to cover a call that answers successfully but returns JSON
  that fails to parse/validate — a failure mode invisible to API-level
  error handling alone.
- **Accurate reporting**: `response.model` (not the originally requested
  model string) is what's read back, tracked, and persisted as the
  "executed" model for a call — critical when the primary request was
  `"openrouter/auto"`, or when a fallback occurred.

### 3.2 Dynamic engine toggle (UI)
The Streamlit sidebar offers two modes:
- **Random per run** (default): a model is drawn at random from
  `MODEL_POOL` fresh on every Run click. The dropdown/list of candidate
  engines is **hidden entirely** in this mode — the user is shown only a
  plain caption, never the pool contents, until after the run completes
  and the resolved engine is reported in the results.
- **Fixed**: the standard dropdown is shown, listing every `MODEL_POOL`
  entry, and the chosen model is used for every call in that run.

### 3.3 Token & timeout safety
| Constant                    | Value | Applies to                                    |
|------------------------------|-------|------------------------------------------------|
| `SINGLE_AGENT_MAX_TOKENS`    | 3000  | The one monolithic Single-Agent call            |
| `JUDGE_MAX_TOKENS`           | 1500  | Each of the 3 judge calls (Multi-Agent)         |
| `ADVOCATE_MAX_TOKENS`        | 1500  | Each of the 4 advocate calls (Multi-Agent)      |
| `REQUEST_TIMEOUT_SECONDS`    | 90    | Every OpenRouter request, primary or fallback   |

Every JSON-mode prompt also requires RFC 8259–compliant output: no
markdown fences, no commentary outside the JSON object, escaped
line breaks/quotes inside string values, straight quotes only. Judge
prompts additionally cap `reasoning` at 120 words and require `verdict`
as the first JSON field specifically so a truncated response
(`finish_reason="length"`) still preserves the verdict.

## 4. Malformed/Truncated Output Recovery

`parse_json_response` (`common/llm_client.py`) is the last line of defense
against non-compliant model output, tried in escalating order:
1. Strip markdown fences.
2. Extract the outermost `{...}` block (drops stray commentary).
3. Drop trailing commas.
4. Escape unescaped inner quotes (heuristic: a `"` is only a real string
   terminator if followed by a JSON structural character).
5. Normalize smart/curly quotes to straight ones.
6. Close an unterminated string and balance any unclosed
   `{`/`[` (recovers a `finish_reason="length"` truncation), reapplying
   step 3 afterward.

Parsing is done via `json.loads(..., strict=False)` (tolerates raw control
characters inside strings) followed by `schema.model_validate(...)` — never
Pydantic's own stricter `model_validate_json`. If every repair candidate
still fails, a `ValueError` is raised that explicitly names truncation as
the likely cause when `finish_reason == "length"`, rather than surfacing a
cryptic parser error.

## 5. Persistence — `court_runs.db`

Single SQLite table, `trial_runs`, written once per completed run
(`common/database.py`), shared by both architectures and tagged by
`architecture_mode` so they can be queried and compared directly:

| Column                     | Type    | Notes                                            |
|------------------------------|---------|---------------------------------------------------|
| `id`                        | INTEGER | Primary key, autoincrement                        |
| `timestamp`                 | TEXT    | UTC ISO-8601, set at insert time                  |
| `architecture_mode`         | TEXT    | `"single_agent"` \| `"multi_agent"`               |
| `model`                     | TEXT    | The resolved/executed engine(s) — see §3.1        |
| `prosecution_summary`       | TEXT    |                                                    |
| `defense_summary`           | TEXT    |                                                    |
| `judge_barak_reasoning`     | TEXT    |                                                    |
| `judge_barak_verdict`       | TEXT    | `"Justified"` \| `"Not Justified"`                |
| `judge_elon_reasoning`      | TEXT    |                                                    |
| `judge_elon_verdict`        | TEXT    |                                                    |
| `judge_shamgar_reasoning`   | TEXT    |                                                    |
| `judge_shamgar_verdict`     | TEXT    |                                                    |
| `prompt_tokens`             | INTEGER | Summed across all calls in the run                |
| `completion_tokens`         | INTEGER |                                                    |
| `total_tokens`              | INTEGER |                                                    |
| `cost_usd`                  | REAL    | OpenRouter-reported actual cost where available, else estimated |
| `cost_ils`                  | REAL    | `cost_usd * ILS_PER_USD`                          |
| `execution_time_sec`        | REAL    | Wall-clock for the whole run                      |

`init_db()` is idempotent and self-migrating: it creates the table if
absent and adds any missing column (currently `model`) to a pre-existing
database via `ALTER TABLE`, so upgrading the code never requires manually
touching or discarding an existing `court_runs.db`.

## 6. UI/UX Constraints

- **No franchise-specific place names.** "Westeros" and equivalent
  franchise-proper-nouns must not appear anywhere — UI copy, prompts,
  titles, docstrings, or config defaults. Use neutral/institutional
  language instead: "the Realm," "the tribunal," "the case."
- **State-locking during a run.** While a simulation is executing
  (`st.session_state.is_running`), the UI must not allow the user to
  trigger a second run, switch the engine-selection mode, or navigate to
  the Historical Runs view — those controls are disabled or the view is
  hidden entirely until the run completes (`is_running` is reset in a
  `finally` block so this never gets stuck on even if the run raises).
- **Engine visibility is mode-dependent** (§3.2): Random mode never shows
  a list of candidate engines pre-run; Fixed mode always shows the full
  dropdown.
- **Post-run reporting is explicit.** After a run completes, the exact
  engine that answered is surfaced in the results view and in Historical
  Runs — this is a deliberate exception to keeping the engine hidden
  pre-run, since the point of hiding it beforehand is to keep engine
  choice out of the user's hands mid-configuration, not to hide the audit
  trail after the fact.
