# CLAUDE.md — `project_multi_agent/`

Scoped to this package. Read alongside the root [`../CLAUDE.md`](../CLAUDE.md)
and [`../spec.md`](../spec.md) §1–2 (case/roles, architecture modes) — this
file only covers what's specific to working inside `project_multi_agent/`.

## Scope

The decomposed architecture: seven independent OpenRouter calls,
coordinated by `TribunalOrchestrator`, compared against the monolithic
call in `project_single_agent/`.

| File | Responsibility |
|---|---|
| `agents.py` | `AdvocateAgent` (free-form legal brief) and `JudgeAgent` (structured `JudgeOpinion`) — one class instance per persona, each wrapping exactly one OpenRouter call. |
| `orchestrator.py` | `TribunalOrchestrator` — builds all 7 agents and drives the 3-phase workflow below. |
| `main.py` | CLI entry point (`python -m project_multi_agent.main`). |

**Workflow** (`TribunalOrchestrator.run()`):
1. Two prosecution advocates (Daenerys Targaryen, Grey Worm) run
   concurrently via `ThreadPoolExecutor`.
2. Two defense advocates (Jon Snow, Tyrion Lannister) run concurrently.
3. All three judges run concurrently against the *combined* prosecution
   and defense output from steps 1–2.

Every call's usage is recorded into the shared `CostTracker` as it
completes (`common/cost_tracker.py`), keyed `advocate:{name}` /
`judge:{name}`.

## Guardrails

### 1. Independent judge perspectives — never merge, average, or leak
- Each of the three judges — **Barak** (Purposive Interpretation),
  **Elon** (Jewish Law / Historic Justice), **Shamgar** (Rule of Law &
  Institutional Authority) — is a fully separate `JudgeAgent` instance and
  a fully separate OpenRouter call. `_run_judges()` submits all three to
  the thread pool with the *same* combined prosecution/defense text and
  nothing else; no judge's prompt or response is ever passed to another
  judge.
- Don't add any step that combines, averages, or reconciles the three
  `JudgeOpinion` results before they're returned from `orchestrator.run()`
  — `TribunalResult.judge_opinions` must stay a dict of three independent,
  possibly-disagreeing opinions. A majority-rule *display* is fine
  downstream (that's a UI concern in `app.py`); a majority-rule *verdict*
  computed here is not — it would misrepresent what actually happened.
- If you add a fourth judge or change a judge's persona/philosophy, update
  it in `common/personas.py` (`JUDGE_PERSONAS`, `JUDGE_ORDER`) — not in
  `agents.py` or `orchestrator.py`, which only consume those constants.

### 2. `max_tokens=1500` per turn — advocates and judges alike
- Both `ADVOCATE_MAX_TOKENS` and `JUDGE_MAX_TOKENS` (`common/llm_client.py`)
  are `1500`. Every `create_chat_completion` / `create_structured_completion`
  call in `agents.py` passes the matching named constant — never a literal
  number. If a call site in this package ever hardcodes `max_tokens=`,
  that's a bug: fix the constant in `common/llm_client.py` instead of
  overriding it locally.
- The judge prompt in `JudgeAgent.deliberate` caps `"reasoning"` at 120
  words and instructs the model to decide the verdict *first*, then write
  reasoning to justify it — both exist specifically so a 1500-token budget
  is comfortable headroom, not a tight squeeze. Don't raise the word limit
  without also reconsidering whether `JUDGE_MAX_TOKENS` still covers it.
- The advocate prompt caps a brief at 150–200 words in three labeled
  sections (Core Legal Claim / Supporting Agreed Facts / Concluding Plea).
  Keep advocate and judge prompts' format instructions structurally
  parallel to the single-agent architecture's equivalent instructions in
  `project_single_agent/main.py`'s `build_prompt()` — see root `CLAUDE.md`
  rule 5.

### 3. Verdict extraction standards
- `JudgeOpinion` (`agents.py`) declares `verdict` **before** `reasoning`.
  This is not cosmetic: `build_schema_instructions()` embeds this same
  field order into the schema shown to the model, so a response cut off
  by the token limit mid-`"reasoning"` still has `verdict` already
  written and parseable. Never reorder these fields, and never add a new
  structured judge/advocate schema without applying the same
  verdict-first convention.
- `JudgeAgent.deliberate` returns `(JudgeOpinion, CallResult)` via
  `create_structured_completion` — never split the create-call and the
  parse-call apart into `create_chat_completion` + a separate
  `parse_json_response` call. Doing so silently drops the fallback
  retry-on-parse-failure behavior (empty response, truncated response,
  malformed JSON on attempt 1) that `create_structured_completion`
  provides — see `common/CLAUDE.md` guardrail 2 and spec.md §3.1.
- `Verdict` is the `Literal["Justified", "Not Justified"]` type alias —
  exactly two values, no free text, no third option (e.g. "Undecided").
  If the tribunal issue ever needs a different verdict space, change this
  alias in both `project_multi_agent/agents.py` and
  `project_single_agent/main.py` together (they're currently independent
  duplicate definitions — keep them in sync).

## Working here

- Changing the number of advocates or judges? Update the persona
  dictionaries and order constants in `common/personas.py` first; both
  `orchestrator.py`'s concurrency (`ThreadPoolExecutor(max_workers=len(keys))`)
  and `TribunalResult`'s dict-shaped fields already scale to however many
  keys those constants define.
- Adding a new field to `JudgeOpinion` or the advocate output? Mirror the
  change in `project_single_agent/main.py`'s equivalent schema, and update
  `common/database.py`'s `TrialRunRecord`/`SCHEMA` plus `app.py`'s render
  code — a field that exists in only one architecture breaks the
  cross-architecture comparison this whole project exists to enable
  (`intent.txt`, `spec.md` §1).
