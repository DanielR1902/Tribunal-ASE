# CLAUDE.md — `project_single_agent/`

Scoped to this package. Read alongside the root [`../CLAUDE.md`](../CLAUDE.md)
and [`../spec.md`](../spec.md) §1–2 (case/roles, architecture modes) — this
file only covers what's specific to working inside `project_single_agent/`.

## Scope

The monolithic architecture: **one** OpenRouter call is asked to
internally role-play all four advocates and all three judges, then return
the entire result as a single structured `TribunalVerdict` JSON object.
This is the baseline `project_multi_agent/`'s 7-call decomposed pipeline is
compared against.

| File | Responsibility |
|---|---|
| `main.py` | Everything: schema (`JudgeOpinion`, `TribunalVerdict`), prompt construction (`build_prompt`), CLI output formatting, and the entry point (`python -m project_single_agent.main`). |

`app.py` imports `TribunalVerdict` and `build_prompt` directly from this
module for the Streamlit "Run Single-Agent Simulation" path — it does not
duplicate the prompt or schema, so don't create a second copy of either
when working on the UI side.

## Guardrails

### 1. `max_tokens=3000` cap
- `SINGLE_AGENT_MAX_TOKENS` (`common/llm_client.py`) is `3000` — sized as
  headroom for the *entire* payload this one call must produce: 2 advocate
  summaries + 3 full judge verdicts, each with its own word limit and JSON
  structural overhead, not just one role's worth. `main()`'s
  `create_structured_completion(...)` call passes this constant, never a
  literal. If you need more budget for a prompt change, raise the constant
  in `common/llm_client.py` — don't hardcode a different number here.
- Because this is a single call covering everything, a truncation here is
  more expensive to recover from than in the multi-agent architecture (one
  retry re-does the *whole* deliberation, not just one judge). This is
  exactly why the prompt enforces strict per-field word limits (200 words
  for each summary, 120 for each judge's reasoning) — keep those limits
  in place, and keep them numerically consistent with the equivalent
  limits in `project_multi_agent/agents.py`'s prompts (root `CLAUDE.md`
  rule 5).

### 2. Complete payload validation — verdict + reasoning for all three judges, in one pass
- `TribunalVerdict` requires `prosecution_summary`, `defense_summary`, and
  all three of `judge_barak`, `judge_elon`, `judge_shamgar` (each a
  `JudgeOpinion`) — Pydantic validation fails the whole response if even
  one is missing, by design. There is no "partial success" for this
  architecture: either the model produced a complete deliberation for
  Barak (Purposive Interpretation), Elon (Jewish Law / Historic Justice),
  *and* Shamgar (Rule of Law & Institutional Authority) in a single pass,
  or the call is retried via `create_structured_completion`'s fallback —
  never silently accept a payload with a missing judge.
- Every nested `JudgeOpinion` declares `verdict` **before** `reasoning`,
  matching `project_multi_agent/agents.py`'s convention exactly, for the
  same reason: `build_schema_instructions()` embeds this field order into
  the schema shown to the model, so a judge whose `"reasoning"` gets cut
  off by the token cap still has its `verdict` already written. The
  prompt's "OUTPUT FIELD ORDER (critical)" instruction in `build_prompt()`
  reinforces this explicitly for all three judges at once — don't remove
  it when editing the prompt.
- `main()` always calls `create_structured_completion(...)`, never
  `create_chat_completion(...)` followed by a separate
  `TribunalVerdict.model_validate_json(...)` or `parse_json_response(...)`
  call. Splitting them apart silently drops the retry-on-parse-failure
  behavior (empty response, truncated response, malformed JSON on attempt
  1 all fall back to `VERIFIED_STABLE_POOL` and retry) — see
  `common/CLAUDE.md` guardrail 2 and spec.md §3.1/§4.

## Working here

- `Verdict` and `JudgeOpinion` here are independent definitions from
  `project_multi_agent/agents.py`'s (not imported/shared) — if you change
  the verdict space, the judge schema shape, or the field order in one,
  make the identical change in the other. Keeping them structurally
  duplicated-but-identical is what makes the two architectures' logged
  runs (`court_runs.db`) directly comparable (`intent.txt` — this is the
  whole point of having two architectures at all).
- Adding a new judge or advocate role? It has to exist in both
  architectures to preserve comparability: add the persona to
  `common/personas.py`, add the corresponding field to `TribunalVerdict`
  here *and* to `project_multi_agent/orchestrator.py`'s workflow, and add
  matching columns to `common/database.py`'s schema and `app.py`'s render
  code.
- `build_prompt()` has no side effects and takes no arguments — it's
  called fresh on every run (`app.py` calls it directly too). Don't give
  it hidden state; if it needs new inputs, take them as parameters.
