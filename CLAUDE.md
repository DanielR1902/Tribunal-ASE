# CLAUDE.md — Step 3: Agentic Operational Manual

Operational conventions for working in this repository. Read alongside
[`intent.txt`](./intent.txt) (why this project exists) and
[`spec.md`](./spec.md) (what it must do) — this file covers *how* to work
in the code day to day.

## Directory Blueprint

The conceptual roles below map onto real files in this repo — there is no
separate `agents/` or `storage/` package; those responsibilities live
inside `project_multi_agent/` and `common/` respectively.

```
tribunal/
├── app.py                        # Streamlit GUI — entry point for interactive use
├── intent.txt                    # Step 1 — why this project exists
├── spec.md                       # Step 2 — what it must do
├── CLAUDE.md                     # Step 3 — this file
├── README.md                     # Human-facing setup/usage docs
├── requirements.txt
├── .env                          # OPENROUTER_API_KEY and overrides (never commit real keys)
├── court_runs.db                 # SQLite persistence — see spec.md §5
│
├── common/                       # Shared foundation for both architectures
│   ├── case_data.py              #   Canonical case file — single source of truth for facts
│   ├── personas.py               #   Advocate + judge persona/system prompts
│   ├── llm_client.py             #   OpenRouter client, model pools, fallback, JSON repair (spec.md §3-4)
│   ├── cost_tracker.py           #   Token usage -> USD/ILS cost accounting
│   └── database.py               #   ("storage/") SQLite schema, migration, read/write
│
├── project_single_agent/
│   └── main.py                   # ("agents/" — monolithic) one call plays every role
│
└── project_multi_agent/          # ("agents/" — decomposed)
    ├── agents.py                 #   AdvocateAgent, JudgeAgent classes
    ├── orchestrator.py           #   TribunalOrchestrator — coordinates the 7-call run
    └── main.py                   #   CLI entry point
```

**Single source of truth, don't duplicate:**
- Case facts → `common/case_data.py` only.
- Persona/system prompts → `common/personas.py` only.
- Model pools, token caps, timeout, fallback logic, JSON repair →
  `common/llm_client.py` only. If you need a new constant that both
  architectures should share, it belongs here, not duplicated in
  `app.py` and `project_*/main.py`.
- DB schema and migrations → `common/database.py` only.

## Runtime Commands

**Environment setup** (create once per checkout):

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# POSIX (bash/zsh)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run the app:**

```bash
python -m streamlit run app.py
```

**Run a single architecture from the CLI** (bypasses the Streamlit UI):

```bash
python -m project_single_agent.main
python -m project_multi_agent.main
```

**Verify before considering any change done.** These three checks are
cheap, fast, and catch the two failure classes that have recurred in this
project's history (stale bytecode masking a real fix, and a broken import
after refactoring a shared module) — run all three, not just one:

```bash
# 1. Syntax-check every file (clears __pycache__ first to rule out stale bytecode)
find . -type d -name "__pycache__" -exec rm -rf {} +   # or: Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse -Force
python -m py_compile app.py common/*.py project_multi_agent/*.py project_single_agent/*.py

# 2. Import the actual entry point end-to-end
python -c "import app"

# 3. Catch unused imports / undefined names (pip install pyflakes if missing)
python -m pyflakes app.py common/*.py project_multi_agent/*.py project_single_agent/*.py
```

All three must exit `0` with no output (pyflakes) / no traceback (the
other two) before a change is considered complete.

## Engineering Rules

1. **RFC 8259–compliant JSON output, enforced at the prompt.** Every
   structured (JSON-mode) call must instruct the model explicitly: no
   markdown fences, no commentary outside the object, escape newlines/
   quotes inside string values, straight quotes only, verdict field first
   (see `build_schema_instructions()` in `common/llm_client.py`). Don't
   relax this to "the parser will fix it" — the repair chain
   (`parse_json_response`) is a safety net for the models that ignore
   instructions, not a substitute for asking correctly.

2. **Never call the OpenAI SDK directly for a chat completion.** Always go
   through `create_chat_completion` (free-form output) or
   `create_structured_completion` (schema-validated JSON output) in
   `common/llm_client.py`. These are what implement the two-tier
   fallback, the timeout, and — for the structured variant — retry on a
   parse failure, not just an API-level one. A direct
   `client.chat.completions.create(...)` call bypasses all of that
   silently.

3. **Graceful degradation on 404 / rate limit / timeout / truncation —
   never let a single bad model crash the run.** `RETRYABLE_API_ERRORS`
   in `common/llm_client.py` defines exactly which exceptions trigger
   fallback (404, 429, timeout, connection error) and which don't
   (auth/bad-request errors surface immediately — no model swap fixes
   those). If you add a new call site, route it through the shared
   helpers above rather than writing new try/except handling; if you need
   to change retry behavior, change it once in `llm_client.py`.

4. **Preserve database state — `court_runs.db` is a durable record, not
   scratch state.**
   - Never write a migration that drops or recreates `trial_runs`; add
     columns via `ALTER TABLE` in `init_db()`'s migration step, following
     the existing pattern for the `model` column.
   - Never delete rows from `trial_runs` as a side effect of a code
     change. Clearing test data is a deliberate, explicit, user-requested
     action — back up the file first (a timestamped copy is enough) if
     you ever do it.
   - Schema changes must stay backward-compatible with existing rows
     (new columns get a `DEFAULT`), since `init_db()` runs against
     whatever `court_runs.db` a user already has on disk.

5. **Keep the two architectures comparable.** Both log to the identical
   schema and both respect the same token caps / prompt-formatting rules
   for equivalent roles (a judge's constraints are the same whether it's
   one of three independent calls or one field inside the monolithic
   call). If you change a judge's word limit, field order, or JSON
   formatting instructions in one architecture, mirror it in the other —
   see spec.md §2-3 for what "equivalent role" means across the two.

6. **No franchise-specific place names in user-facing or developer-facing
   text.** "Westeros" (or equivalents) must not reappear — see spec.md
   §6. Use "the Realm," "the tribunal," "the case."

7. **Random-mode engine selection stays hidden pre-run, revealed
   post-run.** Don't add a dropdown, list, or preview of `MODEL_POOL`
   contents anywhere in the Random-mode UI path — that defeats the point
   (spec.md §3.2). The resolved engine is reported only after a run
   completes, via `response.model` / `extract_usage`'s `executed_model`
   (spec.md §3.1), never guessed or displayed in advance.
