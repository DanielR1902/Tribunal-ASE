# Westeros Tribunal

A fictional-legal simulation used to compare two LLM agent architectures —
**single-agent** (one monolithic prompt) vs. **multi-agent** (dedicated
agents per role) — on the same task: running a tribunal case with
Google Gemini via the official `google-genai` SDK.

**Case T-001: The Realm v. Jon Snow.** Jon Snow stands accused of the
intentional killing of Daenerys Targaryen. Four advocates argue the case
(two prosecution, two defense) and three judges — each modeled on a
distinct real-world judicial philosophy — independently decide whether the
killing was **Justified** or **Not Justified** as the necessary defense of
others and of the realm. This is a creative/legal-reasoning exercise built
on the *Game of Thrones* fiction; it is not legal advice and the judicial
"models" are stylized, fictionalized approximations of jurisprudential
schools of thought, not statements about any real jurist's actual views.

## Project structure

```
westeros_tribunal/
├── .env.example
├── requirements.txt
├── README.md
├── app.py                   # Streamlit GUI — run/inspect both architectures from the browser
├── common/
│   ├── __init__.py
│   ├── case_data.py        # Canonical facts, charge sheet, tribunal issue
│   ├── personas.py         # Persona prompts for 4 advocates & 3 judges
│   ├── database.py         # SQLite schema init and execution logger
│   └── cost_tracker.py     # Token usage -> USD/ILS cost estimation
├── project_single_agent/
│   ├── __init__.py
│   └── main.py             # One monolithic Gemini call, structured output
└── project_multi_agent/
    ├── __init__.py
    ├── agents.py            # AdvocateAgent / JudgeAgent classes
    ├── orchestrator.py      # Coordinates the 7-call courtroom workflow
    └── main.py              # Entry point for the multi-agent run
```

## Setup

1. **Python 3.11+** is recommended (the code uses modern type-hint syntax
   such as `str | None` and `list[str]`).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
   and create your `.env` file:

   ```bash
   cp .env.example .env
   # then edit .env and set GEMINI_API_KEY=...
   ```

   Optional variables in `.env`:

   | Variable         | Default              | Purpose                                   |
   |------------------|-----------------------|--------------------------------------------|
   | `GEMINI_MODEL`   | `gemini-2.5-flash`   | Which Gemini model both architectures call |
   | `ILS_PER_USD`    | `3.65`                | USD → ILS exchange rate used for reporting |
   | `COURT_DB_PATH`  | `court_runs.db`       | SQLite file location (repo root by default)|

## Running the GUI (Streamlit)

The easiest way to use this project is the browser-based GUI:

```bash
streamlit run app.py
```

This opens a page with two buttons — **Run Single-Agent Simulation** and
**Run Multi-Agent Simulation** — a live view of the prosecution/defense
arguments, a card per judge with their reasoning and a color-coded
Justified/Not Justified badge, a majority-rule final outcome, a budget
summary (execution time, token breakdown, cost in USD/ILS), and a
**Historical Runs** tab that reads `court_runs.db` so you can browse and
re-inspect any past deliberation, from either architecture, without
re-running it.

## Running the simulations (command line)

From the repository root (`westeros_tribunal/`):

```bash
# Single-agent architecture: one monolithic structured-output call
python -m project_single_agent.main

# Multi-agent architecture: 4 advocate agents + 3 judge agents (7 calls)
python -m project_multi_agent.main
```

Each run prints:

1. The prosecution's combined arguments (Daenerys Targaryen & Grey Worm).
2. The defense's combined arguments (Jon Snow & Tyrion Lannister).
3. Each judge's step-by-step reasoning and independent verdict (Judge
   Barak, Judge Elon, Judge Shamgar) — the three verdicts are never merged
   into a single consensus, even when they disagree.
4. A budget summary box with execution time, token usage, estimated cost
   (USD and ILS), and the SQLite row ID the run was logged under.

Both architectures log their run to the same SQLite database
(`court_runs.db`, table `trial_runs`), tagged by `architecture_mode`
(`single_agent` or `multi_agent`), so you can compare them directly, e.g.:

```bash
sqlite3 court_runs.db "SELECT architecture_mode, total_tokens, cost_usd, execution_time_sec FROM trial_runs ORDER BY id;"
```

## Architecture notes

- **Single-agent** (`project_single_agent/main.py`): a single
  `client.models.generate_content(...)` call, with the full case file, all
  four advocate personas, and all three judge personas embedded in one
  prompt. Gemini's structured output (`response_schema` bound to a Pydantic
  model) forces the reply into `prosecution_summary`, `defense_summary`,
  and three independent `{reasoning, verdict}` judge objects.

- **Multi-agent** (`project_multi_agent/`): `agents.py` defines
  `AdvocateAgent` (free-form spoken argument) and `JudgeAgent` (structured
  `{reasoning, verdict}` output) as thin wrappers around one Gemini call
  each. `orchestrator.py`'s `TribunalOrchestrator` runs the two prosecution
  advocates concurrently, then the two defense advocates concurrently, then
  submits the combined arguments to all three judges concurrently and
  independently — 7 Gemini calls total, with token usage aggregated across
  all of them via `common/cost_tracker.py`.

## Cost estimation

`common/cost_tracker.py` estimates cost from Gemini's published
per-million-token pricing for `gemini-2.5-flash`, `gemini-2.5-flash-lite`,
and `gemini-2.5-pro` (with `gemini-2.5-pro`'s long-context pricing tier
applied above 200k prompt tokens). Prices are Google's list prices at the
time this project was written and can change — treat the reported cost as
an estimate, not an invoice. See
[ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
for current rates, and update `PRICING_USD_PER_MILLION_TOKENS` in
`common/cost_tracker.py` if they drift.
