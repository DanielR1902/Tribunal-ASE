"""Single-Agent architecture for the Westeros Tribunal simulation.

A single Gemini call is asked to play every advocate and every judge at
once, and to return the entire result as one structured (Pydantic-validated)
JSON object. This is the "monolithic prompt" baseline that the multi-agent
architecture in ``project_multi_agent`` is compared against.

Run with:  python -m project_single_agent.main
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Make the sibling `common` package importable whether this file is run as
# `python -m project_single_agent.main` from the repo root, or directly as
# `python main.py` from inside project_single_agent/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import case_data  # noqa: E402
from common.cost_tracker import CostTracker, get_ils_exchange_rate  # noqa: E402
from common.database import TrialRunRecord, init_db, log_trial_run  # noqa: E402
from common.personas import ADVOCATE_PERSONAS, JUDGE_PERSONAS  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL = "gemini-3.5-flash"
ARCHITECTURE_MODE = "single_agent"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

Verdict = Literal["Justified", "Not Justified"]


class JudgeOpinion(BaseModel):
    reasoning: str
    verdict: Verdict


class TribunalVerdict(BaseModel):
    prosecution_summary: str
    defense_summary: str
    judge_barak: JudgeOpinion
    judge_elon: JudgeOpinion
    judge_shamgar: JudgeOpinion


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt() -> str:
    case_briefing = case_data.get_case_briefing()

    advocate_blocks = "\n\n".join(
        f"- {p['name']} ({p['side'].upper()}): {p['persona']}"
        for p in ADVOCATE_PERSONAS.values()
    )
    judge_blocks = "\n\n".join(
        f"- {j['name']} [{j['model_label']}]: {j['persona']}"
        for j in JUDGE_PERSONAS.values()
    )

    return f"""You are simulating a full tribunal proceeding in a single pass. You must
internally role-play FOUR advocates and THREE judges, then output only the
combined result of that internal deliberation.

{case_briefing}

ADVOCATES YOU MUST VOICE (two for the prosecution, two for the defense).
Each advocate argues strictly in character, using only the agreed facts
above and reasonable inferences from them — no facts may be invented that
contradict the agreed record:

{advocate_blocks}

JUDGES YOU MUST VOICE. Each judge is a fully independent decision-maker who
applies their own distinctive judicial philosophy to the SAME record and the
SAME tribunal issue. The judges do not confer and their verdicts must NOT be
merged, averaged, or forced into agreement — each one reaches their own
conclusion strictly through their own interpretive method, even if that
means the three verdicts disagree with each other:

{judge_blocks}

YOUR TASK, in this exact order:
1. Have Daenerys Targaryen and Grey Worm jointly make the strongest
   prosecution case against Jon Snow, then produce "prosecution_summary": a
   faithful prose summary (not a transcript) of their combined argument,
   written in third person, at least three sentences.
2. Have Jon Snow and Tyrion Lannister jointly make the strongest defense of
   Jon Snow's actions, then produce "defense_summary": a faithful prose
   summary of their combined argument, written in third person, at least
   three sentences.
3. Have Judge Barak apply his four-fold purposive/proportionality scrutiny
   to the tribunal issue and produce his own "reasoning" (step-by-step, in
   his own voice and method) and independent "verdict".
4. Have Judge Elon apply his tradition-grounded, modesty-constrained method
   to the SAME tribunal issue and produce his own "reasoning" and
   independent "verdict", reached entirely on his own terms.
5. Have Judge Shamgar apply his institutional-competence, rule-of-law method
   to the SAME tribunal issue and produce his own "reasoning" and
   independent "verdict", reached entirely on his own terms.

Return ONLY the structured result matching the required schema. Do not
merge the three judges' verdicts into a consensus; report each independently
even if they conflict.
"""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_result(verdict: TribunalVerdict) -> None:
    _print_header(f"{case_data.CASE_ID}  —  SINGLE-AGENT ARCHITECTURE")
    print(f"Tribunal issue: {case_data.TRIBUNAL_ISSUE}\n")

    _print_header("PROSECUTION ARGUMENTS (Daenerys Targaryen & Grey Worm)")
    print(verdict.prosecution_summary)

    _print_header("DEFENSE ARGUMENTS (Jon Snow & Tyrion Lannister)")
    print(verdict.defense_summary)

    judge_map = {
        "barak": verdict.judge_barak,
        "elon": verdict.judge_elon,
        "shamgar": verdict.judge_shamgar,
    }
    for key in ("barak", "elon", "shamgar"):
        persona = JUDGE_PERSONAS[key]
        opinion = judge_map[key]
        _print_header(f"{persona['name']}  [{persona['model_label']}]")
        print(opinion.reasoning)
        print(f"\n>>> VERDICT: {opinion.verdict}")


def print_budget_box(run_id: int, model: str, execution_time: float, tracker: CostTracker) -> None:
    rate = get_ils_exchange_rate()
    lines = [
        f"Architecture        : {ARCHITECTURE_MODE}",
        f"Model               : {model}",
        f"SQLite Run ID       : {run_id}  (court_runs.db, table trial_runs)",
        f"Execution time      : {execution_time:.2f} sec",
        f"Prompt tokens       : {tracker.prompt_tokens:,}",
        f"Completion tokens   : {tracker.completion_tokens:,}",
        f"Total tokens        : {tracker.total_tokens:,}",
        f"Estimated cost      : ${tracker.cost_usd:.6f} USD  (~₪{tracker.cost_ils:.6f} ILS @ {rate:.2f})",
    ]
    width = max(len(line) for line in lines) + 4
    print("\n┌" + "─" * width + "┐")
    print("│" + " TRIBUNAL RUN BUDGET SUMMARY ".center(width) + "│")
    print("├" + "─" * width + "┤")
    for line in lines:
        print("│ " + line.ljust(width - 1) + "│")
    print("└" + "─" * width + "┘")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env at the "
            "repository root and add your Gemini API key before running."
        )
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    client = genai.Client(api_key=api_key)
    db_path = init_db()
    tracker = CostTracker(model=model)

    print(f"Running {case_data.CASE_ID} — single-agent architecture (model: {model})...")

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TribunalVerdict,
            temperature=0.7,
        ),
    )
    execution_time = time.perf_counter() - start

    verdict: TribunalVerdict | None = response.parsed
    if verdict is None:
        # Fall back to manual validation if the SDK could not auto-parse
        # (e.g. an older SDK version) — response.text still holds the JSON.
        verdict = TribunalVerdict.model_validate_json(response.text)

    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage else 0
    completion_tokens = usage.candidates_token_count if usage else 0
    tracker.record("single_call", prompt_tokens, completion_tokens)

    print_result(verdict)

    record = TrialRunRecord(
        architecture_mode=ARCHITECTURE_MODE,
        prosecution_summary=verdict.prosecution_summary,
        defense_summary=verdict.defense_summary,
        judge_barak_reasoning=verdict.judge_barak.reasoning,
        judge_barak_verdict=verdict.judge_barak.verdict,
        judge_elon_reasoning=verdict.judge_elon.reasoning,
        judge_elon_verdict=verdict.judge_elon.verdict,
        judge_shamgar_reasoning=verdict.judge_shamgar.reasoning,
        judge_shamgar_verdict=verdict.judge_shamgar.verdict,
        prompt_tokens=tracker.prompt_tokens,
        completion_tokens=tracker.completion_tokens,
        total_tokens=tracker.total_tokens,
        cost_usd=tracker.cost_usd,
        cost_ils=tracker.cost_ils,
        execution_time_sec=execution_time,
    )
    run_id = log_trial_run(record, db_path=db_path)

    print_budget_box(run_id, model, execution_time, tracker)


if __name__ == "__main__":
    main()
