"""Multi-Agent architecture for the Tribunal simulation.

Four dedicated AdvocateAgent instances and three dedicated JudgeAgent
instances each make their own independent OpenRouter call, coordinated by
``TribunalOrchestrator``. This is compared against the single monolithic
call in ``project_single_agent``.

Run with:  python -m project_multi_agent.main
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Make the sibling `common` package (and this package's own absolute import
# path) importable whether this file is run as `python -m
# project_multi_agent.main` from the repo root, or directly as `python
# main.py` from inside project_multi_agent/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import case_data  # noqa: E402
from common.cost_tracker import CostTracker, get_ils_exchange_rate  # noqa: E402
from common.database import TrialRunRecord, init_db, log_trial_run  # noqa: E402
from common.llm_client import get_client, get_model  # noqa: E402
from common.personas import JUDGE_PERSONAS  # noqa: E402
from project_multi_agent.orchestrator import TribunalOrchestrator, TribunalResult  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

ARCHITECTURE_MODE = "multi_agent"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_result(result: TribunalResult) -> None:
    _print_header(f"{case_data.CASE_ID}  —  MULTI-AGENT ARCHITECTURE")
    print(f"Tribunal issue: {case_data.TRIBUNAL_ISSUE}\n")

    _print_header("PROSECUTION ARGUMENTS (Daenerys Targaryen & Grey Worm)")
    print(result.prosecution_summary)

    _print_header("DEFENSE ARGUMENTS (Jon Snow & Tyrion Lannister)")
    print(result.defense_summary)

    for key in ("barak", "elon", "shamgar"):
        persona = JUDGE_PERSONAS[key]
        opinion = result.judge_opinions[key]
        _print_header(f"{persona['name']}  [{persona['model_label']}]")
        print(opinion.reasoning)
        print(f"\n>>> VERDICT: {opinion.verdict}")


def print_budget_box(run_id: int, model: str, execution_time: float, tracker: CostTracker) -> None:
    rate = get_ils_exchange_rate()
    lines = [
        f"Architecture        : {ARCHITECTURE_MODE}",
        f"Requested model     : {model}",
    ]
    if tracker.executed_model_summary != model:
        lines.append(f"Executed model(s)   : {tracker.executed_model_summary}")
    lines += [
        f"Agent calls made    : {len(tracker.calls)} (4 advocates + 3 judges)",
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

    print("\nPer-call token breakdown:")
    for call in tracker.calls:
        print(
            f"  - {call.label:<18} model={call.executed_model:<32} "
            f"prompt={call.prompt_tokens:>6}  "
            f"completion={call.completion_tokens:>6}  "
            f"cost=${call.cost_usd:.6f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env at the "
            "repository root and add your OpenRouter API key before running."
        )
    model = get_model()

    client = get_client(api_key)
    db_path = init_db()
    tracker = CostTracker(model=model)

    print(f"Running {case_data.CASE_ID} — multi-agent architecture (model: {model})...")

    start = time.perf_counter()
    orchestrator = TribunalOrchestrator(client=client, model=model, tracker=tracker)
    result = orchestrator.run()
    execution_time = time.perf_counter() - start

    print_result(result)

    barak = result.judge_opinions["barak"]
    elon = result.judge_opinions["elon"]
    shamgar = result.judge_opinions["shamgar"]

    record = TrialRunRecord(
        architecture_mode=ARCHITECTURE_MODE,
        model=tracker.executed_model_summary,
        prosecution_summary=result.prosecution_summary,
        defense_summary=result.defense_summary,
        judge_barak_reasoning=barak.reasoning,
        judge_barak_verdict=barak.verdict,
        judge_elon_reasoning=elon.reasoning,
        judge_elon_verdict=elon.verdict,
        judge_shamgar_reasoning=shamgar.reasoning,
        judge_shamgar_verdict=shamgar.verdict,
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
