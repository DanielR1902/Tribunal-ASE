"""Workflow orchestration for the multi-agent architecture.

Sequence:
    1. The two prosecution advocates (Daenerys Targaryen, Grey Worm) each
       generate their argument, concurrently.
    2. The two defense advocates (Jon Snow, Tyrion Lannister) each generate
       their argument, concurrently.
    3. The combined prosecution and defense arguments are submitted to all
       three judges, who deliberate independently and concurrently.

That is 4 advocate calls + 3 judge calls = 7 total Gemini calls per run,
whose token usage is aggregated into a single ``CostTracker``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from common import case_data
from common.cost_tracker import CostTracker
from common.personas import (
    ADVOCATE_PERSONAS,
    DEFENSE_KEYS,
    JUDGE_ORDER,
    JUDGE_PERSONAS,
    PROSECUTION_KEYS,
)

from project_multi_agent.agents import AdvocateAgent, JudgeAgent, JudgeOpinion

PROSECUTION_STANCE = (
    "Your goal is to persuade the tribunal that Jon Snow's killing of "
    "Daenerys Targaryen was NOT a justified defense of others or of the "
    "realm — that it was an unlawful, personally motivated act, carried "
    "out in secret against an unarmed woman, without any attempt at "
    "lawful process."
)

DEFENSE_STANCE = (
    "Your goal is to persuade the tribunal that Jon Snow's killing of "
    "Daenerys Targaryen WAS justified as the necessary defense of others "
    "and of the realm, given the scale of the threatened harm, what Jon "
    "knew at the time, and the absence of any safer alternative in the "
    "moment he acted."
)


@dataclass
class TribunalResult:
    prosecution_summary: str
    defense_summary: str
    prosecution_raw: dict[str, str] = field(default_factory=dict)
    defense_raw: dict[str, str] = field(default_factory=dict)
    judge_opinions: dict[str, JudgeOpinion] = field(default_factory=dict)


class TribunalOrchestrator:
    """Builds all 7 agents and drives the courtroom workflow."""

    def __init__(self, client, model: str, tracker: CostTracker) -> None:
        self.client = client
        self.model = model
        self.tracker = tracker

        self.advocates: dict[str, AdvocateAgent] = {
            key: AdvocateAgent(client, model, persona)
            for key, persona in ADVOCATE_PERSONAS.items()
        }
        self.judges: dict[str, JudgeAgent] = {
            key: JudgeAgent(client, model, persona)
            for key, persona in JUDGE_PERSONAS.items()
        }

    def _run_side(self, keys: list[str], stance_instruction: str) -> dict[str, str]:
        """Run every advocate on one side concurrently, recording usage."""
        case_briefing = case_data.get_case_briefing()
        results: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            futures = {
                key: pool.submit(
                    self.advocates[key].generate_argument, case_briefing, stance_instruction
                )
                for key in keys
            }
            for key, future in futures.items():
                call_result = future.result()
                self.tracker.record(
                    label=f"advocate:{key}",
                    prompt_tokens=call_result.prompt_tokens,
                    completion_tokens=call_result.completion_tokens,
                )
                results[key] = call_result.text

        return results

    def _run_judges(self, prosecution_text: str, defense_text: str) -> dict[str, JudgeOpinion]:
        """Run all three judges concurrently and independently."""
        case_briefing = case_data.get_case_briefing()
        opinions: dict[str, JudgeOpinion] = {}

        with ThreadPoolExecutor(max_workers=len(JUDGE_ORDER)) as pool:
            futures = {
                key: pool.submit(
                    self.judges[key].deliberate, case_briefing, prosecution_text, defense_text
                )
                for key in JUDGE_ORDER
            }
            for key, future in futures.items():
                opinion, call_result = future.result()
                self.tracker.record(
                    label=f"judge:{key}",
                    prompt_tokens=call_result.prompt_tokens,
                    completion_tokens=call_result.completion_tokens,
                )
                opinions[key] = opinion

        return opinions

    @staticmethod
    def _combine(raw: dict[str, str], persona_lookup: dict) -> str:
        parts = []
        for key, text in raw.items():
            name = persona_lookup[key]["name"]
            parts.append(f"[{name}]\n{text}")
        return "\n\n".join(parts)

    def run(self) -> TribunalResult:
        # Step 1: prosecution advocates (concurrent).
        prosecution_raw = self._run_side(PROSECUTION_KEYS, PROSECUTION_STANCE)
        prosecution_summary = self._combine(prosecution_raw, ADVOCATE_PERSONAS)

        # Step 2: defense advocates (concurrent).
        defense_raw = self._run_side(DEFENSE_KEYS, DEFENSE_STANCE)
        defense_summary = self._combine(defense_raw, ADVOCATE_PERSONAS)

        # Step 3: combined payload submitted to all 3 judges (concurrent,
        # each judge fully independent of the other two).
        judge_opinions = self._run_judges(prosecution_summary, defense_summary)

        return TribunalResult(
            prosecution_summary=prosecution_summary,
            defense_summary=defense_summary,
            prosecution_raw=prosecution_raw,
            defense_raw=defense_raw,
            judge_opinions=judge_opinions,
        )
