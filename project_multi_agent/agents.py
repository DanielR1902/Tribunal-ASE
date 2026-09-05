"""Dedicated advocate and judge agent classes for the multi-agent architecture.

Each agent wraps exactly one OpenRouter chat-completion call and is
responsible for a single persona. ``AdvocateAgent`` produces free-form
spoken argument; ``JudgeAgent`` produces a structured (Pydantic-validated)
reasoning + verdict pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from common.llm_client import USAGE_ACCOUNTING_EXTRA_BODY, build_schema_instructions, extract_usage
from common.personas import AdvocatePersona, JudgePersona

Verdict = Literal["Justified", "Not Justified"]


class JudgeOpinion(BaseModel):
    reasoning: str
    verdict: Verdict


@dataclass
class CallResult:
    """Raw text plus token/cost accounting for a single OpenRouter call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None = None
    executed_model: str | None = None


class AdvocateAgent:
    """One advocate (prosecution or defense) speaking in their own voice."""

    def __init__(self, client, model: str, persona: AdvocatePersona) -> None:
        self.client = client
        self.model = model
        self.persona = persona

    @property
    def name(self) -> str:
        return self.persona["name"]

    @property
    def side(self) -> str:
        return self.persona["side"]

    def generate_argument(self, case_briefing: str, stance_instruction: str) -> CallResult:
        """Produce this advocate's spoken argument to the tribunal."""
        prompt = f"""{case_briefing}

You are appearing before the tribunal as one of two advocates for the
{self.side.upper()}. {stance_instruction}

Deliver your argument on the tribunal issue directly to the court, in your
own voice and character, in 3 to 5 paragraphs. Ground every claim in the
agreed facts above; do not invent facts that contradict them. Speak in the
first person, as yourself, addressed to the tribunal.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.persona["persona"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            extra_body=USAGE_ACCOUNTING_EXTRA_BODY,
        )
        text = (response.choices[0].message.content or "").strip()
        prompt_tokens, completion_tokens, actual_cost, executed_model = extract_usage(response)
        return CallResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=actual_cost,
            executed_model=executed_model,
        )


class JudgeAgent:
    """One independent judge, applying their own judicial method."""

    def __init__(self, client, model: str, persona: JudgePersona) -> None:
        self.client = client
        self.model = model
        self.persona = persona

    @property
    def name(self) -> str:
        return self.persona["name"]

    @property
    def model_label(self) -> str:
        return self.persona["model_label"]

    def deliberate(
        self, case_briefing: str, prosecution_text: str, defense_text: str
    ) -> tuple[JudgeOpinion, CallResult]:
        """Independently reason through the tribunal issue and reach a verdict.

        This judge never sees the other two judges' output — each JudgeAgent
        call is fully independent, so the three verdicts are never merged or
        averaged into a consensus.
        """
        prompt = f"""{case_briefing}

PROSECUTION'S COMBINED ARGUMENT (Daenerys Targaryen & Grey Worm):
{prosecution_text}

DEFENSE'S COMBINED ARGUMENT (Jon Snow & Tyrion Lannister):
{defense_text}

You are one of three independent judges on this tribunal. You do not confer
with the other two judges, you have not seen their reasoning, and your
verdict must never be merged or averaged with theirs. Apply your own
distinctive judicial method, exactly as described in your persona, to reach
your own conclusion on the tribunal issue above. Provide step-by-step
reasoning consistent with your method, then a single binary verdict.
""" + build_schema_instructions(JudgeOpinion)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.persona["persona"]},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            extra_body=USAGE_ACCOUNTING_EXTRA_BODY,
        )
        content = response.choices[0].message.content or ""
        opinion = JudgeOpinion.model_validate_json(content)

        prompt_tokens, completion_tokens, actual_cost, executed_model = extract_usage(response)
        call_result = CallResult(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=actual_cost,
            executed_model=executed_model,
        )
        return opinion, call_result
