"""Dedicated advocate and judge agent classes for the multi-agent architecture.

Each agent wraps exactly one Gemini call and is responsible for a single
persona. ``AdvocateAgent`` produces free-form spoken argument; ``JudgeAgent``
produces a structured (Pydantic-validated) reasoning + verdict pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from common.personas import AdvocatePersona, JudgePersona

Verdict = Literal["Justified", "Not Justified"]


class JudgeOpinion(BaseModel):
    reasoning: str
    verdict: Verdict


@dataclass
class CallResult:
    """Raw text plus token accounting for a single Gemini API call."""

    text: str
    prompt_tokens: int
    completion_tokens: int


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
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.persona["persona"],
                temperature=0.8,
            ),
        )
        usage = response.usage_metadata
        return CallResult(
            text=(response.text or "").strip(),
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
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
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.persona["persona"],
                response_mime_type="application/json",
                response_schema=JudgeOpinion,
                temperature=0.4,
            ),
        )
        opinion: JudgeOpinion | None = response.parsed
        if opinion is None:
            opinion = JudgeOpinion.model_validate_json(response.text)

        usage = response.usage_metadata
        call_result = CallResult(
            text=response.text or "",
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
        )
        return opinion, call_result
