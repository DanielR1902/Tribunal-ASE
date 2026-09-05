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

from common.llm_client import (
    ADVOCATE_MAX_TOKENS,
    JSON_OBJECT_RESPONSE_FORMAT,
    JUDGE_MAX_TOKENS,
    USAGE_ACCOUNTING_EXTRA_BODY,
    build_schema_instructions,
    create_chat_completion,
    extract_finish_reason,
    extract_usage,
    parse_json_response,
)
from common.personas import AdvocatePersona, JudgePersona

Verdict = Literal["Justified", "Not Justified"]


class JudgeOpinion(BaseModel):
    # "verdict" is declared FIRST (and build_schema_instructions() embeds
    # this same field order in the schema shown to the model) so that even
    # if a response gets cut off by the token limit partway through
    # "reasoning", the verdict — the one field that must never be lost — has
    # already been written.
    verdict: Verdict
    reasoning: str


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
        """Produce this advocate's argument to the tribunal as a short legal brief."""
        prompt = f"""{case_briefing}

You are appearing before the tribunal as one of two advocates for the
{self.side.upper()}. {stance_instruction}

Write a formal, concise legal brief in the first person, as yourself — NOT a
theatrical monologue or narrative roleplay. Ground every claim strictly in
the agreed facts above; do not invent facts that contradict them.

STRICT FORMAT — exactly these three labeled sections, in this order:
1. Core Legal Claim — one line naming the single controlling legal theory
   (e.g. "Absence of Imminent Peril" or "Extrajudicial Execution").
2. Supporting Agreed Facts — 3 to 4 bullet points, each citing one agreed
   fact and its legal significance.
3. Concluding Plea — one sentence stating what you ask the tribunal to find.

STRICT LENGTH LIMIT: 150-200 words total. Do not pad with rhetorical
flourish, repetition, or dramatic address to the court — every sentence
must carry legal weight.
"""
        response = create_chat_completion(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": self.persona["persona"]},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=ADVOCATE_MAX_TOKENS,
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

PROSECUTION'S BRIEF (Daenerys Targaryen & Grey Worm):
{prosecution_text}

DEFENSE'S BRIEF (Jon Snow & Tyrion Lannister):
{defense_text}

You are one of three independent judges on this tribunal. You do not confer
with the other two judges, you have not seen their reasoning, and your
verdict must never be merged or averaged with theirs. Apply your own
distinctive judicial method, exactly as described in your persona, to reach
your own conclusion on the tribunal issue above.

Write a concise, sharp, formal judicial opinion — NOT a lengthy narrative.
Decide your verdict FIRST, applying your distinctive judicial method, then
write "reasoning" to justify it. Your "reasoning" must be at most 120 words,
structured as exactly these three parts:
1. Legal Standard Applied — the controlling test under your judicial
   method, in one line.
2. Application to the Agreed Facts — 2 to 3 bulleted points applying that
   standard to the specific facts above.
3. Verdict Rationale — one concluding sentence stating why the standard is,
   or is not, met.

OUTPUT FIELD ORDER (critical): write "verdict" as the FIRST field in the
JSON object, followed by "reasoning" as the second field — this way your
verdict is already recorded even if the response gets cut off before
"reasoning" finishes.
""" + build_schema_instructions(JudgeOpinion)
        response = create_chat_completion(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": self.persona["persona"]},
                {"role": "user", "content": prompt},
            ],
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
            temperature=0.4,
            max_tokens=JUDGE_MAX_TOKENS,
            extra_body=USAGE_ACCOUNTING_EXTRA_BODY,
        )
        content = response.choices[0].message.content or ""
        opinion = parse_json_response(content, JudgeOpinion, finish_reason=extract_finish_reason(response))

        prompt_tokens, completion_tokens, actual_cost, executed_model = extract_usage(response)
        call_result = CallResult(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=actual_cost,
            executed_model=executed_model,
        )
        return opinion, call_result
