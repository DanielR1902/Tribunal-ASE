"""Canonical case file for Case T-001: The Realm v. Jon Snow.

This module is the single source of truth for the facts of the case. Both
the single-agent and multi-agent architectures import from here so that
every advocate and judge — regardless of which architecture is running —
reasons over exactly the same record.
"""

from __future__ import annotations

CASE_ID: str = "Case T-001: The Realm v. Jon Snow"
ACCUSED: str = "Jon Snow"
DECEASED: str = "Daenerys Targaryen"

ALLEGED_ACT: str = (
    "Jon Snow intentionally killed Daenerys Targaryen by stabbing her during "
    "a private meeting in the throne room, in the immediate aftermath of the "
    "fall of King's Landing."
)

# Agreed / stipulated facts. Both prosecution and defense accept these as the
# base premises of the case; the dispute is over their legal significance,
# not their occurrence.
AGREED_FACTS: list[str] = [
    (
        "King's Landing had surrendered: its bells rang and organized "
        "resistance had ceased. Daenerys then used Drogon against streets "
        "and civilians, causing destruction on a vast scale."
    ),
    (
        "After the victory, Daenerys told her assembled forces that the "
        "campaign of “liberation” would continue beyond King's "
        "Landing. Jon had seen the city and heard the speech."
    ),
    (
        "Tyrion Lannister renounced his office as Hand and was imprisoned. "
        "He warned Jon that Daenerys would treat Jon's sisters, and anyone "
        "else she regarded as an obstacle, as enemies."
    ),
    (
        "Jon asked Daenerys to forgive Tyrion and to show mercy. She "
        "refused to let others choose what was good and presented her own "
        "judgment as decisive."
    ),
    (
        "Daenerys was unarmed and was not attacking Jon when he killed her. "
        "Jon used their intimacy to get close enough to strike. He had not "
        "convened a council, attempted detention, or sought a public "
        "surrender of power."
    ),
]

TRIBUNAL_ISSUE: str = (
    "Was Jon Snow's intentional killing of Daenerys Targaryen justified as "
    "the necessary defense of others and of the realm, given what he knew, "
    "the scale of the threatened harm, the absence or presence of safer "
    "alternatives, and his lack of formal authority?"
)

TRIBUNAL_SCOPE: str = (
    "Each judge decides Justified or Not Justified with step-by-step "
    "reasoning. There is no sentencing phase. The three judicial opinions "
    "are NOT merged into a single collective compromise — each judge's "
    "reasoning and verdict must be reported independently, even where the "
    "judges disagree with one another."
)

# The standing case theory each side argues from, shown as a static outline
# in the "Canonical Case Facts" section regardless of whether any run has
# happened yet — distinct from the AI-generated prose summaries a specific
# run produces.
PROSECUTION_ARGUMENTS: list[str] = [
    "Unlawful assassination.",
    "Breach of allegiance.",
    "Lack of formal judicial authority.",
    "Absence of immediate lethal threat at the exact moment of the strike.",
    "Failure to attempt detention.",
]

DEFENSE_ARGUMENTS: list[str] = [
    "Defense of necessity and defense of others.",
    "Prevention of mass future atrocities following the burning of King's Landing.",
    "Absence of any feasible lawful alternative to halt an absolute monarch commanding a dragon.",
]


def get_case_briefing() -> str:
    """Render the canonical case file as a single briefing block of text.

    This is the shared block of context injected into every prompt — the
    monolithic single-agent prompt as well as every individual advocate and
    judge call in the multi-agent architecture — so that all reasoning is
    grounded in the same agreed record.
    """
    facts_block = "\n".join(f"  {i}. {fact}" for i, fact in enumerate(AGREED_FACTS, start=1))
    return (
        f"CASE: {CASE_ID}\n"
        f"ACCUSED: {ACCUSED}\n"
        f"DECEASED: {DECEASED}\n"
        f"ALLEGED ACT: {ALLEGED_ACT}\n\n"
        f"AGREED FACTS (stipulated by all parties; only their legal "
        f"significance is disputed):\n{facts_block}\n\n"
        f"TRIBUNAL ISSUE FOR JUDGMENT:\n  {TRIBUNAL_ISSUE}\n\n"
        f"SCOPE OF JUDGMENT:\n  {TRIBUNAL_SCOPE}"
    )
