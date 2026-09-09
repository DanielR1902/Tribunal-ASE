"""Streamlit GUI for the Tribunal simulation.

Lets a user trigger a Single-Agent or Multi-Agent run from the browser,
watch the deliberation render live, and browse past runs logged in
court_runs.db — instead of driving everything from the command line.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from common import case_data
from common.cost_tracker import CostTracker, get_ils_exchange_rate
from common.database import TrialRunRecord, get_db_path, init_db, log_trial_run
from common.llm_client import (
    JSON_OBJECT_RESPONSE_FORMAT,
    SINGLE_AGENT_MAX_TOKENS,
    USAGE_ACCOUNTING_EXTRA_BODY,
    VERIFIED_STABLE_POOL,
    create_structured_completion,
    extract_usage,
    get_client,
    get_model,
    pick_random_model,
)
from common.personas import (
    ADVOCATE_PERSONAS,
    DEFENSE_KEYS,
    JUDGE_ORDER,
    JUDGE_PERSONAS,
    PROSECUTION_KEYS,
)
from project_multi_agent.orchestrator import TribunalOrchestrator
from project_single_agent.main import TribunalVerdict, build_prompt as build_single_prompt

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL = get_model()
CUSTOM_AGENT_DEFAULT_MODEL = "google/gemini-2.5-flash"

# Display label for each architecture_mode value stored in court_runs.db —
# shared by render_result, render_history_panel, and
# render_historical_deliberation so the three stay in sync by construction.
ARCHITECTURE_LABELS: dict[str, str] = {
    "single_agent": "Single-Agent",
    "multi_agent": "Multi-Agent",
    "custom_agent": "Custom-Agent",
}

# (persona key, full role label) — one shared source of truth for the
# Model Assignment table and the historical-feed engine list, in the
# fixed Prosecution 1/2, Defense 1/2, Judge 1/2/3 order.
ROLE_ASSIGNMENT_ORDER: list[tuple[str, str]] = [
    (PROSECUTION_KEYS[0], "Prosecution Advocate 1"),
    (PROSECUTION_KEYS[1], "Prosecution Advocate 2"),
    (DEFENSE_KEYS[0], "Defense Advocate 1"),
    (DEFENSE_KEYS[1], "Defense Advocate 2"),
    (JUDGE_ORDER[0], "Judge 1"),
    (JUDGE_ORDER[1], "Judge 2"),
    (JUDGE_ORDER[2], "Judge 3"),
]


def _persona_display_name(key: str) -> str:
    """The character/persona name for an advocate or judge persona key —
    looked up from whichever of the two persona dicts actually defines it,
    since ROLE_ASSIGNMENT_ORDER spans both.
    """
    if key in ADVOCATE_PERSONAS:
        return ADVOCATE_PERSONAS[key]["name"]
    return JUDGE_PERSONAS[key]["name"]


def _parse_role_models(role_models_json: str | None) -> dict[str, str]:
    """Deserialize a ``role_models_json`` DB value back into the
    persona-key -> model dict it was serialized from. Empty/missing/
    malformed input (single- and multi-agent rows always store '') returns
    ``{}`` rather than raising, since a Custom-Agent check always guards
    the caller anyway.
    """
    if not role_models_json:
        return {}
    try:
        data = json.loads(role_models_json)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}

st.set_page_config(
    page_title="Tribunal",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme — "LegalTech Command Center": a dark Obsidian canvas with layered
# slate card surfaces, an Electric Indigo / Vivid Cyan digital accent
# system, Inter for every text role, and JetBrains Mono reserved for
# model tags, case codes, and telemetry metrics. Presentation-only: purely
# CSS plus, further below, small pure-string HTML builders for the judge/
# advocate/history/case-overview/outcome cards. Nothing here touches request
# routing, model selection, parsing, database logic, or session state.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* Canvas + surface — deep midnight obsidian with two elevated slate tiers */
    --trib-bg: #090D16;
    --trib-panel: #111827;
    --trib-panel-2: #151E2E;
    --trib-border: #1E293B;
    --trib-border-strong: #2A3B52;
    --trib-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    --trib-shadow-lift: 0 10px 30px -4px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(99, 102, 241, 0.18);

    /* Typography */
    --trib-text-heading: #F8FAFC;
    --trib-text-body: #94A3B8;
    --trib-text-dim: #64748B;
    --trib-chip-bg: #151E2E;

    /* Digital brand accents */
    --trib-indigo: #6366F1;
    --trib-indigo-deep: #4F46E5;
    --trib-indigo-bg: rgba(99, 102, 241, 0.12);
    --trib-indigo-border: rgba(99, 102, 241, 0.35);
    --trib-cyan: #06B6D4;
    --trib-cyan-bg: rgba(6, 182, 212, 0.12);
    --trib-cyan-border: rgba(6, 182, 212, 0.35);

    /* Status */
    --trib-emerald: #34D399;
    --trib-emerald-bg: rgba(16, 185, 129, 0.12);
    --trib-emerald-border: rgba(16, 185, 129, 0.3);
    --trib-rose: #FB7185;
    --trib-rose-bg: rgba(244, 63, 94, 0.12);
    --trib-rose-border: rgba(244, 63, 94, 0.3);
}

/* Base typography scale: 16px body text, headings sized 24-32px so the
   hierarchy stays legible at the larger base. */
html { font-size: 16px; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stHeader"] {
    background: var(--trib-bg) !important;
}
.stApp, .stApp p, .stApp li, .stApp label,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    color: var(--trib-text-body) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px !important;
}

h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    color: var(--trib-text-heading) !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 32px !important; }
h2 { font-size: 28px !important; }
h3 { font-size: 24px !important; }
h4 { font-size: 20px !important; }

.stApp small, [data-testid="stMarkdownContainer"] small {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.03em;
    color: var(--trib-text-dim) !important;
    font-size: 0.85rem !important;
}

[data-testid="stCaptionContainer"], .stCaption { color: var(--trib-text-dim) !important; font-size: 14px !important; }

/* Inline code / model pills (e.g. `google/gemini-2.5-flash` in "**Engine:**
   `...`" markdown, and the `Database: `path`` caption): Streamlit's
   default inline-code style assumes a light theme and renders a stark
   white pill, which clashes hard against the obsidian canvas. Recolored
   to a dark translucent slate tint with a luminous cyan value and a
   faint indigo-glow border so every engine/model tag in the app —
   whether plain markdown code or a purpose-built badge — reads as one
   consistent "digital" pill. */
code, [data-testid="stMarkdownContainer"] code, .stCode, .engine-tag {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: #38BDF8 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 0.85rem !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25) !important;
}

/* Alerts (st.info/warning/error/success): a uniform dark slate treatment
   instead of Streamlit's default light pastel fills, which would clash
   hard against the obsidian canvas. */
[data-testid="stAlert"] {
    background: var(--trib-panel-2) !important;
    border: 1px solid var(--trib-border) !important;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAlert"] p { color: var(--trib-text-body) !important; }

/* Primary execution buttons: a glowing electric-indigo gradient, sharp
   white bold text, subtle lift on hover — no clumsy borders. Full width
   of whatever column/container holds them (paired with width="stretch"
   from Python). */
.stButton > button {
    background: linear-gradient(135deg, var(--trib-indigo-deep), var(--trib-indigo)) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 15.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em;
    box-shadow: 0 2px 12px -2px rgba(99, 102, 241, 0.5) !important;
    transition: background 0.15s ease, transform 0.12s ease, box-shadow 0.15s ease;
}
.stButton > button:hover:not(:disabled) {
    background: linear-gradient(135deg, var(--trib-indigo), #818CF8) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 22px -2px rgba(99, 102, 241, 0.65) !important;
    color: #FFFFFF !important;
}
.stButton > button:active:not(:disabled) { transform: translateY(0); }
.stButton > button:disabled { background: #1E293B !important; border-color: #1E293B !important; color: var(--trib-text-dim) !important; box-shadow: none !important; }

/* "View" affordance inside the history feed: a sleek ghost button that
   lights up with a cyan border/glow on hover — deliberately quieter than
   the primary run buttons, and crisp rather than lifted. */
.trib-feed-row-actions .stButton > button {
    background: transparent !important;
    color: var(--trib-text-body) !important;
    border: 1px solid var(--trib-border-strong) !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    padding: 0.42rem 0.7rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.trib-feed-row-actions .stButton > button:hover:not(:disabled) {
    background: var(--trib-cyan-bg) !important;
    color: var(--trib-cyan) !important;
    border-color: var(--trib-cyan) !important;
    transform: none !important;
    box-shadow: 0 0 0 1px rgba(6, 182, 212, 0.25) !important;
}

[data-testid="stExpander"] {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border) !important;
    border-radius: 10px;
    box-shadow: var(--trib-shadow);
}
[data-testid="stExpander"] summary,
[data-testid="stExpanderHeader"] {
    background: #111827 !important;
    border-color: #1E293B !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}
[data-testid="stExpander"] details > div,
[data-testid="stExpanderDetails"] {
    background: #0D131F !important;
    border-top: 1px solid #1E293B !important;
    color: #E2E8F0 !important;
}
[data-testid="stExpander"] summary svg,
[data-testid="stExpanderHeader"] svg,
[data-testid="stExpanderToggleIcon"] svg {
    color: #94A3B8 !important;
    fill: #94A3B8 !important;
}

[data-testid="stMetric"] {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border);
    border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10.5px !important;
    letter-spacing: 0.08em;
    color: var(--trib-text-dim) !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--trib-text-heading) !important;
}

hr { border-color: var(--trib-border) !important; }

/* Dropdowns: dark inputs, a crisp 1px hairline border, an indigo focus
   glow on interaction. BaseWeb renders the open menu in a portal appended
   outside the normal widget DOM, so every portal container is targeted
   explicitly (and forcefully, with both `background` and
   `background-color`) rather than relying on inheritance. */
[data-baseweb="select"] > div {
    background: #0D131F !important;
    border: 1px solid var(--trib-border) !important;
    border-radius: 8px !important;
    color: var(--trib-text-heading) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-baseweb="select"]:focus-within > div {
    border-color: var(--trib-indigo) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
}

/* Force dark background and styling on dropdown popovers and menus */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
div[data-baseweb="select"] ul,
[role="listbox"] {
    background-color: #0F172A !important;
    background: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.8) !important;
    color: #E2E8F0 !important;
}

/* Force dark styling on every option/item */
li[data-baseweb="menu-item"],
[role="option"] {
    background-color: transparent !important;
    color: #CBD5E1 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

/* Hover and active states */
li[data-baseweb="menu-item"]:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"],
li[data-baseweb="menu-item"][aria-selected="true"] {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
}

/* Selectbox input wrapper when closed / idle */
div[data-baseweb="select"] > div:first-child,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stSelectbox"] > div > div {
    background-color: #111827 !important;
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
}

/* Hover and focus states on the select input */
div[data-baseweb="select"] > div:first-child:hover {
    border-color: #38BDF8 !important;
}

div[data-baseweb="select"] > div:first-child:focus-within {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
}

/* Arrow / chevron icon only — background reset to transparent so it can
   never render as a filled square, fill/stroke kept to a neutral slate
   (independent of the selected-text color). */
div[data-testid="stSelectbox"] svg {
    background: transparent !important;
    fill: #94A3B8 !important;
    stroke: #94A3B8 !important;
}

/* Target strictly the text node of the selected item — never the icon
   or its wrapper, so the chevron is unaffected by this color. */
div[data-testid="stSelectbox"] [data-baseweb="select"] div[class*="singleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] div[class*="ValueContainer"] > div:not([data-baseweb="icon"]) {
    color: #60A5FA !important;
    font-weight: 500 !important;
}

/* Radio buttons: comfortable size, electric indigo on hover/selection. */
[data-testid="stRadio"] label { font-size: 15px !important; }
[data-testid="stRadio"] [role="radiogroup"] label:hover p { color: var(--trib-indigo) !important; }

/* Native bordered container (st.container(border=True)) — used only to
   wrap the Tribunal Controls form into one elevated dark card; every
   other container in the app explicitly passes border=False, so this
   selector only ever touches that one spot. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--trib-shadow);
    padding: 4px 6px;
}

/* --- Top global header: a unified dark command bar --- */
.trib-topbar {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    background: var(--trib-panel); border: 1px solid var(--trib-border);
    border-radius: 12px; padding: 14px 22px; margin-bottom: 18px;
    box-shadow: var(--trib-shadow);
}
.trib-topbar-left { display: flex; align-items: center; gap: 12px; }
.trib-topbar-icon { font-size: 26px; filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.65)); }
.trib-topbar-title {
    font-family: 'Inter', sans-serif; font-size: 21px; font-weight: 800;
    letter-spacing: 0.09em; text-transform: uppercase; color: var(--trib-text-heading);
}
.trib-topbar-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--trib-cyan);
    background: var(--trib-cyan-bg); border: 1px solid var(--trib-cyan-border);
    border-radius: 999px; padding: 5px 12px; white-space: nowrap;
}

/* --- Panel headers (left/right column titles) --- */
.trib-panel-header { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.trib-panel-title { font-size: 19px; font-weight: 700; color: var(--trib-text-heading); }
.trib-count-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 600;
    color: var(--trib-cyan);
    background: var(--trib-chip-bg);
    border: 1px solid var(--trib-border);
    border-radius: 999px;
    padding: 1px 9px;
}

/* --- "Active Chamber & Bench" toggle: quiet dark pill by default, lights
   up indigo whenever a historical run is what's actually showing on the
   right, doubling as a status indicator. --- */
.trib-return-live .stButton > button {
    background: var(--trib-panel-2) !important;
    color: var(--trib-text-body) !important;
    border: 1px solid var(--trib-border-strong) !important;
    border-radius: 999px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.9rem !important;
    box-shadow: none !important;
}
.trib-return-live--active .stButton > button {
    background: var(--trib-indigo-bg) !important;
    color: var(--trib-indigo) !important;
    border-color: var(--trib-indigo-border) !important;
}

/* --- Historical Runs feed items: high-density cards, dark hairline
   borders, monospace timestamps, a clean cyan highlight on hover. --- */
.trib-feed-item {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border);
    border-radius: 8px;
    padding: 10px 12px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.trib-feed-item:hover {
    border-color: var(--trib-cyan-border);
    box-shadow: 0 2px 10px -2px rgba(6, 182, 212, 0.18);
}
.trib-feed-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.trib-feed-row + .trib-feed-row { margin-top: 6px; }
.trib-feed-time {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600;
    color: var(--trib-text-heading);
}
.trib-feed-engine {
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    color: var(--trib-text-dim); white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; max-width: 60%;
}

/* --- Status / mode pills --- */
.trib-pill {
    display: inline-flex; align-items: center;
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.03em;
    padding: 2px 9px; border-radius: 999px; text-transform: uppercase;
    white-space: nowrap;
}
.trib-pill--mode { background: var(--trib-chip-bg); color: var(--trib-text-body); border: 1px solid var(--trib-border); }
.trib-pill--justified { background: var(--trib-emerald-bg); color: var(--trib-emerald); border: 1px solid var(--trib-emerald-border); }
.trib-pill--not-justified { background: var(--trib-rose-bg); color: var(--trib-rose); border: 1px solid var(--trib-rose-border); }

/* --- Tribunal card system (judge / advocate cards, built as raw HTML below) --- */
.trib-card {
    border: 1px solid var(--trib-border);
    background: var(--trib-panel);
    border-radius: 10px;
    box-shadow: var(--trib-shadow);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 14px;
    height: 100%;
    box-sizing: border-box;
}
.trib-card--prosecution { border-top: 3px solid var(--trib-rose); }
.trib-card--defense { border-top: 3px solid var(--trib-cyan); }
.trib-card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.trib-card-name { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 17px; color: var(--trib-text-heading); }
.trib-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    align-self: flex-start;
    max-width: 100%;
    box-sizing: border-box;
}
.trib-badge--prosecution { color: var(--trib-rose); background: var(--trib-rose-bg); border: 1px solid var(--trib-rose-border); white-space: nowrap; }
.trib-badge--defense { color: var(--trib-cyan); background: var(--trib-cyan-bg); border: 1px solid var(--trib-cyan-border); white-space: nowrap; }

/* Judge identity block: name + legal-doctrine subtitle rendered as a
   small accent tag, stacked to the left of the verdict pill. */
.trib-judge-identity { display: flex; flex-direction: column; gap: 6px; }
.trib-doctrine-label {
    display: inline-flex; align-items: center; align-self: flex-start;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase; color: var(--trib-cyan);
    background: var(--trib-cyan-bg); border: 1px solid var(--trib-cyan-border);
    border-radius: 6px; padding: 3px 8px; line-height: 1.4;
}

.trib-card-body { font-size: 14.5px; line-height: 1.65; color: var(--trib-text-body); margin: 0; text-align: left; }
.trib-reasoning-p { margin: 0 0 10px 0; }
.trib-reasoning-list { margin: 0 0 10px 0; padding-left: 20px; }
.trib-reasoning-list li { margin-bottom: 5px; }
.trib-reasoning-p:last-child, .trib-reasoning-list:last-child { margin-bottom: 0; }

/* Prominent verdict pill — top-right of a judge card's header. */
.trib-verdict-pill {
    display: inline-flex; align-items: center; flex: none;
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
    padding: 6px 12px; border-radius: 999px; white-space: nowrap;
}
.trib-verdict-pill--justified { background: var(--trib-emerald-bg); color: var(--trib-emerald); border: 1px solid var(--trib-emerald-border); }
.trib-verdict-pill--not-justified { background: var(--trib-rose-bg); color: var(--trib-rose); border: 1px solid var(--trib-rose-border); }

/* --- Explicitly-bordered boxes (Budget Summary): a real HTML wrapper we
   control end to end, matching the card system's hairline border + soft
   shadow rather than Streamlit's own container styling. --- */
.trib-bordered-box {
    border: 1px solid var(--trib-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--trib-shadow);
    padding: 1rem !important;
    background: var(--trib-panel);
    margin-bottom: 14px;
}

/* --- Budget metrics grid + the highlighted "active run" tile --- */
.trib-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
}
.trib-metric {
    display: flex; flex-direction: column; gap: 4px;
    padding: 8px 10px; border-radius: 8px;
    background: var(--trib-panel-2); border: 1px solid var(--trib-border);
}
.trib-metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--trib-text-dim);
}
.trib-metric-value {
    font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 600;
    color: var(--trib-text-heading);
}
.trib-metric--run {
    background: linear-gradient(135deg, var(--trib-indigo-deep), var(--trib-indigo)) !important;
    border-color: var(--trib-indigo);
}
.trib-metric--run .trib-metric-label, .trib-metric--run .trib-metric-value {
    color: #FFFFFF !important;
}

/* --- Case overview card: metadata badges as paired tech-stat chips, a
   terminal-style Tribunal Issue inset with an indigo accent border, and
   instructional text below. --- */
.trib-overview-card {
    border: 1px solid var(--trib-border);
    background: var(--trib-panel);
    border-radius: 12px;
    box-shadow: var(--trib-shadow);
    padding: 18px 20px;
    margin-bottom: 14px;
}
.trib-tag-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
.trib-tag {
    display: flex; flex-direction: column; gap: 3px;
    padding: 8px 12px; border-radius: 8px;
    background: var(--trib-chip-bg); border: 1px solid var(--trib-border);
    min-width: 128px;
}
.trib-tag-label {
    font-family: 'Inter', sans-serif; font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--trib-cyan); font-weight: 600;
}
.trib-tag-value { font-size: 13px; color: var(--trib-text-heading); font-weight: 600; }
.trib-issue-box {
    background: #0A0F1A; border: 1px solid var(--trib-indigo-border);
    border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;
    box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.06);
}
.trib-issue-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--trib-indigo); font-weight: 700;
    display: block; margin-bottom: 6px;
}
.trib-issue-value { font-size: 14.5px; line-height: 1.6; color: var(--trib-text-heading); }
.trib-overview-scope {
    font-size: 14.5px; line-height: 1.6; color: var(--trib-text-body);
    margin: 0; padding-top: 12px; border-top: 1px solid var(--trib-border);
}

/* --- Custom-Agent model assignment table (live + historical views): a
   futuristic dark data grid — deep header row, alternating subtle row
   tints, and the engine column formatted as a small monospace chip. --- */
.trib-assign-table-wrap {
    border: 1px solid var(--trib-border); border-radius: 10px; overflow: hidden;
}
.trib-assign-table { width: 100%; border-collapse: collapse; }
.trib-assign-table thead tr { background: linear-gradient(135deg, #151E2E, #1E293B); }
.trib-assign-table th {
    text-align: left; padding: 10px 14px; border: none;
    font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--trib-text-heading);
}
.trib-assign-table td {
    text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--trib-border);
    font-size: 14px;
}
.trib-assign-table tbody tr:nth-child(even) { background: rgba(255, 255, 255, 0.02); }
.trib-assign-table tr:last-child td { border-bottom: none; }
.trib-assign-table td:first-child { color: var(--trib-text-heading); font-weight: 600; }
.trib-assign-table td:nth-child(2) { color: var(--trib-text-body); }
/* Same digital-pill treatment as the global inline-code rule above —
   dark translucent slate, a faint indigo-glow border, luminous cyan text —
   so the table's engine chip reads as the same "model tag" species as
   every other engine value in the app. */
.trib-engine-chip {
    display: inline-flex; align-items: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
    background: rgba(30, 41, 59, 0.7); color: #38BDF8;
    border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 5px; padding: 2px 7px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
}

/* --- Per-role model lines inside a Custom-Agent history feed card's
   "Engines Used" expander: minimalist, indented persona name, with the
   engine itself rendered as the same digital pill. --- */
.trib-feed-models { display: flex; flex-direction: column; gap: 6px; }
.trib-feed-model-line {
    padding-left: 4px; font-size: 12.5px; line-height: 1.6;
    white-space: normal; overflow-wrap: break-word;
}
.trib-feed-model-name { font-family: 'Inter', sans-serif; color: var(--trib-text-body); font-weight: 500; }
.trib-feed-model-engine {
    display: inline-flex; align-items: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
    background: rgba(30, 41, 59, 0.7); color: #38BDF8;
    border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 5px; padding: 2px 7px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
}

/* --- Final Tribunal Outcome: a bold terminal-style summary banner —
   large verdict label plus a compact per-judge ratio bar. --- */
.trib-outcome-card {
    border: 1px solid; border-radius: 14px;
    padding: 22px 26px; box-shadow: var(--trib-shadow); margin-bottom: 8px;
}
.trib-outcome-top { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.trib-outcome-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.trib-outcome-verdict-large { font-size: 30px; font-weight: 800; letter-spacing: -0.01em; }
.trib-outcome-ratio { display: flex; align-items: center; gap: 12px; }
.trib-ratio-bar { display: flex; gap: 4px; }
.trib-ratio-seg { width: 30px; height: 8px; border-radius: 4px; display: inline-block; }
.trib-outcome-count { font-family: 'Inter', sans-serif; font-size: 13.5px; font-weight: 700; color: var(--trib-text-body); }

/* --- Hide Streamlit's default chrome: main menu, Deploy button, header/
   decoration bar, and the "Made with Streamlit" footer. --- */
#MainMenu { visibility: hidden !important; display: none !important; }
.stDeployButton { visibility: hidden !important; display: none !important; }
header, [data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
header button { visibility: hidden !important; display: none !important; }
footer { visibility: hidden !important; display: none !important; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


def _format_reasoning_html(text: str) -> str:
    """Render LLM-authored reasoning/argument text as clean HTML: runs of
    consecutive lines starting with '-', '*', or '•' become a real
    ``<ul><li>`` list (proper bullet margin/spacing per the design spec)
    instead of one flat ``<br>``-joined blob; every other line is grouped
    into ordinary paragraphs. Every line is HTML-escaped here, so callers
    must NOT re-escape the result — this is pure presentation over
    already-fetched text, nothing to do with how it was generated.
    """
    lines = [ln.strip() for ln in text.strip().split("\n")]
    parts: list[str] = []
    buffer: list[str] = []
    in_list = False

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        if in_list:
            items = "".join(f"<li>{html.escape(item)}</li>" for item in buffer)
            parts.append(f'<ul class="trib-reasoning-list">{items}</ul>')
        else:
            joined = "<br>".join(html.escape(ln) for ln in buffer)
            parts.append(f'<p class="trib-reasoning-p">{joined}</p>')
        buffer = []

    for line in lines:
        if not line:
            continue
        is_bullet = line.startswith(("- ", "* ", "• "))
        content = line[2:].strip() if is_bullet else line
        if is_bullet != in_list:
            flush()
            in_list = is_bullet
        buffer.append(content)
    flush()
    return "".join(parts)


def _judge_card_html(persona: dict, opinion: dict) -> str:
    """Render one judge as a card: identity block (name + uppercase
    doctrine subtitle) with a prominent verdict pill top-right, and the
    full reasoning below as clean paragraphs/bullets. All interpolated
    text is HTML-escaped (by ``_format_reasoning_html`` for the reasoning,
    inline here for everything else) since it can originate from LLM
    output, not just our own literal strings.
    """
    is_justified = opinion["verdict"] == "Justified"
    verdict_class = "trib-verdict-pill--justified" if is_justified else "trib-verdict-pill--not-justified"
    verdict_label = html.escape(opinion["verdict"]).upper()
    name = html.escape(persona["name"])
    doctrine = html.escape(persona["model_label"])
    reasoning_html = _format_reasoning_html(opinion["reasoning"])
    return f"""<div class="trib-card">
  <div class="trib-card-header">
    <div class="trib-judge-identity">
      <div class="trib-card-name">{name}</div>
      <span class="trib-doctrine-label">{doctrine}</span>
    </div>
    <span class="trib-verdict-pill {verdict_class}">{verdict_label}</span>
  </div>
  <div class="trib-card-body">{reasoning_html}</div>
</div>"""


def _advocate_card_html(title: str, side: str, text: str) -> str:
    """Render a prosecution/defense argument as a card: a colored top
    border and badge per side (rose/PROSECUTION, cyan/DEFENSE), name
    header, body text in the card. ``text`` is HTML-escaped (by
    ``_format_reasoning_html``) since it's LLM-generated.
    """
    is_prosecution = side == "prosecution"
    card_class = "trib-card--prosecution" if is_prosecution else "trib-card--defense"
    badge_class = "trib-badge--prosecution" if is_prosecution else "trib-badge--defense"
    badge_label = "PROSECUTION" if is_prosecution else "DEFENSE"
    body_html = _format_reasoning_html(text)
    return f"""<div class="trib-card {card_class}">
  <div class="trib-card-header">
    <div class="trib-card-name">{html.escape(title)}</div>
    <span class="trib-badge {badge_class}">{badge_label}</span>
  </div>
  <div class="trib-card-body">{body_html}</div>
</div>"""


def _budget_metric_html(label: str, value: str, *, is_run_badge: bool = False) -> str:
    """One metric tile inside the Budget Summary bordered box. The active
    Run Number tile (``is_run_badge=True``) gets a solid dark background,
    distinct from the rest — this is what makes it the "Active Run
    Indicator" rather than just another generic metric.
    """
    css_class = "trib-metric trib-metric--run" if is_run_badge else "trib-metric"
    return (
        f'<div class="{css_class}">'
        f'<span class="trib-metric-label">{html.escape(label)}</span>'
        f'<span class="trib-metric-value">{html.escape(value)}</span>'
        f"</div>"
    )


def _budget_box_html(metrics: list[tuple[str, str]], run_id: int) -> str:
    """The Budget Summary metrics as one explicitly-bordered card
    (``.trib-bordered-box``) containing a grid of metric tiles, with the
    Run ID tile highlighted Glacier as the active-run indicator. The
    section title itself is a real st.subheader() call at the caller (see
    render_budget_box / render_history_tab) — not built into this HTML —
    so it renders as a standard prominent header matching "Final Tribunal
    Outcome" and the rest of the app, not a small all-caps eyebrow badge.
    """
    tiles = "".join(_budget_metric_html(label, value) for label, value in metrics)
    tiles += _budget_metric_html("Run #", f"#{run_id}", is_run_badge=True)
    return f"""<div class="trib-bordered-box">
  <div class="trib-metric-grid">{tiles}</div>
</div>"""


def _case_overview_card_html() -> str:
    """The core case-facts fields (Case ID / Accused / Deceased / Alleged
    act) as crisp metadata badges side by side, a full-width Tribunal
    Issue box, and the tribunal-scope guidance as instructional text
    underneath — all inside one sleek bordered card. The three sub-topics
    (Agreed Facts / Prosecution / Defense Arguments) stay in a separate
    real st.expander below — they can't be embedded inside this same raw
    HTML block since they're interactive Streamlit components, not static
    text.
    """
    tags = [
        ("Case ID", case_data.CASE_ID),
        ("Accused", case_data.ACCUSED),
        ("Deceased", case_data.DECEASED),
        ("Alleged act", case_data.ALLEGED_ACT),
    ]
    tag_html = "".join(
        f'<div class="trib-tag"><span class="trib-tag-label">{html.escape(label)}</span>'
        f'<span class="trib-tag-value">{html.escape(value)}</span></div>'
        for label, value in tags
    )
    return f"""<div class="trib-overview-card">
  <div class="trib-tag-grid">{tag_html}</div>
  <div class="trib-issue-box">
    <span class="trib-issue-label">Tribunal Issue</span>
    <span class="trib-issue-value">{html.escape(case_data.TRIBUNAL_ISSUE)}</span>
  </div>
  <p class="trib-overview-scope">{html.escape(case_data.TRIBUNAL_SCOPE)}</p>
</div>"""


def _outcome_banner_html(judges: dict) -> str:
    """The Final Tribunal Outcome as a bold, definitive summary card: a
    large verdict label plus a compact per-judge ratio bar (one segment
    per judge, colored by that judge's individual vote) and a "N of 3
    Judges" counter — custom HTML rather than st.success/st.error so it
    can use the same subtle emerald/rose verdict tones as the pill badges
    elsewhere, which Streamlit gives no stable per-variant selector to
    apply to its own alert components.
    """
    votes = Counter(opinion["verdict"] for opinion in judges.values())
    majority_verdict, majority_count = votes.most_common(1)[0]
    is_justified = majority_verdict == "Justified"
    bg = "var(--trib-emerald-bg)" if is_justified else "var(--trib-rose-bg)"
    border = "var(--trib-emerald-border)" if is_justified else "var(--trib-rose-border)"
    fg = "var(--trib-emerald)" if is_justified else "var(--trib-rose)"
    label = html.escape(majority_verdict).upper()

    segments = "".join(
        '<span class="trib-ratio-seg" style="background:'
        + ("var(--trib-emerald)" if judges[key]["verdict"] == "Justified" else "var(--trib-rose)")
        + '"></span>'
        for key in JUDGE_ORDER
    )

    return f"""<div class="trib-outcome-card" style="background:{bg};border-color:{border}">
  <div class="trib-outcome-top">
    <span class="trib-outcome-eyebrow" style="color:{fg}">Final Tribunal Outcome</span>
    <span class="trib-outcome-verdict-large" style="color:{fg}">{label}</span>
  </div>
  <div class="trib-outcome-ratio">
    <div class="trib-ratio-bar">{segments}</div>
    <span class="trib-outcome-count">{majority_count} of 3 Judges</span>
  </div>
</div>"""


def _model_assignment_table_html(role_models: dict[str, str]) -> str:
    """The Custom-Agent per-role engine assignment as a clean 3-column
    SaaS data table (Role / Name / Assigned Engine) inside a card matching
    the rest of the design system — rendered by both render_result (live
    run) and render_historical_deliberation (a past Custom-Agent run),
    positioned between "Arguments of Record" and "Judicial Deliberation"
    in each. ``role_models`` maps persona key -> the model string assigned
    to that role.
    """
    rows = "".join(
        f"<tr><td>{html.escape(full_label)}</td>"
        f"<td>{html.escape(_persona_display_name(key))}</td>"
        f'<td><span class="trib-engine-chip">{html.escape(role_models.get(key, "—"))}</span></td></tr>'
        for key, full_label in ROLE_ASSIGNMENT_ORDER
    )
    return f"""<div class="trib-overview-card">
  <span class="trib-tag-label" style="display:block;margin-bottom:10px;">Model Assignment</span>
  <div class="trib-assign-table-wrap">
    <table class="trib-assign-table">
      <thead><tr><th>Role</th><th>Name</th><th>Assigned Engine / Model</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def _role_models_feed_lines_html(role_models: dict[str, str]) -> str:
    """The same per-role assignment as clean, indented "Name - model"
    lines (no role title) — minimalist gray text with the engine name in
    monospace — for the "Engines Used" expander inside a Custom-Agent
    Historical Runs feed card (see render_history_panel). Returns '' when
    there's nothing to show (row isn't a Custom-Agent run).
    """
    lines = "".join(
        '<div class="trib-feed-model-line">'
        f'<span class="trib-feed-model-name">{html.escape(_persona_display_name(key))}</span>'
        f' - <span class="trib-feed-model-engine">{html.escape(role_models[key])}</span>'
        "</div>"
        for key, _full_label in ROLE_ASSIGNMENT_ORDER
        if key in role_models
    )
    return f'<div class="trib-feed-models">{lines}</div>' if lines else ""


# ---------------------------------------------------------------------------
# Client / execution helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_client(api_key: str) -> OpenAI:
    return get_client(api_key)


def run_single_agent(api_key: str, model: str) -> dict:
    client = get_cached_client(api_key)
    tracker = CostTracker(model=model)

    start = time.perf_counter()
    verdict, response = create_structured_completion(
        client,
        model=model,
        schema=TribunalVerdict,
        messages=[{"role": "user", "content": build_single_prompt()}],
        response_format=JSON_OBJECT_RESPONSE_FORMAT,
        temperature=0.7,
        max_tokens=SINGLE_AGENT_MAX_TOKENS,
        extra_body=USAGE_ACCOUNTING_EXTRA_BODY,
    )
    execution_time = time.perf_counter() - start

    prompt_tokens, completion_tokens, actual_cost, executed_model = extract_usage(response)
    tracker.record(
        "single_call",
        prompt_tokens,
        completion_tokens,
        actual_cost_usd=actual_cost,
        executed_model=executed_model,
    )

    judges = {
        "barak": {"reasoning": verdict.judge_barak.reasoning, "verdict": verdict.judge_barak.verdict},
        "elon": {"reasoning": verdict.judge_elon.reasoning, "verdict": verdict.judge_elon.verdict},
        "shamgar": {"reasoning": verdict.judge_shamgar.reasoning, "verdict": verdict.judge_shamgar.verdict},
    }

    run_id = _persist_run(
        architecture_mode="single_agent",
        prosecution_summary=verdict.prosecution_summary,
        defense_summary=verdict.defense_summary,
        judges=judges,
        tracker=tracker,
        execution_time=execution_time,
    )

    return {
        "architecture": "single_agent",
        "requested_model": model,
        "executed_model": tracker.executed_model_summary,
        "prosecution_summary": verdict.prosecution_summary,
        "defense_summary": verdict.defense_summary,
        "judges": judges,
        "tracker": tracker,
        "execution_time": execution_time,
        "run_id": run_id,
    }


def run_multi_agent(api_key: str, model: str) -> dict:
    client = get_cached_client(api_key)
    tracker = CostTracker(model=model)

    start = time.perf_counter()
    orchestrator = TribunalOrchestrator(client=client, model=model, tracker=tracker)
    result = orchestrator.run()
    execution_time = time.perf_counter() - start

    judges = {
        key: {"reasoning": result.judge_opinions[key].reasoning, "verdict": result.judge_opinions[key].verdict}
        for key in JUDGE_ORDER
    }

    run_id = _persist_run(
        architecture_mode="multi_agent",
        prosecution_summary=result.prosecution_summary,
        defense_summary=result.defense_summary,
        judges=judges,
        tracker=tracker,
        execution_time=execution_time,
    )

    return {
        "architecture": "multi_agent",
        "requested_model": model,
        "executed_model": tracker.executed_model_summary,
        "prosecution_summary": result.prosecution_summary,
        "defense_summary": result.defense_summary,
        "judges": judges,
        "tracker": tracker,
        "execution_time": execution_time,
        "run_id": run_id,
    }


def run_custom_agent(api_key: str, models: dict[str, str]) -> dict:
    """Same 7-call workflow as ``run_multi_agent``, except each advocate/
    judge persona key gets its own engine from ``models`` (built by the
    Custom Agent selectors in render_controls_panel) instead of one shared
    model — dispatched via ``TribunalOrchestrator``'s optional ``models``
    override, which is the only thing that changes per role.
    """
    client = get_cached_client(api_key)
    # No single "model" applies to a custom run — CostTracker.model is only
    # a fallback for pricing estimation / a call missing its own
    # executed_model, and every call here reports its own via extract_usage,
    # so this label is never actually used for cost math.
    tracker = CostTracker(model="custom-agent")

    start = time.perf_counter()
    orchestrator = TribunalOrchestrator(
        client=client, model=DEFAULT_MODEL, tracker=tracker, models=models
    )
    result = orchestrator.run()
    execution_time = time.perf_counter() - start

    judges = {
        key: {"reasoning": result.judge_opinions[key].reasoning, "verdict": result.judge_opinions[key].verdict}
        for key in JUDGE_ORDER
    }

    run_id = _persist_run(
        architecture_mode="custom_agent",
        prosecution_summary=result.prosecution_summary,
        defense_summary=result.defense_summary,
        judges=judges,
        tracker=tracker,
        execution_time=execution_time,
        role_models=models,
    )

    return {
        "architecture": "custom_agent",
        "requested_model": ", ".join(sorted(set(models.values()))) if models else "custom",
        "executed_model": tracker.executed_model_summary,
        "prosecution_summary": result.prosecution_summary,
        "defense_summary": result.defense_summary,
        "judges": judges,
        "tracker": tracker,
        "execution_time": execution_time,
        "run_id": run_id,
        "custom_models": dict(models),
    }


def _persist_run(
    architecture_mode: str,
    prosecution_summary: str,
    defense_summary: str,
    judges: dict,
    tracker: CostTracker,
    execution_time: float,
    role_models: dict[str, str] | None = None,
) -> int:
    record = TrialRunRecord(
        architecture_mode=architecture_mode,
        model=tracker.executed_model_summary,
        prosecution_summary=prosecution_summary,
        defense_summary=defense_summary,
        judge_barak_reasoning=judges["barak"]["reasoning"],
        judge_barak_verdict=judges["barak"]["verdict"],
        judge_elon_reasoning=judges["elon"]["reasoning"],
        judge_elon_verdict=judges["elon"]["verdict"],
        judge_shamgar_reasoning=judges["shamgar"]["reasoning"],
        judge_shamgar_verdict=judges["shamgar"]["verdict"],
        prompt_tokens=tracker.prompt_tokens,
        completion_tokens=tracker.completion_tokens,
        total_tokens=tracker.total_tokens,
        cost_usd=tracker.cost_usd,
        cost_ils=tracker.cost_ils,
        execution_time_sec=execution_time,
        role_models_json=json.dumps(role_models) if role_models else None,
    )
    db_path = init_db()
    return log_trial_run(record, db_path=db_path)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_judges(judges: dict) -> None:
    cols = st.columns(3)
    for col, key in zip(cols, JUDGE_ORDER):
        persona = JUDGE_PERSONAS[key]
        opinion = judges[key]
        with col:
            st.markdown(_judge_card_html(persona, opinion), unsafe_allow_html=True)


def render_majority_outcome(judges: dict) -> None:
    # The card itself (see _outcome_banner_html) carries its own bold
    # "Final Tribunal Outcome" eyebrow + large verdict label, so no
    # separate st.subheader is needed above it.
    st.markdown(_outcome_banner_html(judges), unsafe_allow_html=True)
    st.caption(
        "The tribunal issues no single collective opinion — each judge's "
        "reasoning above stands independently. This is a simple majority "
        "count of the three otherwise-unmerged verdicts, shown for "
        "convenience."
    )


def render_budget_box(tracker: CostTracker, execution_time: float, run_id: int) -> None:
    rate = get_ils_exchange_rate()
    st.subheader("Budget Summary")
    st.markdown(
        _budget_box_html(
            [
                ("Execution time", f"{execution_time:.2f} sec"),
                ("Total tokens", f"{tracker.total_tokens:,}"),
                ("Estimated cost (USD)", f"${tracker.cost_usd:.6f}"),
                ("Estimated cost (ILS)", f"₪{tracker.cost_ils:.6f}"),
                ("Prompt tokens", f"{tracker.prompt_tokens:,}"),
                ("Completion tokens", f"{tracker.completion_tokens:,}"),
            ],
            run_id,
        ),
        unsafe_allow_html=True,
    )

    st.caption(f"ILS rate used: {rate:.2f}  •  Database: `{get_db_path()}`")

    if len(tracker.calls) > 1:
        with st.expander("Per-call token breakdown"):
            breakdown_df = pd.DataFrame(
                [
                    {
                        "Agent call": c.label,
                        "Prompt tokens": c.prompt_tokens,
                        "Completion tokens": c.completion_tokens,
                        "Total tokens": c.total_tokens,
                        "Cost (USD)": round(c.cost_usd, 6),
                    }
                    for c in tracker.calls
                ]
            )
            st.dataframe(breakdown_df, width="stretch", hide_index=True)


def render_result(result: dict) -> None:
    label = ARCHITECTURE_LABELS.get(result["architecture"], "Multi-Agent")
    st.markdown(f"### Results — {label} Architecture")
    st.markdown(f"**Engine:** `{result['executed_model']}`")

    with st.expander("Arguments of Record", expanded=False):
        col_p, col_d = st.columns(2)
        with col_p:
            st.markdown(
                _advocate_card_html("Prosecution", "prosecution", result["prosecution_summary"]),
                unsafe_allow_html=True,
            )
        with col_d:
            st.markdown(
                _advocate_card_html("Defense", "defense", result["defense_summary"]),
                unsafe_allow_html=True,
            )

    if result["architecture"] == "custom_agent" and result.get("custom_models"):
        st.markdown(
            _model_assignment_table_html(result["custom_models"]), unsafe_allow_html=True
        )

    st.subheader("Judicial Deliberation")
    render_judges(result["judges"])

    render_majority_outcome(result["judges"])

    render_budget_box(result["tracker"], result["execution_time"], result["run_id"])


# ---------------------------------------------------------------------------
# Left column, top section — Tribunal Controls / Launch Chamber
# ---------------------------------------------------------------------------
#
# Mode/model selection and the two run triggers live at the very top of the
# left column, directly above the Historical Runs feed (see
# render_history_panel below). The right column stays strictly a content
# view — see render_chamber_panel — with no controls of its own. API-key
# loading here is exactly the last known-working expression:
# os.getenv("OPENROUTER_API_KEY"), populated by load_dotenv() above.

def _custom_model_index(pool: list[str]) -> int:
    """Index of CUSTOM_AGENT_DEFAULT_MODEL in ``pool``, or 0 if absent."""
    return pool.index(CUSTOM_AGENT_DEFAULT_MODEL) if CUSTOM_AGENT_DEFAULT_MODEL in pool else 0


def render_controls_panel():
    """Renders mode/model selection and the run buttons inside one
    elevated white card; returns the ``st.empty()`` spinner-slot
    placeholder that ``_execute_pending_run`` (called later from
    ``main()``) fills in once this whole page has rendered — see that
    placeholder's own comment below for why.
    """
    is_running = st.session_state.get("is_running", False)

    # Wraps the whole form in one clean, elevated white card (12px
    # radius — see the [data-testid="stVerticalBlockBorderWrapper"] rule
    # in THEME_CSS). Purely a visual grouping: every widget inside behaves
    # exactly as it did outside a container, same keys, same session
    # state, same click handling below.
    with st.container(border=True):
        st.markdown(
            '<div class="trib-panel-header"><span class="trib-panel-title">Tribunal Controls</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Launch Chamber — configure the engine, then run a simulation.")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.error(
                "OPENROUTER_API_KEY not found. Copy `.env.example` to `.env` in "
                "the project root and add your key from https://openrouter.ai/keys, "
                "then restart Streamlit."
            )
        if is_running:
            st.warning("🔒 A simulation is running — controls are locked until it finishes.")

        # Stacked, not side-by-side: this panel lives in the narrow left
        # column, so a horizontal row of controls would cramp/wrap instead
        # of reading cleanly.
        selection_mode = st.radio(
            "Mode",
            options=["Random per run", "Fixed", "Custom Agent"],
            index=0,
            help=(
                "Random per run (default): each time you click a Run button, a "
                "model is drawn at random from a diverse pool spanning several "
                "providers. Fixed: always use the model chosen below. Custom "
                "Agent: assign a distinct engine to each advocate and judge role."
            ),
            disabled=is_running,
        )
        random_mode = selection_mode == "Random per run"
        custom_mode = selection_mode == "Custom Agent"

        model: str | None = None
        custom_models: dict[str, str] = {}

        if custom_mode:
            # Granular per-role engine assignment — one selector per
            # advocate and judge persona, each defaulting to
            # CUSTOM_AGENT_DEFAULT_MODEL. Keyed by persona key (not role
            # label) so TribunalOrchestrator's ``models`` override (see
            # project_multi_agent/orchestrator.py) can look each one up
            # directly by the same key it already iterates
            # ADVOCATE_PERSONAS / JUDGE_PERSONAS by.
            st.caption("Assign an engine to each advocate and judge role.")
            advocate_roles = [
                (PROSECUTION_KEYS[0], "Prosecutor 1"),
                (PROSECUTION_KEYS[1], "Prosecutor 2"),
                (DEFENSE_KEYS[0], "Defense 1"),
                (DEFENSE_KEYS[1], "Defense 2"),
            ]
            for key, role_label in advocate_roles:
                persona_name = ADVOCATE_PERSONAS[key]["name"]
                custom_models[key] = st.selectbox(
                    f"{role_label} — {persona_name}",
                    options=VERIFIED_STABLE_POOL,
                    index=_custom_model_index(VERIFIED_STABLE_POOL),
                    key=f"custom_model_{key}",
                    disabled=is_running,
                )
            judge_roles = list(zip(JUDGE_ORDER, ["Judge 1", "Judge 2", "Judge 3"]))
            for key, role_label in judge_roles:
                persona_name = JUDGE_PERSONAS[key]["name"]
                custom_models[key] = st.selectbox(
                    f"{role_label} — {persona_name}",
                    options=VERIFIED_STABLE_POOL,
                    index=_custom_model_index(VERIFIED_STABLE_POOL),
                    key=f"custom_model_{key}",
                    disabled=is_running,
                )
        elif random_mode:
            # Deliberately no dropdown and no list of candidate engines
            # here — the whole point of "Random per run" is that the user
            # doesn't know (and isn't shown) which engine will answer
            # until after the run.
            st.caption("An engine will be picked at random when you click Run.")
        else:
            # Fixed mode only offers the verified-stable pool — no
            # "openrouter/auto" (a meta-router, not a specific engine to
            # pick) and no experimental/unproven entries from the dynamic
            # pool.
            model = st.selectbox(
                "OpenRouter model",
                options=VERIFIED_STABLE_POOL,
                index=VERIFIED_STABLE_POOL.index(DEFAULT_MODEL) if DEFAULT_MODEL in VERIFIED_STABLE_POOL else 0,
                disabled=is_running,
            )

        run_disabled = is_running or not api_key

        if custom_mode:
            run_custom_clicked = st.button(
                "🧩 Run Custom-Agent", width="stretch", disabled=run_disabled
            )
            run_single_clicked = False
            run_multi_clicked = False
        else:
            run_single_clicked = st.button(
                "⚖️ Run Single-Agent", width="stretch", disabled=run_disabled
            )
            run_multi_clicked = st.button(
                "🏛️ Run Multi-Agent", width="stretch", disabled=run_disabled
            )
            run_custom_clicked = False

        st.caption(
            "Single-Agent = 1 monolithic call.  •  Multi-Agent = 7 calls "
            "(4 advocates + 3 independent judges)."
        )

        any_clicked = run_single_clicked or run_multi_clicked or run_custom_clicked

        if is_running:
            st.info("🔒 A simulation is currently running. Please wait for it to finish.")
        elif any_clicked and not api_key:
            st.error("Cannot run: OPENROUTER_API_KEY is not set. See the controls above for setup instructions.")
        elif any_clicked:
            # Engage the execution lock and stash which run to perform,
            # then rerun immediately so the browser sees the controls
            # above as disabled *before* the (blocking) run actually
            # starts — otherwise a second click during the run could
            # interrupt it mid-flight.
            if run_custom_clicked:
                st.session_state["pending_run"] = "custom_agent"
                st.session_state["pending_model"] = None
                st.session_state["pending_custom_models"] = dict(custom_models)
            else:
                st.session_state["pending_run"] = "single_agent" if run_single_clicked else "multi_agent"
                st.session_state["pending_model"] = pick_random_model() if random_mode else model
                st.session_state["pending_custom_models"] = None
            st.session_state["is_running"] = True
            st.rerun()

        # Reserve this run's spinner slot right here — the same spot it's
        # always occupied — but don't actually perform the blocking call
        # yet; that happens in _execute_pending_run(), invoked from
        # main() only after the *entire* page (including the now
        # disabled/locked Historical Runs controls below) has finished
        # rendering for this pass. Doing the blocking work here instead
        # would mean the `st.rerun()` inside it cuts this pass short
        # before render_history_panel ever runs with is_running=True, so
        # its disabled buttons would never actually reach the browser.
        spinner_slot = st.empty()

        if st.session_state.get("run_error"):
            st.error(st.session_state["run_error"])
            st.session_state["run_error"] = None

    return spinner_slot


_RUN_SPINNER_LABELS: dict[str, str] = {
    "single_agent": "Running single-agent tribunal simulation...",
    "multi_agent": "Running multi-agent tribunal simulation...",
    "custom_agent": "Running custom-agent tribunal simulation...",
}


def _execute_pending_run(spinner_slot) -> None:
    """Perform the actual blocking OpenRouter call(s) for whichever run
    render_controls_panel armed via its click handler — called from
    main() after the whole page has rendered (see the comment above
    ``spinner_slot`` in render_controls_panel for why the ordering
    matters). ``finally`` always drops the execution lock and reruns,
    regardless of success or failure, which is what re-enables the
    Historical Runs controls once this completes or errors.
    """
    is_running = st.session_state.get("is_running", False)
    pending_run = st.session_state.get("pending_run")
    if not (is_running and pending_run):
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = st.session_state.get("pending_model")
    active_custom_models = st.session_state.get("pending_custom_models")
    try:
        with spinner_slot, st.spinner(_RUN_SPINNER_LABELS.get(pending_run, "Running tribunal simulation...")):
            if pending_run == "single_agent":
                st.session_state["last_result"] = run_single_agent(api_key, active_model)
            elif pending_run == "multi_agent":
                st.session_state["last_result"] = run_multi_agent(api_key, active_model)
            else:  # "custom_agent"
                st.session_state["last_result"] = run_custom_agent(api_key, active_custom_models or {})
        st.session_state["run_error"] = None
        st.session_state["selected_view"] = "active"
    except Exception as exc:  # noqa: BLE001
        arch_label = {
            "single_agent": "Single-agent",
            "multi_agent": "Multi-agent",
            "custom_agent": "Custom-agent",
        }.get(pending_run, "Run")
        # Stash the error rather than calling st.error() here directly —
        # the st.rerun() below would wipe it from the screen before the
        # user ever saw it, since this whole branch's render is discarded.
        st.session_state["run_error"] = f"{arch_label} run failed: {exc}"
    finally:
        st.session_state["is_running"] = False
        st.session_state["pending_run"] = None
        st.session_state["pending_model"] = None
        st.session_state["pending_custom_models"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Left column, bottom section — Historical Runs & Audit Trail
# ---------------------------------------------------------------------------

def _load_history_df() -> pd.DataFrame:
    db_path = init_db()
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM trial_runs ORDER BY id DESC", conn)
        except Exception:
            return pd.DataFrame()


def render_history_panel(is_running: bool) -> None:
    df = _load_history_df()

    st.markdown(
        f'<div class="trib-panel-header">'
        f'<span class="trib-panel-title">Historical Runs</span>'
        f'<span class="trib-count-pill">{len(df)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Audit trail of every logged simulation, most recent first.")

    # Dedicated toggle back to the live chamber — sits above the feed so
    # it's reachable regardless of how far the user has scrolled, and
    # visibly "lit" (blue) whenever a historical run is what's actually
    # showing on the right, so it doubles as a status indicator. Clicking
    # it fully resets to a clean simulation state: the historical
    # selection is cleared *and* any stale live output is cleared too, so
    # the right panel comes back empty and ready for a new run rather than
    # silently re-showing whatever was last run.
    # Disabled (not hidden) while a simulation is actively running: the
    # feed stays visible so the user can still see what's logged, but
    # every interactive control below — this toggle and each row's "View"
    # button — is inert until is_running flips back to False, which
    # happens in the controls panel's `finally` block regardless of
    # whether the run succeeded or raised, so this re-enables on both
    # completion and error.
    selected_view = st.session_state.get("selected_view", "active")
    is_history_view = selected_view == "history"
    active_wrapper_class = "trib-return-live--active" if is_history_view else ""
    st.markdown(f'<div class="trib-return-live {active_wrapper_class}">', unsafe_allow_html=True)
    if st.button(
        "🏛️ Return to Live Chamber" if is_history_view else "🏛️ Active Chamber & Bench",
        key="return_to_live_chamber",
        width="stretch",
        disabled=is_running or not is_history_view,
    ):
        st.session_state["selected_view"] = "active"
        st.session_state["hist_loaded_run_id"] = None
        st.session_state["last_result"] = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if is_running:
        st.info("🔒 History browsing is locked while a simulation is running.")

    if df.empty:
        st.caption("No runs logged yet. Run a simulation above to populate this feed.")
        return

    # A fixed-height scrollable feed (Streamlit's nearest equivalent to
    # `max-height: 80vh; overflow-y: auto` — st.container only accepts a
    # pixel height, not a viewport-relative one).
    with st.container(height=600, border=False):
        for row in df.itertuples():
            votes = Counter(
                [row.judge_barak_verdict, row.judge_elon_verdict, row.judge_shamgar_verdict]
            )
            majority_verdict, _ = votes.most_common(1)[0]
            is_justified = majority_verdict == "Justified"
            mode_label = ARCHITECTURE_LABELS.get(row.architecture_mode, "Multi-Agent")
            pill_class = "trib-pill--justified" if is_justified else "trib-pill--not-justified"
            pill_label = html.escape(majority_verdict).upper()
            # Custom-Agent rows: the per-role engines, readable name and
            # all, tucked into a collapsed expander so the card stays
            # compact — see the "Engines Used" expander below.
            role_models = (
                _parse_role_models(getattr(row, "role_models_json", ""))
                if row.architecture_mode == "custom_agent"
                else {}
            )

            card_col, action_col = st.columns([0.78, 0.22])
            with card_col:
                st.markdown(
                    f"""<div class="trib-feed-item">
  <div class="trib-feed-row">
    <span class="trib-feed-time">{html.escape(str(row.timestamp))}</span>
    <span class="trib-feed-engine">{html.escape(str(row.model or 'unknown'))}</span>
  </div>
  <div class="trib-feed-row">
    <span class="trib-pill trib-pill--mode">{html.escape(mode_label)}</span>
    <span class="trib-pill {pill_class}">{pill_label}</span>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )
                if role_models:
                    with st.expander(
                        "Engines Used", expanded=False, key=f"engine_cfg_{row.id}"
                    ):
                        st.markdown(
                            _role_models_feed_lines_html(role_models), unsafe_allow_html=True
                        )
            with action_col:
                st.markdown('<div class="trib-feed-row-actions">', unsafe_allow_html=True)
                if st.button("View", key=f"hist_view_{row.id}", width="stretch", disabled=is_running):
                    # Selecting any past run immediately resets the live
                    # chamber: clearing last_result here (not just
                    # switching selected_view) guarantees the right panel
                    # can never show a stale live result underneath/behind
                    # the historical breakdown — it's already logged in
                    # court_runs.db, so nothing is actually lost.
                    st.session_state["hist_loaded_run_id"] = row.id
                    st.session_state["selected_view"] = "history"
                    st.session_state["last_result"] = None
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            # Pure spacer between bordered feed cards, not a visible rule —
            # each card already carries its own border, so a second line
            # here would just compete with it.
            st.markdown('<hr style="margin:8px 0;border:none;height:0;">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Right column — Simulation Chamber & Judicial Bench
# ---------------------------------------------------------------------------

def render_historical_deliberation(run_id: int) -> None:
    df = _load_history_df()
    matches = df[df["id"] == run_id] if not df.empty else df
    if matches.empty:
        st.caption("Selected run no longer exists.")
        return

    row = matches.iloc[0]
    judges = {
        "barak": {"reasoning": row["judge_barak_reasoning"], "verdict": row["judge_barak_verdict"]},
        "elon": {"reasoning": row["judge_elon_reasoning"], "verdict": row["judge_elon_verdict"]},
        "shamgar": {"reasoning": row["judge_shamgar_reasoning"], "verdict": row["judge_shamgar_verdict"]},
    }
    label = ARCHITECTURE_LABELS.get(row["architecture_mode"], "Multi-Agent")

    # Same reset as the left-panel toggle (see render_history_panel), also
    # available right here at the top of the historical view itself so the
    # user doesn't have to scroll back to the left column to leave it.
    if st.button("← Back to New Simulation / Live Chamber", key="hist_back_to_live_top"):
        st.session_state["selected_view"] = "active"
        st.session_state["hist_loaded_run_id"] = None
        st.session_state["last_result"] = None
        st.rerun()

    st.markdown(f"### Run #{run_id} — {label} Architecture")
    st.caption(f"Logged {row['timestamp']}")
    st.markdown(f"**Engine:** `{row['model'] or 'unknown'}`")

    with st.expander("Arguments of Record", expanded=False):
        col_p, col_d = st.columns(2)
        with col_p:
            st.markdown(
                _advocate_card_html("Prosecution", "prosecution", row["prosecution_summary"]),
                unsafe_allow_html=True,
            )
        with col_d:
            st.markdown(
                _advocate_card_html("Defense", "defense", row["defense_summary"]),
                unsafe_allow_html=True,
            )

    if row["architecture_mode"] == "custom_agent":
        role_models = _parse_role_models(row.get("role_models_json"))
        if role_models:
            st.markdown(_model_assignment_table_html(role_models), unsafe_allow_html=True)

    st.subheader("Judicial Deliberation")
    render_judges(judges)
    render_majority_outcome(judges)

    rate = get_ils_exchange_rate()
    st.subheader("Budget Summary")
    st.markdown(
        _budget_box_html(
            [
                ("Execution time", f"{row['execution_time_sec']:.2f} sec"),
                ("Total tokens", f"{int(row['total_tokens']):,}"),
                ("Estimated cost (USD)", f"${row['cost_usd']:.6f}"),
                ("Estimated cost (ILS)", f"₪{row['cost_ils']:.6f}"),
                ("Prompt tokens", f"{int(row['prompt_tokens']):,}"),
                ("Completion tokens", f"{int(row['completion_tokens']):,}"),
            ],
            run_id,
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"ILS rate: {rate:.2f}")


def render_chamber_panel() -> None:
    st.markdown(
        '<div class="trib-panel-header"><span class="trib-panel-title">Simulation Chamber &amp; Judicial Bench</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(case_data.CASE_ID)

    st.markdown(_case_overview_card_html(), unsafe_allow_html=True)

    with st.expander("Case Facts & Arguments", expanded=False):
        st.markdown("**Agreed Facts**")
        for i, fact in enumerate(case_data.AGREED_FACTS, start=1):
            st.markdown(f"{i}. {fact}")
        st.markdown("**Prosecution Arguments**")
        for point in case_data.PROSECUTION_ARGUMENTS:
            st.markdown(f"- {point}")
        st.markdown("**Defense Arguments**")
        for point in case_data.DEFENSE_ARGUMENTS:
            st.markdown(f"- {point}")

    st.divider()

    # Dynamic state switching: the "active" simulation chamber shows this
    # session's most recent run, while "history" (set via a feed item's
    # "View" button, which also clears last_result — see
    # render_history_panel) replaces this whole section with a read-only
    # breakdown of a past run instead. "Back to New Simulation / Live
    # Chamber" (in the left column, and at the top of the historical view
    # itself) resets both back to a clean slate.
    selected_view = st.session_state.get("selected_view", "active")
    if selected_view == "history" and st.session_state.get("hist_loaded_run_id") is not None:
        render_historical_deliberation(st.session_state["hist_loaded_run_id"])
    elif st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])
    else:
        st.caption("No deliberation yet. Run a simulation on the left, or select a historical run to inspect it.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "pending_run" not in st.session_state:
        st.session_state["pending_run"] = None
    if "pending_model" not in st.session_state:
        st.session_state["pending_model"] = None
    if "pending_custom_models" not in st.session_state:
        st.session_state["pending_custom_models"] = None
    if "run_error" not in st.session_state:
        st.session_state["run_error"] = None
    # Which run the right-hand "Simulation Chamber & Judicial Bench" panel
    # is currently displaying: "active" (the most recent run from this
    # session) or "history" (a run loaded from the left-hand feed via its
    # "View" button — see render_history_panel / render_historical_deliberation).
    # Cleared back to "active" — along with hist_loaded_run_id and
    # last_result — by the "Active Chamber & Bench" toggle at the top of
    # the history feed, or the matching button atop the historical view.
    if "selected_view" not in st.session_state:
        st.session_state["selected_view"] = "active"
    if "hist_loaded_run_id" not in st.session_state:
        st.session_state["hist_loaded_run_id"] = None

    st.markdown(
        """<div class="trib-topbar">
  <div class="trib-topbar-left">
    <span class="trib-topbar-icon">⚖️</span>
    <span class="trib-topbar-title">The Tribunal</span>
  </div>
  <span class="trib-topbar-pill">Unmerged Protocol v2.5</span>
</div>""",
        unsafe_allow_html=True,
    )

    is_running = st.session_state.get("is_running", False)

    left_col, right_col = st.columns([1, 2.3], gap="large")

    # Left column: Tribunal Controls / Launch Chamber up top, Historical
    # Runs underneath.
    with left_col:
        spinner_slot = render_controls_panel()
        st.divider()
        render_history_panel(is_running)

    # Right column stays strictly a content view: the live chamber /
    # judicial bench, or a selected historical run's breakdown — no
    # controls of its own (see render_chamber_panel).
    with right_col:
        render_chamber_panel()

    # The actual blocking OpenRouter call(s), if a run is armed — invoked
    # only now, after the whole page above (including render_history_panel
    # with is_running=True, so its "View" buttons and the "Active Chamber
    # & Bench" toggle are genuinely disabled in the browser) has finished
    # rendering for this pass. See _execute_pending_run's docstring.
    _execute_pending_run(spinner_slot)


if __name__ == "__main__":
    main()
