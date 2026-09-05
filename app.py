"""Streamlit GUI for the Tribunal simulation.

Lets a user trigger a Single-Agent or Multi-Agent run from the browser,
watch the deliberation render live, and browse past runs logged in
court_runs.db — instead of driving everything from the command line.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
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
from common.personas import JUDGE_ORDER, JUDGE_PERSONAS
from project_multi_agent.orchestrator import TribunalOrchestrator
from project_single_agent.main import TribunalVerdict, build_prompt as build_single_prompt

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_MODEL = get_model()

st.set_page_config(
    page_title="Tribunal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme — extracted from design/Tribunal Bench.dc.html (the exported mockup):
# dark-slate ground (#0f172a) with elevated card panels (#131d33), Spectral
# serif for headings/card names, IBM Plex Sans for body copy, IBM Plex Mono
# (uppercase, letter-spaced) for eyebrow labels/badges/metrics, and judicial
# gold (#f59e0b / #d97706) as the sole accent color for actions and emphasis.
# Presentation-only: purely CSS plus, further below, small pure-string HTML
# builders for the judge/advocate cards. Nothing here touches request
# routing, model selection, parsing, or database logic.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --trib-bg: #0f172a;
    --trib-panel: #131d33;
    --trib-sidebar: #111a2e;
    --trib-border: #1e293b;
    --trib-border-strong: #334155;
    --trib-border-gold: #78350f;
    --trib-text: #f8fafc;
    --trib-text-heading: #f1f5f9;
    --trib-text-body: #cbd5e1;
    --trib-text-muted: #94a3b8;
    --trib-text-dim: #64748b;
    --trib-gold: #f59e0b;
    --trib-gold-deep: #d97706;
    --trib-gold-bright: #fbbf24;
    --trib-gold-pale: #fcd34d;
    --trib-green: #34d399;
    --trib-red: #f87171;
    --trib-red-pale: #fca5a5;
    --trib-red-deep: #b91c1c;
    --trib-cyan: #67e8f9;
    --trib-cyan-deep: #0e7490;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stHeader"] {
    background: var(--trib-bg) !important;
}
.stApp, .stApp p, .stApp li, .stApp label {
    color: var(--trib-text-body);
    font-family: 'IBM Plex Sans', Helvetica, sans-serif;
}

h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Spectral', Georgia, serif !important;
    font-weight: 500 !important;
    color: var(--trib-text) !important;
    letter-spacing: 0.01em;
}
h1 { border-left: 6px solid var(--trib-gold-deep); padding-left: 14px; }

.stApp small, [data-testid="stMarkdownContainer"] small {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.05em;
    color: var(--trib-text-dim) !important;
    text-transform: uppercase;
    font-size: 0.78rem !important;
}

[data-testid="stSidebar"] {
    background: var(--trib-sidebar) !important;
    border-right: 1px solid var(--trib-border);
}
[data-testid="stSidebar"] .stApp,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--trib-text-body);
}

.stButton > button {
    background: linear-gradient(180deg, var(--trib-gold) 0%, var(--trib-gold-deep) 100%) !important;
    color: #1c1207 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 1px 0 rgba(255,255,255,0.18) inset;
}
.stButton > button:hover:not(:disabled) { filter: brightness(1.08); color: #1c1207 !important; }
.stButton > button:disabled { background: var(--trib-border-strong) !important; color: var(--trib-text-dim) !important; }

[data-testid="stTabs"] button[role="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    color: var(--trib-text-dim);
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--trib-gold-bright) !important;
    border-bottom-color: var(--trib-gold-deep) !important;
}

[data-testid="stExpander"] {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border) !important;
    border-radius: 3px;
}

[data-testid="stMetric"] {
    background: var(--trib-panel);
    border: 1px solid var(--trib-border);
    border-radius: 3px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10.5px !important;
    letter-spacing: 0.1em;
    color: var(--trib-text-dim) !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--trib-text) !important;
}

[data-testid="stAlert"] { border-radius: 3px; font-family: 'IBM Plex Sans', sans-serif; }

hr { border-color: var(--trib-border) !important; }

[data-baseweb="select"] > div { background: var(--trib-bg) !important; border-color: var(--trib-border-strong) !important; }

/* --- Tribunal card system (judge / advocate cards, built as raw HTML below) --- */
.trib-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.14em;
    color: var(--trib-text-dim);
    text-transform: uppercase;
}
.trib-card {
    border: 1px solid var(--trib-border);
    background: var(--trib-panel);
    border-radius: 3px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 14px;
    height: 100%;
    box-sizing: border-box;
}
.trib-card--prosecution { border-top: 2px solid var(--trib-red-deep); }
.trib-card--defense { border-top: 2px solid var(--trib-cyan-deep); }
.trib-card-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.trib-card-name { font-family: 'Spectral', Georgia, serif; font-size: 16.5px; color: var(--trib-text); }
.trib-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    padding: 4px 8px;
    border-radius: 2px;
    text-transform: uppercase;
    white-space: nowrap;
    align-self: flex-start;
}
.trib-badge--prosecution { color: var(--trib-red-pale); border: 1px solid #7f1d1d; }
.trib-badge--defense { color: var(--trib-cyan); border: 1px solid #155e75; }
.trib-badge--doctrine { color: var(--trib-gold-pale); border: 1px solid var(--trib-border-gold); background: rgba(217,119,6,0.1); }
.trib-card-body { font-size: 13.5px; line-height: 1.65; color: var(--trib-text-body); margin: 0; }
.trib-verdict-row {
    display: flex; align-items: center; gap: 9px;
    padding: 9px 11px; border-radius: 2px;
    background: var(--trib-bg); border: 1px solid var(--trib-border);
}
.trib-dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
.trib-verdict-text { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.1em; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


def _judge_card_html(persona: dict, opinion: dict) -> str:
    """Render one judge as a card matching design/Tribunal Bench.dc.html's
    Judicial Bench article: name in Spectral, a gold doctrine badge, a
    verdict row with a colored status dot, and the full reasoning text.
    All interpolated text is HTML-escaped since it can originate from LLM
    output, not just our own literal strings.
    """
    is_justified = opinion["verdict"] == "Justified"
    dot_color = "var(--trib-green)" if is_justified else "var(--trib-red)"
    verdict_label = html.escape(opinion["verdict"]).upper()
    name = html.escape(persona["name"])
    doctrine = html.escape(persona["model_label"])
    reasoning = html.escape(opinion["reasoning"]).replace("\n", "<br>")
    return f"""<div class="trib-card">
  <div class="trib-card-header">
    <div class="trib-card-name">{name}</div>
  </div>
  <span class="trib-badge trib-badge--doctrine">{doctrine}</span>
  <div class="trib-verdict-row">
    <span class="trib-dot" style="background:{dot_color}"></span>
    <span class="trib-verdict-text" style="color:{dot_color}">{verdict_label}</span>
  </div>
  <p class="trib-card-body">{reasoning}</p>
</div>"""


def _advocate_card_html(title: str, side: str, text: str) -> str:
    """Render a prosecution/defense argument as a card matching the
    mockup's Arguments of Record articles: a colored top border and badge
    per side (red/PROSECUTION, cyan/DEFENSE), name in Spectral, body text
    in the card. ``text`` is HTML-escaped since it's LLM-generated.
    """
    is_prosecution = side == "prosecution"
    card_class = "trib-card--prosecution" if is_prosecution else "trib-card--defense"
    badge_class = "trib-badge--prosecution" if is_prosecution else "trib-badge--defense"
    badge_label = "PROSECUTION" if is_prosecution else "DEFENSE"
    body = html.escape(text).replace("\n", "<br>")
    return f"""<div class="trib-card {card_class}">
  <div class="trib-card-header">
    <div class="trib-card-name">{html.escape(title)}</div>
    <span class="trib-badge {badge_class}">{badge_label}</span>
  </div>
  <p class="trib-card-body">{body}</p>
</div>"""


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


def _persist_run(
    architecture_mode: str,
    prosecution_summary: str,
    defense_summary: str,
    judges: dict,
    tracker: CostTracker,
    execution_time: float,
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
    )
    db_path = init_db()
    return log_trial_run(record, db_path=db_path)


# ---------------------------------------------------------------------------
# Accordion / mutually-exclusive collapsible state
# ---------------------------------------------------------------------------
#
# Both layers here use the same pattern: a plain st.button + a single
# session_state variable naming whichever section is currently open (never
# two independently-toggled booleans kept in sync via callbacks). A real
# st.expander with a keyed on_change *looks* like the natural fit for
# mutual exclusion, but in practice it's unreliable for this: two sibling
# expanders each flipping the other's keyed state via on_change can lag a
# render behind the click (the widget's own instantiation on this same
# script pass can re-assert a value the callback just changed), producing
# exactly the "takes two clicks" / flicker behavior this exists to avoid.
# A single-variable "which one is active" model has no such race — setting
# it to "arguments" makes "deliberation" inactive by construction, in the
# same click, with no second widget's callback in the loop. This is also
# what already drives the top-level Run/History accordion below, and it
# has shown no such glitch.


def _toggle_top_section(open_key: str, other_key: str) -> None:
    """Flip ``open_key``'s open/closed flag; if it just opened, force
    ``other_key`` closed so only one top-level section is open at once."""
    st.session_state[open_key] = not st.session_state.get(open_key, False)
    if st.session_state[open_key]:
        st.session_state[other_key] = False


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
    votes = Counter(opinion["verdict"] for opinion in judges.values())
    majority_verdict, majority_count = votes.most_common(1)[0]
    st.subheader("Final Tribunal Outcome (Majority Rule)")
    if majority_verdict == "Justified":
        st.success(f"**{majority_verdict}** — {majority_count} of 3 judges")
    else:
        st.error(f"**{majority_verdict}** — {majority_count} of 3 judges")
    st.caption(
        "The tribunal issues no single collective opinion — each judge's "
        "reasoning above stands independently. This is a simple majority "
        "count of the three otherwise-unmerged verdicts, shown for "
        "convenience."
    )


def render_budget_box(tracker: CostTracker, execution_time: float, run_id: int) -> None:
    rate = get_ils_exchange_rate()
    st.subheader("Budget Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Execution time", f"{execution_time:.2f} sec")
    m2.metric("Total tokens", f"{tracker.total_tokens:,}")
    m3.metric("Estimated cost (USD)", f"${tracker.cost_usd:.6f}")
    m4.metric("Estimated cost (ILS)", f"₪{tracker.cost_ils:.6f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Prompt tokens", f"{tracker.prompt_tokens:,}")
    d2.metric("Completion tokens", f"{tracker.completion_tokens:,}")
    d3.metric("SQLite Run ID", run_id)

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
    label = "Single-Agent" if result["architecture"] == "single_agent" else "Multi-Agent"
    st.markdown(f"### Results — {label} Architecture")
    st.caption("The prosecution's and defense's arguments from this run are shown inside the “Canonical Case Facts” section above.")

    st.markdown(f"**Engine:** `{result['executed_model']}`")

    st.subheader("Judicial Deliberation")
    render_judges(result["judges"])

    render_majority_outcome(result["judges"])

    render_budget_box(result["tracker"], result["execution_time"], result["run_id"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str | None, str | None, bool]:
    st.sidebar.title("⚖️ Tribunal Controls")

    is_running = st.session_state.get("is_running", False)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.sidebar.error("OPENROUTER_API_KEY not found")
        st.sidebar.caption(
            "Copy `.env.example` to `.env` in the project root and add your "
            "key from https://openrouter.ai/keys, then restart Streamlit."
        )

    if is_running:
        st.sidebar.warning("🔒 A simulation is running — controls are locked until it finishes.")

    st.sidebar.subheader("Model selection")
    selection_mode = st.sidebar.radio(
        "Mode",
        options=["Random per run", "Fixed"],
        index=0,
        help=(
            "Random per run (default): each time you click a Run button, a "
            "model is drawn at random from a diverse pool spanning several "
            "providers. Fixed: always use the model chosen below."
        ),
        label_visibility="collapsed",
        disabled=is_running,
    )
    random_mode = selection_mode == "Random per run"

    if random_mode:
        # Deliberately no dropdown and no list of candidate engines here —
        # the whole point of "Random per run" is that the user doesn't know
        # (and isn't shown) which engine will answer until after the run.
        model: str | None = None
        st.sidebar.caption("An engine will be picked at random when you click Run.")
    else:
        # Fixed mode only offers the verified-stable pool — no
        # "openrouter/auto" (a meta-router, not a specific engine to pick)
        # and no experimental/unproven entries from the dynamic pool.
        model = st.sidebar.selectbox(
            "OpenRouter model",
            options=VERIFIED_STABLE_POOL,
            index=VERIFIED_STABLE_POOL.index(DEFAULT_MODEL) if DEFAULT_MODEL in VERIFIED_STABLE_POOL else 0,
            disabled=is_running,
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Single-Agent = 1 monolithic call.\n\n"
        "Multi-Agent = 7 calls (4 advocates + 3 independent judges)."
    )

    return api_key, model, random_mode


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def render_run_tab(api_key: str | None, model: str | None, random_mode: bool) -> None:
    is_running = st.session_state.get("is_running", False)

    # Reserve the Case Facts slot at the top of the page now, but fill it in
    # further down — after a button click (if any) has updated
    # st.session_state["last_result"] — so a fresh run's arguments show up
    # immediately instead of one rerun later.
    case_facts_slot = st.container()

    st.markdown("#### Run a Simulation")
    col1, col2 = st.columns(2)
    run_single_clicked = col1.button(
        "⚖️ Run Single-Agent Simulation", width="stretch", disabled=is_running
    )
    run_multi_clicked = col2.button(
        "🏛️ Run Multi-Agent Simulation", width="stretch", disabled=is_running
    )

    if is_running:
        st.info("🔒 A simulation is currently running. Please wait for it to finish.")
    elif (run_single_clicked or run_multi_clicked) and not api_key:
        st.error("Cannot run: OPENROUTER_API_KEY is not set. See the sidebar for setup instructions.")
    elif run_single_clicked or run_multi_clicked:
        # Engage the execution lock and stash which run to perform, then
        # rerun immediately so the browser sees the controls above as
        # disabled *before* the (blocking) run actually starts — otherwise
        # a second click during the run could interrupt it mid-flight.
        st.session_state["pending_run"] = "single_agent" if run_single_clicked else "multi_agent"
        st.session_state["pending_model"] = pick_random_model() if random_mode else model
        st.session_state["is_running"] = True
        st.rerun()

    pending_run = st.session_state.get("pending_run")
    if is_running and pending_run:
        active_model = st.session_state.get("pending_model")
        try:
            if pending_run == "single_agent":
                with st.spinner("Running single-agent tribunal simulation..."):
                    st.session_state["last_result"] = run_single_agent(api_key, active_model)
            else:
                with st.spinner("Running multi-agent tribunal simulation..."):
                    st.session_state["last_result"] = run_multi_agent(api_key, active_model)
            st.session_state["run_error"] = None
        except Exception as exc:  # noqa: BLE001
            arch_label = "Single-agent" if pending_run == "single_agent" else "Multi-agent"
            # Stash the error rather than calling st.error() here directly —
            # the st.rerun() below would wipe it from the screen before the
            # user ever saw it, since this whole branch's render is discarded.
            st.session_state["run_error"] = f"{arch_label} run failed: {exc}"
        finally:
            st.session_state["is_running"] = False
            st.session_state["pending_run"] = None
            st.session_state["pending_model"] = None
            st.rerun()

    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])
        st.session_state["run_error"] = None

    st.divider()

    last_result = st.session_state.get("last_result")

    with case_facts_slot:
        with st.expander("📜 Canonical Case Facts", expanded=False):
            st.markdown(f"**Case ID:** {case_data.CASE_ID}")
            st.markdown(f"**Accused:** {case_data.ACCUSED}")
            st.markdown(f"**Deceased:** {case_data.DECEASED}")
            st.markdown(f"**Alleged act:** {case_data.ALLEGED_ACT}")
            st.markdown("**Agreed facts:**")
            for i, fact in enumerate(case_data.AGREED_FACTS, start=1):
                st.markdown(f"{i}. {fact}")
            st.markdown(f"**Tribunal issue:** {case_data.TRIBUNAL_ISSUE}")
            st.caption(case_data.TRIBUNAL_SCOPE)

            st.divider()
            st.markdown("**Prosecution Arguments (case theory):**")
            for point in case_data.PROSECUTION_ARGUMENTS:
                st.markdown(f"- {point}")
            st.markdown("**Defense Arguments (case theory):**")
            for point in case_data.DEFENSE_ARGUMENTS:
                st.markdown(f"- {point}")

            if last_result:
                st.divider()
                st.markdown('<span class="trib-eyebrow">Arguments of Record — this run</span>', unsafe_allow_html=True)
                col_p, col_d = st.columns(2)
                with col_p:
                    st.markdown(
                        _advocate_card_html("Prosecution", "prosecution", last_result["prosecution_summary"]),
                        unsafe_allow_html=True,
                    )
                with col_d:
                    st.markdown(
                        _advocate_card_html("Defense", "defense", last_result["defense_summary"]),
                        unsafe_allow_html=True,
                    )

    if last_result:
        render_result(last_result)
    else:
        st.caption("No run yet in this session. Click a button above to start.")


def render_history_tab() -> None:
    if st.session_state.get("is_running", False):
        st.warning(
            "🔒 A simulation is currently running. Historical runs are locked "
            "until it finishes — this avoids interrupting the active run."
        )
        return

    db_path = init_db()

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM trial_runs ORDER BY id DESC", conn)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        st.info("No runs logged yet. Run a simulation from the first tab to populate history.")
        return

    # (a) Historical Runs Table — overview dataframe, collapsed by default.
    st.markdown("#### Historical Runs")
    with st.expander("Historical Runs Overview", expanded=False):
        summary_cols = [
            "id",
            "timestamp",
            "model",
            "architecture_mode",
            "judge_barak_verdict",
            "judge_elon_verdict",
            "judge_shamgar_verdict",
            "total_tokens",
            "cost_usd",
            "cost_ils",
            "execution_time_sec",
        ]
        st.dataframe(df[summary_cols], width="stretch", hide_index=True)

    # (b) Run Selector & Action Trigger — selecting a dropdown option alone
    # does NOT change what's displayed below; only clicking "Load Selected
    # Run" commits the selection and (re)populates the detail sections.
    st.markdown("#### Inspect a Past Deliberation")
    options = [
        f"Run #{row.id} | {row.timestamp} | {row.model or 'unknown'} | {row.architecture_mode}"
        for row in df.itertuples()
    ]
    col_select, col_load = st.columns([4, 1])
    with col_select:
        selected = st.selectbox("Select a run", options=options, label_visibility="collapsed")
    with col_load:
        load_clicked = st.button("Load Selected Run", width="stretch")

    if load_clicked:
        selected_id = int(selected.split("#")[1].split(" ")[0])
        st.session_state["hist_loaded_run_id"] = selected_id
        # A newly loaded run's detail section always starts fully collapsed
        # (neither "arguments" nor "deliberation" active), regardless of
        # whatever was open for a previously loaded run.
        st.session_state["hist_active_detail"] = None

    loaded_run_id = st.session_state.get("hist_loaded_run_id")
    if loaded_run_id is None or loaded_run_id not in df["id"].values:
        st.caption("Select a run above, then click “Load Selected Run” to inspect it.")
        return

    row = df[df["id"] == loaded_run_id].iloc[0]
    judges = {
        "barak": {"reasoning": row["judge_barak_reasoning"], "verdict": row["judge_barak_verdict"]},
        "elon": {"reasoning": row["judge_elon_reasoning"], "verdict": row["judge_elon_verdict"]},
        "shamgar": {"reasoning": row["judge_shamgar_reasoning"], "verdict": row["judge_shamgar_verdict"]},
    }

    # (c) Final Tribunal Outcome banner/card.
    render_majority_outcome(judges)
    st.markdown(f"**Engine:** `{row['model'] or 'unknown'}`")

    # (d)+(e) Arguments of Record / Judicial Deliberation — a single
    # "which one is active" state (None | "arguments" | "deliberation"),
    # not two synced booleans, so there's no dual-widget race to glitch:
    # clicking a header sets this variable directly, which by construction
    # makes the sibling inactive in that same click.
    active_detail = st.session_state.get("hist_active_detail")

    col_args, col_delib = st.columns(2)
    with col_args:
        args_active = active_detail == "arguments"
        if st.button(
            f"{'▾' if args_active else '▸'} Show Arguments of Record",
            key="hist_toggle_arguments",
            width="stretch",
        ):
            st.session_state["hist_active_detail"] = None if args_active else "arguments"
            st.rerun()
    with col_delib:
        delib_active = active_detail == "deliberation"
        if st.button(
            f"{'▾' if delib_active else '▸'} Show Judicial Deliberation",
            key="hist_toggle_deliberation",
            width="stretch",
        ):
            st.session_state["hist_active_detail"] = None if delib_active else "deliberation"
            st.rerun()

    active_detail = st.session_state.get("hist_active_detail")
    if active_detail == "arguments":
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
    elif active_detail == "deliberation":
        render_judges(judges)

    # (f) Budget & Token Telemetry Summary.
    st.subheader("Budget Summary")
    rate = get_ils_exchange_rate()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Execution time", f"{row['execution_time_sec']:.2f} sec")
    m2.metric("Total tokens", f"{int(row['total_tokens']):,}")
    m3.metric("Estimated cost (USD)", f"${row['cost_usd']:.6f}")
    m4.metric("Estimated cost (ILS)", f"₪{row['cost_ils']:.6f}")
    st.caption(f"Prompt tokens: {int(row['prompt_tokens']):,} • Completion tokens: {int(row['completion_tokens']):,} • ILS rate: {rate:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("⚖️ Tribunal")
    st.caption(case_data.CASE_ID)

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "pending_run" not in st.session_state:
        st.session_state["pending_run"] = None
    if "pending_model" not in st.session_state:
        st.session_state["pending_model"] = None
    if "run_error" not in st.session_state:
        st.session_state["run_error"] = None
    # Top-level accordion: Run Simulation open, Historical Runs closed, on
    # first load. Mutually exclusive thereafter (see _toggle_top_section).
    if "section_run_open" not in st.session_state:
        st.session_state["section_run_open"] = True
    if "section_history_open" not in st.session_state:
        st.session_state["section_history_open"] = False
    # Historical Runs' own state: which run is actually displayed (only
    # changes when "Load Selected Run" is clicked — see render_history_tab),
    # and which ONE detail section is active: None | "arguments" |
    # "deliberation" — always None (both closed) right after a fresh load.
    if "hist_loaded_run_id" not in st.session_state:
        st.session_state["hist_loaded_run_id"] = None
    if "hist_active_detail" not in st.session_state:
        st.session_state["hist_active_detail"] = None

    api_key, model, random_mode = render_sidebar()

    is_running = st.session_state.get("is_running", False)

    if is_running:
        # Historical Runs isn't rendered at all while a simulation is
        # active — render_history_tab() also keeps its own is_running
        # guard as a defense-in-depth backstop — so there is nothing to
        # accordion-toggle away from during a run; only the Run Simulation
        # section exists on screen.
        st.markdown("#### 🏟️ Run Simulation")
        render_run_tab(api_key, model, random_mode)
        return

    # Accordion pattern: two mutually-exclusive collapsible sections,
    # replacing st.tabs. Implemented as a toggle button + a plain
    # conditional block (not st.expander) because the nested "Arguments of
    # Record" / "Judicial Deliberation" sections inside Historical Runs
    # must themselves be real st.expander widgets, and Streamlit forbids
    # nesting an expander inside another expander.
    run_open = st.session_state.get("section_run_open", True)
    if st.button(
        f"{'▾' if run_open else '▸'} 🏟️ Run Simulation",
        key="section_run_toggle",
        width="stretch",
    ):
        _toggle_top_section("section_run_open", "section_history_open")
        st.rerun()
    if st.session_state.get("section_run_open", True):
        render_run_tab(api_key, model, random_mode)

    st.divider()

    history_open = st.session_state.get("section_history_open", False)
    if st.button(
        f"{'▾' if history_open else '▸'} 📚 Historical Runs",
        key="section_history_toggle",
        width="stretch",
    ):
        _toggle_top_section("section_history_open", "section_run_open")
        st.rerun()
    if st.session_state.get("section_history_open", False):
        render_history_tab()


if __name__ == "__main__":
    main()
