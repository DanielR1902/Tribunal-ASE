"""Streamlit GUI for the Westeros Tribunal simulation.

Lets a user trigger a Single-Agent or Multi-Agent run from the browser,
watch the deliberation render live, and browse past runs logged in
court_runs.db — instead of driving everything from the command line.

Run with:  streamlit run app.py
"""

from __future__ import annotations

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
    MODEL_POOL,
    USAGE_ACCOUNTING_EXTRA_BODY,
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
# Client / execution helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_client(api_key: str) -> OpenAI:
    return get_client(api_key)


def run_single_agent(api_key: str, model: str) -> dict:
    client = get_cached_client(api_key)
    tracker = CostTracker(model=model)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_single_prompt()}],
        response_format={"type": "json_object"},
        temperature=0.7,
        extra_body=USAGE_ACCOUNTING_EXTRA_BODY,
    )
    execution_time = time.perf_counter() - start

    verdict = TribunalVerdict.model_validate_json(response.choices[0].message.content or "")

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
# Rendering helpers
# ---------------------------------------------------------------------------

def verdict_badge(verdict: str) -> None:
    if verdict == "Justified":
        st.success(f"✅ Verdict: {verdict}")
    else:
        st.error(f"⛔ Verdict: {verdict}")


def render_judges(judges: dict) -> None:
    cols = st.columns(3)
    for col, key in zip(cols, JUDGE_ORDER):
        persona = JUDGE_PERSONAS[key]
        opinion = judges[key]
        with col:
            with st.container(border=True):
                st.markdown(f"**{persona['name']}**")
                st.caption(persona["model_label"])
                verdict_badge(opinion["verdict"])
                with st.expander("Read full reasoning"):
                    st.write(opinion["reasoning"])


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


def render_budget_box(
    tracker: CostTracker, execution_time: float, run_id: int, requested_model: str
) -> None:
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

    st.caption(f"Requested model: `{requested_model}`  •  ILS rate used: {rate:.2f}  •  Database: `{get_db_path()}`")

    if len(tracker.calls) > 1:
        with st.expander("Per-call token breakdown"):
            breakdown_df = pd.DataFrame(
                [
                    {
                        "Agent call": c.label,
                        "Model executed": c.executed_model,
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

    executed = result["executed_model"]
    requested = result["requested_model"]
    if executed == requested:
        st.info(f"🧭 **Model used for this trial:** `{executed}`")
    else:
        st.info(
            f"🧭 **Model used for this trial:** `{executed}`  \n"
            f"(requested as `{requested}`, resolved by OpenRouter)"
        )

    st.subheader("Judicial Deliberation")
    render_judges(result["judges"])

    render_majority_outcome(result["judges"])

    render_budget_box(result["tracker"], result["execution_time"], result["run_id"], result["requested_model"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str | None, str | None, bool]:
    st.sidebar.title("⚖️ Tribunal Controls")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.sidebar.error("OPENROUTER_API_KEY not found")
        st.sidebar.caption(
            "Copy `.env.example` to `.env` in the project root and add your "
            "key from https://openrouter.ai/keys, then restart Streamlit."
        )

    st.sidebar.subheader("Model selection")
    selection_mode = st.sidebar.radio(
        "Mode",
        options=["Random per run", "Fixed"],
        index=0,
        help=(
            "Random per run (default): each time you click a Run button, a "
            "model is drawn at random from a diverse pool spanning several "
            "providers — including OpenRouter's own openrouter/auto "
            "meta-router, which dynamically picks a concrete model per "
            "request server-side. Fixed: always use the model chosen below."
        ),
        label_visibility="collapsed",
    )
    random_mode = selection_mode == "Random per run"

    if random_mode:
        model: str | None = None
        st.sidebar.caption("A model will be picked at random from the pool below when you click Run.")
        with st.sidebar.expander("Model pool"):
            for m in MODEL_POOL:
                st.caption(f"• `{m}`")
    else:
        model = st.sidebar.selectbox(
            "OpenRouter model",
            options=MODEL_POOL,
            index=MODEL_POOL.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODEL_POOL else 0,
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
    # Reserve the Case Facts slot at the top of the page now, but fill it in
    # further down — after a button click (if any) has updated
    # st.session_state["last_result"] — so a fresh run's arguments show up
    # immediately instead of one rerun later.
    case_facts_slot = st.container()

    st.markdown("#### Run a Simulation")
    col1, col2 = st.columns(2)
    run_single_clicked = col1.button("⚖️ Run Single-Agent Simulation", width="stretch")
    run_multi_clicked = col2.button("🏛️ Run Multi-Agent Simulation", width="stretch")

    if (run_single_clicked or run_multi_clicked) and not api_key:
        st.error("Cannot run: OPENROUTER_API_KEY is not set. See the sidebar for setup instructions.")
    elif run_single_clicked:
        active_model = pick_random_model() if random_mode else model
        if random_mode:
            st.caption(f"🎲 Randomly selected model for this trial: `{active_model}`")
        with st.spinner(f"Running single-agent tribunal (1 OpenRouter call, model: {active_model})..."):
            try:
                st.session_state["last_result"] = run_single_agent(api_key, active_model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Single-agent run failed: {exc}")
    elif run_multi_clicked:
        active_model = pick_random_model() if random_mode else model
        if random_mode:
            st.caption(f"🎲 Randomly selected model for this trial: `{active_model}`")
        with st.spinner(
            f"Running multi-agent tribunal (7 OpenRouter calls: 4 advocates + 3 judges, model: {active_model})..."
        ):
            try:
                st.session_state["last_result"] = run_multi_agent(api_key, active_model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Multi-agent run failed: {exc}")

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

            if last_result:
                st.divider()
                st.markdown("**Prosecution Arguments**")
                st.info(last_result["prosecution_summary"])
                st.markdown("**Defense Arguments**")
                st.info(last_result["defense_summary"])

    if last_result:
        render_result(last_result)
    else:
        st.caption("No run yet in this session. Click a button above to start.")


def render_history_tab() -> None:
    db_path = init_db()
    conn_str = db_path

    import sqlite3

    with sqlite3.connect(conn_str) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM trial_runs ORDER BY id DESC", conn)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        st.info("No runs logged yet. Run a simulation from the first tab to populate history.")
        return

    with st.expander("📜 Past Runs", expanded=False):
        summary_cols = [
            "id",
            "timestamp",
            "architecture_mode",
            "model",
            "judge_barak_verdict",
            "judge_elon_verdict",
            "judge_shamgar_verdict",
            "total_tokens",
            "cost_usd",
            "cost_ils",
            "execution_time_sec",
        ]
        st.dataframe(df[summary_cols], width="stretch", hide_index=True)

    st.markdown("#### Inspect a Past Deliberation")
    options = [
        f"Run #{row.id} — {row.architecture_mode} — {getattr(row, 'model', '') or 'unknown model'} — {row.timestamp}"
        for row in df.itertuples()
    ]
    selected = st.selectbox("Select a run", options=options)
    selected_id = int(selected.split("#")[1].split(" ")[0])
    row = df[df["id"] == selected_id].iloc[0]

    st.info(f"🧭 **Model executed for this trial:** `{row['model'] or 'unknown (logged before model tracking was added)'}`")

    st.subheader("Prosecution Arguments")
    st.info(row["prosecution_summary"])

    st.subheader("Defense Arguments")
    st.info(row["defense_summary"])

    st.subheader("Judicial Deliberation")
    judges = {
        "barak": {"reasoning": row["judge_barak_reasoning"], "verdict": row["judge_barak_verdict"]},
        "elon": {"reasoning": row["judge_elon_reasoning"], "verdict": row["judge_elon_verdict"]},
        "shamgar": {"reasoning": row["judge_shamgar_reasoning"], "verdict": row["judge_shamgar_verdict"]},
    }
    render_judges(judges)
    render_majority_outcome(judges)

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

    api_key, model, random_mode = render_sidebar()

    tab_run, tab_history = st.tabs(["🏟️ Run Simulation", "📚 Historical Runs"])
    with tab_run:
        render_run_tab(api_key, model, random_mode)
    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    main()
