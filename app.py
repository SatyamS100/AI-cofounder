"""
app.py
======
Streamlit UI for AI Venture Studio. Drives the compiled LangGraph pipeline
from src/graph.py and renders live progress as each agent completes.

THE STREAMLIT <-> LANGGRAPH INTEGRATION PROBLEM (read this before the code):
Streamlit's execution model reruns the ENTIRE script top-to-bottom on every
user interaction. LangGraph's `.stream()` is a generator that yields
incremental state updates as nodes finish, designed to be consumed
continuously by something that's watching it live. These two models don't
naturally fit together.

THE RESOLUTION:
When the user clicks "Analyze," we call `venture_graph.stream(...)` and
iterate it INSIDE THE SAME SCRIPT EXECUTION — there's no rerun between
agent completions. Streamlit supports updating already-rendered widgets
(via st.status().update(), st.empty() containers, etc.) progressively
within a single script run; the "rerun on interaction" model governs when a
NEW run starts, not whether you can update the UI incrementally inside one.
This is what makes the 4 progress indicators update live as agents finish,
without fighting Streamlit's architecture or reaching for websockets/
threading that this single-user demo app doesn't need.

WHY KEYED BY NODE NAME, NOT ASSUMED ORDER:
`.stream()` yields results in COMPLETION order, not the order nodes appear
in the graph. Within Stage 1, market_intelligence might finish before or
after problem_validator depending on web_search latency and cache hits. The
progress-rendering code below keys every update by node name so each card
updates independently of when its sibling nodes finish.
"""

import json
from dotenv import load_dotenv
load_dotenv()   # must happen before any import that initializes the Groq/Tavily clients

import streamlit as st

from src.graph import venture_graph
from src.rag import seed_accelerator_knowledge
from config import STAGE_LABELS, AGENT_METADATA

st.set_page_config(page_title="AI Venture Studio", page_icon="🚀", layout="wide")

# Seed the static accelerator-knowledge RAG collection once per process.
# seed_accelerator_knowledge() is idempotent (skips re-seeding if already
# populated — see rag.py), so calling it on every script rerun is safe and
# cheap, not just safe-but-wasteful.
seed_accelerator_knowledge()


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────
# st.session_state is Streamlit's mechanism for data that survives across
# reruns within one user's browser session. We need this for the final
# result to persist even after the script reruns (e.g. when the user
# expands/collapses a card after the analysis is already done).
if "final_state" not in st.session_state:
    st.session_state.final_state = None


# ─────────────────────────────────────────────────────────────────────────
# HEADER + INPUT
# ─────────────────────────────────────────────────────────────────────────
st.title("🚀 AI Venture Studio")
st.caption(
    "Eight specialized Claude agents evaluate your startup pitch in two "
    "parallel stages, then a VC agent delivers the final verdict."
)

raw_pitch = st.text_area(
    "Describe your startup idea",
    height=120,
    placeholder=(
        "e.g. A marketplace app that lets college hostel residents order "
        "home-style meals from verified local tiffin vendors, with group "
        "ordering to split delivery costs..."
    ),
)

analyze_clicked = st.button("Analyze", type="primary", disabled=not raw_pitch.strip())


# ─────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION + LIVE PROGRESS
# ─────────────────────────────────────────────────────────────────────────

def _node_output_failed(node_output: dict) -> bool:
    """
    Returns True if the substantive result inside a node's output dict is a
    make_failed_result(...) marker (see src/state.py). Used to decide
    whether to show the "partial data" warning banner after the run.
    """
    result_key = next((k for k in node_output if k != "stage"), None)
    if result_key is None:
        return False
    result = node_output[result_key]
    return isinstance(result, dict) and result.get("status") == "failed"


def render_agent_card(node_name: str, node_output: dict):
    """
    Renders one agent's result inside an st.expander. Handles 3 distinct
    cases differently so the UI never silently hides what actually happened:
      1. Normal successful result -> pretty-printed JSON
      2. Failed-after-retry result -> explicit warning, not hidden as if
         it were just empty data (see state.py's make_failed_result)
      3. market_intelligence cache hit -> explicit "reused" badge showing
         WHERE the data came from and HOW OLD it is, per the no-silent-
         reuse requirement from the project design
    """
    display_name = AGENT_METADATA.get(node_name, {}).get("display_name", node_name)

    # node_output is the dict LangGraph returned for this node, e.g.
    # {"market_analysis": {...}, "stage": "..."}. Pull out the actual
    # agent-result dict (the one substantive key besides "stage").
    result_key = next((k for k in node_output if k != "stage"), None)
    result = node_output.get(result_key) if result_key else None

    with st.container(border=True):
        st.markdown(f"#### 📌 {display_name}")
        if result is None:
            st.info("No data returned for this stage.")
            return

        # Case: a list of results (only user_personas)
        if isinstance(result, list):
            for persona in result:
                if persona.get("status") == "failed":
                    st.warning(
                        f"⚠️ {persona.get('agent', 'This persona')} failed after "
                        f"retry: {persona.get('reason', 'unknown error')}"
                    )
                else:
                    st.json(persona)
            return

        # Case: failed-after-retry
        if isinstance(result, dict) and result.get("status") == "failed":
            st.warning(
                f"⚠️ This analysis failed after one retry attempt and was "
                f"skipped: {result.get('reason', 'unknown error')}. "
                f"The VC verdict was generated without this input."
            )
            return

        # Case: market_intelligence cache hit — explicit attribution,
        # never silently presented as fresh data
        if isinstance(result, dict) and result.get("data_source") == "cached":
            st.success(
                f"⚡ Reused market analysis from a similar pitch analyzed "
                f"**{result.get('cached_age_days')} days ago** "
                f"(matched pitch: _{result.get('cached_from_pitch', '')[:120]}..._). "
                f"No new web search was needed."
            )
            st.json(result)
            return

        if isinstance(result, dict) and result.get("data_source") == "live_web_search":
            st.caption("🔍 Fresh web search — no similar prior analysis found.")
            st.json(result)
            return

        # Default: plain successful result
        st.json(result)


if analyze_clicked:
    st.session_state.final_state = None  # clear any previous run's result

    initial_state = {
        "raw_pitch": raw_pitch,
        "refined_pitch": {},
        "market_analysis": {},
        "problem_validation": {},
        "technical_feasibility": {},
        "accelerator_fit": {},
        "user_personas": [],
        "devils_advocate": {},
        "vc_verdict": {},
        "stage": "starting",
        "errors": [],
    }

    st.divider()
    st.subheader("Progress")

    # One st.status container per pipeline stage, matching the spec's
    # 4-indicator layout. Each starts in "running" state and is flipped to
    # "complete" once every node belonging to that stage has reported in.
    stage_containers = {
        0: st.status(f"⏳ {STAGE_LABELS[0]}", expanded=True),
        1: st.status(f"⏳ {STAGE_LABELS[1]}", expanded=True),
        2: st.status(f"⏳ {STAGE_LABELS[2]}", expanded=True),
        3: st.status(f"⏳ {STAGE_LABELS[3]}", expanded=True),
    }

    # Track which nodes have reported in per stage, so we know when to
    # flip a stage's status from "running" to "complete." Stage 1 needs
    # all 3 of its nodes; Stage 2 needs all 3 of its nodes; Stages 0 and 3
    # need just their single node.
    stage_node_counts = {0: 1, 1: 3, 2: 3, 3: 1}
    stage_nodes_done = {0: 0, 1: 0, 2: 0, 3: 0}

    accumulated_state = dict(initial_state)
    run_had_failures = False

    # THE CORE LOOP: this is where LangGraph's generator and Streamlit's
    # single-script-execution UI updates meet. Each iteration is one node
    # completing — we update its stage's status container immediately,
    # without waiting for a Streamlit rerun.
    for step in venture_graph.stream(initial_state):
        # step is a dict like {"market_intelligence": {<node's return dict>}}
        # — LangGraph's .stream() yields one such dict per node completion.
        for node_name, node_output in step.items():
            accumulated_state.update(node_output)

            agent_meta = AGENT_METADATA.get(node_name, {})
            stage_num = agent_meta.get("stage", 0)

            with stage_containers[stage_num]:
                render_agent_card(node_name, node_output)

            stage_nodes_done[stage_num] += 1
            if stage_nodes_done[stage_num] >= stage_node_counts[stage_num]:
                stage_containers[stage_num].update(
                    label=f"✓ {STAGE_LABELS[stage_num]}",
                    state="complete",
                    expanded=False,
                )

            if _node_output_failed(node_output):
                run_had_failures = True

    st.session_state.final_state = accumulated_state
    if run_had_failures or accumulated_state.get("errors"):
        st.warning(
            "⚠️ One or more agents failed after a retry attempt. The VC "
            "verdict below was generated from partial data — see the "
            "individual agent cards above for details."
        )


# ─────────────────────────────────────────────────────────────────────────
# FINAL VC VERDICT
# ─────────────────────────────────────────────────────────────────────────

VERDICT_COLORS = {
    "Pass": "🔴",
    "Watch": "🟡",
    "Invest": "🟢",
    "Lead": "🔵",
}

if st.session_state.final_state and st.session_state.final_state.get("vc_verdict"):
    verdict = st.session_state.final_state["vc_verdict"]

    st.divider()
    st.subheader("🏦 VC Mentor Verdict")

    if verdict.get("status") == "failed":
        st.error(
            f"⚠️ The VC synthesis agent failed after retry: "
            f"{verdict.get('reason', 'unknown error')}. No final verdict "
            f"could be generated — see the individual agent results above."
        )
    else:
        # ── VERDICT BADGE + SCORE ────────────────────────────────────────
        VERDICT_COLORS = {
            "Pass":   ("🔴", "red"),
            "Watch":  ("🟡", "orange"),
            "Invest": ("🟢", "green"),
            "Lead":   ("🔵", "blue"),
        }
        v = verdict.get("investment_verdict", "Unknown")
        badge, color = VERDICT_COLORS.get(v, ("⚪", "gray"))

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Signal Score",
                      f"{verdict.get('overall_signal_score_1_to_10', '?')}/10")
        with col2:
            st.metric("Investor Readiness",
                      f"{verdict.get('current_state_assessment', {}).get('investor_readiness_score', '?')}/10")
        with col3:
            st.markdown(f"## {badge} {v}")
            st.caption(verdict.get("verdict_reasoning", ""))

        # ── CURRENT STATE ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Where You Stand Right Now")
        cs = verdict.get("current_state_assessment", {})
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Biggest Strength:** {cs.get('biggest_current_strength', '—')}")
            st.info(f"**Market Opportunity:** {cs.get('market_opportunity', '—')}")
            st.info(f"**Problem Validation:** {cs.get('problem_validation_strength', '—')}")
        with c2:
            st.error(f"**Biggest Gap:** {cs.get('biggest_current_gap', '—')}")
            st.info(f"**Technical Feasibility:** {cs.get('technical_feasibility', '—')}")

        # ── BULL / BEAR ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚖️ Bull vs. Bear")
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"🐂 **Bull Case**\n\n{verdict.get('key_bull_case', '—')}")
        with c2:
            st.error(f"🐻 **Bear Case**\n\n{verdict.get('key_bear_case', '—')}")

        # ── ROADMAP TO FUNDABLE ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🗺️ Roadmap to Fundable")
        st.info(f"**What changes the verdict:** {verdict.get('roadmap_to_fundable', {}).get('what_changes_the_verdict', '—')}")
        steps = verdict.get("roadmap_to_fundable", {}).get("ordered_steps", [])
        for step in steps:
            with st.expander(f"Step {step.get('step')} — {step.get('action', '')}"):
                st.markdown(f"**Success metric:** {step.get('success_metric', '—')}")
                st.markdown(f"**Timeline:** {step.get('timeline', '—')}")

        # ── TEAM STRUCTURE ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 👥 Team Structure Recommendation")
        team = verdict.get("team_structure_recommendation", {})
        if team.get("cofounder_gaps"):
            st.warning(f"**Cofounder gaps:** {team.get('cofounder_gaps')}")
        hires = team.get("critical_first_hires", [])
        if hires:
            st.markdown("**Critical first hires:**")
            cols = st.columns(min(len(hires), 3))
            for i, hire in enumerate(hires):
                with cols[i % 3]:
                    st.markdown(f"**{hire.get('role', '')}**")
                    st.caption(f"Why critical: {hire.get('why_critical', '')}")
                    st.caption(f"Look for: {hire.get('what_to_look_for', '')}")
        if team.get("outsource_vs_hire"):
            st.info(f"**Outsource vs hire:** {team.get('outsource_vs_hire')}")

        # ── COMPARABLE COMPANIES ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏢 Comparable Companies & What to Learn")
        comps = verdict.get("comparable_companies", {})

        if comps.get("competitor_opportunity"):
            st.success(f"**Biggest near-term competitor opening:** {comps.get('competitor_opportunity')}")

        parallels = comps.get("successful_parallels", [])
        cautionary = comps.get("cautionary_tales", [])

        if parallels:
            st.markdown("**✅ Successful companies to learn from:**")
            for p in parallels:
                with st.expander(f"{p.get('company')} — {p.get('accelerator_or_investor', '')}"):
                    st.markdown(f"**What to copy:** {p.get('what_to_copy', '—')}")

        if cautionary:
            st.markdown("**⚠️ Cautionary tales:**")
            for c in cautionary:
                with st.expander(f"{c.get('company')} — what went wrong"):
                    st.markdown(f"**What killed them:** {c.get('what_killed_them', '—')}")
                    st.markdown(f"**How to avoid it:** {c.get('how_to_avoid', '—')}")

        # ── CRITICAL UNKNOWN + NEXT QUESTION ────────────────────────────
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.warning(f"**❓ Critical unknown:** {verdict.get('critical_unknown', '—')}")
        with c2:
            st.info(f"**💬 Question for next meeting:** {verdict.get('one_question_for_next_meeting', '—')}")

        gaps = verdict.get("data_gaps_acknowledged", [])
        if gaps:
            st.caption(f"⚠️ Data gaps acknowledged: {', '.join(gaps)}")

    # Download full report
    st.download_button(
        label="⬇️ Download full report as JSON",
        data=json.dumps(st.session_state.final_state, indent=2),
        file_name="venture_analysis_report.json",
        mime="application/json",
    )