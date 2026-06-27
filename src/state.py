"""
state.py
========
Defines the shared state object that flows through every node in the
LangGraph StateGraph.

DESIGN DECISION — why a TypedDict:
LangGraph's StateGraph expects a schema it can use to merge partial state
updates returned by each node. TypedDict is the canonical choice because:
  1. It's a plain dict at runtime (zero serialization overhead between nodes).
  2. LangGraph's default "merge" behavior is: each node returns a dict with
     ONLY the keys it touched, and LangGraph shallow-merges that into the
     global state. TypedDict documents exactly which keys are legal.
  3. For fields written by exactly ONE node, no special handling is needed.
  4. For fields that MULTIPLE parallel nodes might write to (like `errors`),
     we need a custom reducer — see the Annotated[...] field below — otherwise
     the second parallel node to finish would silently overwrite the first
     node's writes instead of combining them.

INTERVIEW TALKING POINT:
"Each of my 8 agents owns a unique key in this schema. That's a deliberate
single-writer design — during the two parallel stages, three agents run
concurrently, and because no two agents ever write the same key, there's no
race condition or merge conflict to reason about. The one exception is
`errors`, which uses a reducer function so that if multiple parallel agents
fail simultaneously, their error messages get appended together instead of
one overwriting the other."
"""

from typing import TypedDict, Annotated
import operator


# ─────────────────────────────────────────────────────────────────────────
# RETRY / FAILURE POLICY
# ─────────────────────────────────────────────────────────────────────────
# Every agent in agents.py follows the same resilience contract:
#   1. Call the Claude API.
#   2. If it raises (network error, rate limit, malformed JSON, etc.) —
#      pause briefly and retry EXACTLY ONCE.
#   3. If the retry also fails, do NOT crash the graph. Instead, write back
#      a small "failed" marker dict using the shape below, and log a
#      human-readable message into the shared `errors` list.
#
# WHY THIS MATTERS FOR A MULTI-AGENT SYSTEM SPECIFICALLY:
# A single LLM call failing (rate limit, transient network blip, occasional
# malformed JSON) is common and expected at this scale. If one bad call
# crashed the whole pipeline, a user who waited 25 seconds for 7 successful
# agents would lose everything because the 8th agent hiccuped. Instead, the
# graph keeps moving, and the final VC synthesis agent is explicitly told
# which inputs are missing so it can reason around the gap rather than
# pretending it has full information.
#
# This dict is what a failed agent writes into ITS OWN state key (e.g.
# state["market_analysis"] = make_failed_result("market_intelligence", "...")
# It deliberately mirrors a normal result's shape (a dict with
# confidence_score-like fields set to None) so downstream code and the
# Streamlit UI can handle success/failure with the same code path instead
# of type-checking everywhere.
def make_failed_result(agent_name: str, reason: str) -> dict:
    """Standard shape written by an agent that failed after one retry."""
    return {
        "status": "failed",
        "agent": agent_name,
        "reason": reason,
        "confidence_score": None,
    }


class PersonaReaction(TypedDict):
    """
    Shared shape for the three user-persona agents (early adopter, mainstream,
    skeptical enterprise buyer). They don't return identical fields (the
    enterprise persona has 'would_approve_budget' instead of 'would_use'), so
    in practice we store these as plain dicts in a list — this TypedDict is
    here mainly as documentation of the common fields, and for type hints
    where it's useful.
    """
    persona_name: str
    would_use: bool
    monthly_price_willing_to_pay_usd: float
    biggest_objection: str
    excitement_score: int  # 1-10


def _overwrite(left: str, right: str) -> str:
    return right

class VentureState(TypedDict):
    # ---- Input ----
    raw_pitch: str  # The user's original, unedited startup idea

    # ---- Stage 0: Sequential refinement ----
    refined_pitch: dict  # Output of pitch_refiner agent (full JSON, not just text)

    # ---- Stage 1: Parallel — independent analyses ----
    # These three agents have NO dependency on each other's output, which is
    # exactly why they're safe to run in parallel. Each only depends on
    # `refined_pitch`.
    market_analysis: dict
    problem_validation: dict
    technical_feasibility: dict

    # ---- Stage 2: Parallel — downstream analyses ----
    # These depend on Stage 1 having completed (they reason about market +
    # problem + feasibility together), but are independent of EACH OTHER,
    # so they also run in parallel.
    accelerator_fit: dict
    user_personas: list[dict]      # exactly 3 persona reaction dicts
    devils_advocate: dict

    # ---- Final synthesis (sequential, runs last) ----
    vc_verdict: dict

    # ---- Orchestration / UI metadata ----
    stage: Annotated[str, _overwrite]  # human-readable current stage, used by Streamlit to update UI

    # `operator.add` reducer: if multiple parallel nodes fail and each
    # appends to `errors`, LangGraph concatenates the lists instead of the
    # last writer winning. Without Annotated[...] here, only the LAST
    # parallel node's error list would survive.
    #
    # NOTE: `errors` is the GLOBAL log used by the Streamlit UI to show a
    # banner like "2 agents failed after retry — verdict generated from
    # partial data." Each individual agent's own state key (e.g.
    # `market_analysis`) ALSO carries failure info via make_failed_result()
    # above. The two serve different purposes: `errors` is for an at-a-glance
    # summary; the per-agent dict is for the VC synthesis agent and the
    # per-card UI to know exactly which specific analysis is missing.
    errors: Annotated[list[str], operator.add]
