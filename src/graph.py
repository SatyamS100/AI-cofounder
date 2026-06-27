"""
graph.py
========
Wires the 8 agent functions from agents.py into a LangGraph StateGraph with
two genuinely parallel execution stages. This file is the project's main
interview talking point — see the module-level comments below for exactly
how LangGraph expresses parallelism (it's topology, not a special operator).

GRAPH SHAPE:
    START
      -> pitch_refiner                                          (sequential)
      -> [market_intelligence, problem_validator,
          technical_feasibility]                                (PARALLEL — Stage 1)
      -> [accelerator_fit, user_personas, devils_advocate]       (PARALLEL — Stage 2)
      -> vc_final_call                                           (sequential)
      -> END

HOW LANGGRAPH ACTUALLY EXECUTES THIS IN PARALLEL:
LangGraph has no explicit "run these in parallel" operator. Parallelism
emerges from graph TOPOLOGY: if multiple nodes share the same predecessor
and there's no edge between them, LangGraph's runtime (a Pregel-style
"superstep" executor) sees they're all immediately ready to fire the moment
their shared predecessor finishes, and runs them concurrently.

The harder half is the JOIN/BARRIER. Per LangGraph's own documentation:
"When multiple start nodes are provided [to add_edge], the graph will wait
for ALL of the start nodes to complete before executing the end node." So
the fan-in for Stage 1 -> Stage 2 is expressed by passing a LIST of
predecessor node names into a single add_edge call — that list IS the
synchronization barrier. This is exactly why state.py's `errors` field
needed an `operator.add` reducer: multiple parallel branches can write to
shared state in the same superstep, and LangGraph needs to know how to
combine those writes rather than having the last one silently win.
"""

from langgraph.graph import StateGraph, START, END

from src.state import VentureState
from src.agents import (
    pitch_refiner,
    market_intelligence,
    problem_validator,
    technical_feasibility,
    accelerator_fit,
    user_personas,
    devils_advocate,
    vc_final_call,
)


def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph. Returns a compiled,
    invocable graph object — call build_graph().invoke(initial_state) or
    .stream(initial_state) to run it (app.py uses .stream() so the Streamlit
    UI can update progress indicators live as each node completes).
    """
    builder = StateGraph(VentureState)

    # ---- Register every node BEFORE adding any edges. This ordering is
    # required, not just stylistic: LangGraph's list-form add_edge (used
    # below for the fan-in barriers) validates that every node name in the
    # list has already been registered via add_node, and raises a
    # ValueError ("Need to add_node `X` first") otherwise. ----
    builder.add_node("pitch_refiner", pitch_refiner)
    builder.add_node("market_intelligence", market_intelligence)
    builder.add_node("problem_validator", problem_validator)
    builder.add_node("technical_feasibility", technical_feasibility)
    builder.add_node("accelerator_fit", accelerator_fit)
    builder.add_node("user_personas", user_personas)
    builder.add_node("devils_advocate", devils_advocate)
    builder.add_node("vc_final_call", vc_final_call)

    # ---- Entry point: START -> pitch_refiner (sequential, single edge) ----
    builder.add_edge(START, "pitch_refiner")

    # ---- STAGE 1 FAN-OUT: pitch_refiner -> 3 independent nodes ----
    # Three separate edges FROM THE SAME SOURCE is what creates the fan-out.
    # Each of these three nodes becomes "ready" the instant pitch_refiner
    # completes, and none of them has an edge to either of the others, so
    # LangGraph's executor runs all three concurrently.
    builder.add_edge("pitch_refiner", "market_intelligence")
    builder.add_edge("pitch_refiner", "problem_validator")
    builder.add_edge("pitch_refiner", "technical_feasibility")

    # ---- STAGE 1 FAN-IN -> STAGE 2 FAN-OUT ----
    # Passing a LIST of predecessor nodes into add_edge is the documented
    # LangGraph idiom for a synchronization barrier: the graph will not
    # fire any Stage 2 node until ALL THREE Stage 1 nodes have completed,
    # even though they finish at different times depending on web_search
    # latency, cache hits, etc.
    stage_1_nodes = ["market_intelligence", "problem_validator", "technical_feasibility"]
    builder.add_edge(stage_1_nodes, "accelerator_fit")
    builder.add_edge(stage_1_nodes, "user_personas")
    builder.add_edge(stage_1_nodes, "devils_advocate")

    # ---- STAGE 2 FAN-IN -> final sequential synthesis ----
    # Same barrier pattern: vc_final_call only fires once accelerator_fit,
    # user_personas, AND devils_advocate have all written their state.
    stage_2_nodes = ["accelerator_fit", "user_personas", "devils_advocate"]
    builder.add_edge(stage_2_nodes, "vc_final_call")

    # ---- Exit point ----
    builder.add_edge("vc_final_call", END)

    # .compile() performs structural validation (e.g. catches dead-end nodes
    # with no path to END) and returns a CompiledStateGraph that supports
    # .invoke(), .stream(), and their async counterparts.
    return builder.compile()


# Module-level singleton — built once on import, reused across Streamlit
# reruns rather than rebuilding the graph object on every user interaction.
venture_graph = build_graph()


if __name__ == "__main__":
    # Quick structural sanity check you can run directly:
    #   python -m src.graph
    # This does NOT call the Anthropic API — it just confirms the graph
    # compiles and prints its node/edge structure, useful for debugging
    # the topology without spending API credits.
    graph = build_graph()
    print("Graph compiled successfully.")
    print("Nodes:", list(graph.get_graph().nodes.keys()))
    print("\nMermaid diagram source (paste into https://mermaid.live):\n")
    print(graph.get_graph().draw_mermaid())
