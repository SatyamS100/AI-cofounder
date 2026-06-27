"""
benchmark.py
============
Measures and compares:
  1. Simulated sequential runtime (sum of individual agent timings)
  2. Actual parallel pipeline runtime (real LangGraph run)

Run with:
    python benchmark.py

This gives you REAL numbers for your CV and interview claims.
No mocking — actual Claude API calls, actual timing.
"""

import time
import json
import asyncio
import statistics
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from src.graph import venture_graph
from src.agents import (
    pitch_refiner, market_intelligence, problem_validator,
    technical_feasibility, accelerator_fit, user_personas,
    devils_advocate, vc_final_call,
)
from src.rag import seed_accelerator_knowledge

# ─── TEST PITCH ───────────────────────────────────────────────────────────────
# Use a realistic but generic pitch so results are representative
TEST_PITCH = """
An AI-powered platform that helps college students in India find and book 
verified home-cooked meal tiffin services from local home chefs near their 
hostels, with group ordering discounts, dietary preference filters, and 
weekly subscription plans. Solves the problem of unhealthy, expensive hostel 
food with affordable, home-style meals.
"""

EMPTY_STATE = {
    "raw_pitch": TEST_PITCH,
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


# ─── HELPER: Time a single agent call ─────────────────────────────────────────

def time_agent(agent_fn, state: dict, label: str) -> tuple[float, dict]:
    """
    Runs one agent function, times it, returns (elapsed_seconds, result).
    Used to measure individual agent latency for the sequential baseline.
    """
    start = time.perf_counter()
    result = agent_fn(state)
    elapsed = time.perf_counter() - start
    print(f"  {label:<35} {elapsed:>6.2f}s")
    return elapsed, result


# ─── SEQUENTIAL SIMULATION ────────────────────────────────────────────────────

def run_sequential_benchmark(state: dict) -> dict:
    """
    Runs ALL agents in strict sequence (no parallelism) and records each
    agent's individual time. This is the 'worst case without parallelism'
    baseline — equivalent to a naive linear agent chain.
    
    IMPORTANT: This modifies and accumulates state as it goes, so each
    agent has the output of previous agents available (same as real pipeline).
    The only difference from the real pipeline: NO parallel execution.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL BASELINE (no parallelism)")
    print("="*60)
    
    timings = {}
    accumulated = dict(state)
    
    # Stage 0
    t, r = time_agent(pitch_refiner, accumulated, "pitch_refiner")
    timings["pitch_refiner"] = t
    accumulated.update(r)
    
    # Stage 1 — run sequentially one after another
    t, r = time_agent(market_intelligence, accumulated, "market_intelligence")
    timings["market_intelligence"] = t
    accumulated.update(r)
    
    t, r = time_agent(problem_validator, accumulated, "problem_validator")
    timings["problem_validator"] = t
    accumulated.update(r)
    
    t, r = time_agent(technical_feasibility, accumulated, "technical_feasibility")
    timings["technical_feasibility"] = t
    accumulated.update(r)
    
    # Stage 2 — run sequentially
    t, r = time_agent(accelerator_fit, accumulated, "accelerator_fit")
    timings["accelerator_fit"] = t
    accumulated.update(r)
    
    t, r = time_agent(user_personas, accumulated, "user_personas")
    timings["user_personas"] = t
    accumulated.update(r)
    
    t, r = time_agent(devils_advocate, accumulated, "devils_advocate")
    timings["devils_advocate"] = t
    accumulated.update(r)
    
    # Stage 3
    t, r = time_agent(vc_final_call, accumulated, "vc_final_call")
    timings["vc_final_call"] = t
    accumulated.update(r)
    
    total = sum(timings.values())
    print(f"\n  {'SEQUENTIAL TOTAL':<35} {total:>6.2f}s")
    
    return timings


# ─── PARALLEL PIPELINE (real LangGraph run) ───────────────────────────────────

def run_parallel_benchmark(state: dict) -> dict:
    """
    Runs the actual compiled LangGraph pipeline with real parallel execution.
    Times each node as it completes (from LangGraph's .stream() yield order)
    and records total wall-clock time.
    
    The difference between this total and the sequential total is the 
    real, measured speedup from parallelism.
    """
    print("\n" + "="*60)
    print("PARALLEL PIPELINE (real LangGraph execution)")
    print("="*60)
    
    node_timings = {}
    stage_timings = {0: [], 1: [], 2: [], 3: []}
    
    from config import AGENT_METADATA
    
    wall_start = time.perf_counter()
    last_tick = wall_start
    
    for step in venture_graph.stream(state):
        now = time.perf_counter()
        for node_name, node_output in step.items():
            # Time since last node completed (approximates node duration
            # for sequential nodes; for parallel nodes, this is the time
            # since the LAST parallel node finished, not individual durations)
            elapsed_since_last = now - last_tick
            node_timings[node_name] = elapsed_since_last
            stage = AGENT_METADATA.get(node_name, {}).get("stage", 0)
            stage_timings[stage].append(elapsed_since_last)
            print(f"  {node_name:<35} completed at {now - wall_start:>6.2f}s")
        last_tick = now
    
    total_wall = time.perf_counter() - wall_start
    print(f"\n  {'PARALLEL TOTAL (wall clock)':<35} {total_wall:>6.2f}s")
    
    return {"node_timings": node_timings, "total_wall": total_wall, "stage_timings": stage_timings}


# ─── CACHE BENEFIT MEASUREMENT ────────────────────────────────────────────────

def run_cached_benchmark(state: dict) -> float:
    """
    Runs the pipeline a SECOND TIME with the same pitch to measure the
    cache hit speedup on market_intelligence. The second run should
    skip the web search entirely for market analysis.
    """
    print("\n" + "="*60)
    print("CACHED RUN (same pitch — should hit pitch analysis cache)")
    print("="*60)
    
    wall_start = time.perf_counter()
    for step in venture_graph.stream(state):
        for node_name, node_output in step.items():
            now = time.perf_counter()
            # Check if market_intelligence used cache
            if node_name == "market_intelligence":
                data_source = node_output.get("market_analysis", {}).get("data_source", "unknown")
                print(f"  market_intelligence: data_source = {data_source} ✓" if data_source == "cached" 
                      else f"  market_intelligence: data_source = {data_source} (cache MISS)")
            print(f"  {node_name:<35} completed at {now - wall_start:>6.2f}s")
    
    total = time.perf_counter() - wall_start
    print(f"\n  {'CACHED RUN TOTAL':<35} {total:>6.2f}s")
    return total


# ─── RESULTS ANALYSIS ─────────────────────────────────────────────────────────

def print_analysis(seq_timings: dict, parallel_result: dict, cached_total: float):
    seq_total = sum(seq_timings.values())
    par_total = parallel_result["total_wall"]
    
    speedup = seq_total / par_total
    time_saved = seq_total - par_total
    pct_saved = (time_saved / seq_total) * 100
    cache_speedup = par_total / cached_total
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS — USE THESE IN YOUR CV")
    print("="*60)
    
    print(f"""
  Sequential runtime (no parallelism):  {seq_total:.1f}s
  Parallel runtime (LangGraph):          {par_total:.1f}s
  Cached run (pitch cache hit):          {cached_total:.1f}s

  Speedup from parallelism:             {speedup:.1f}x
  Time saved by parallelism:            {time_saved:.1f}s ({pct_saved:.0f}% reduction)
  Additional speedup from cache:        {cache_speedup:.1f}x (vs parallel fresh run)

  ─── PER-STAGE BREAKDOWN ──────────────────────────────
  Stage 0 (pitch_refiner):              {seq_timings['pitch_refiner']:.1f}s (sequential only)
  Stage 1 sequential would be:          {seq_timings['market_intelligence'] + seq_timings['problem_validator'] + seq_timings['technical_feasibility']:.1f}s
  Stage 1 parallel wall time:           {max(parallel_result['stage_timings'][1]):.1f}s  ← bottleneck agent
  Stage 1 time saved:                   {seq_timings['market_intelligence'] + seq_timings['problem_validator'] + seq_timings['technical_feasibility'] - max(parallel_result['stage_timings'][1]):.1f}s

  Stage 2 sequential would be:          {seq_timings['accelerator_fit'] + seq_timings['user_personas'] + seq_timings['devils_advocate']:.1f}s
  Stage 2 parallel wall time:           {max(parallel_result['stage_timings'][2]):.1f}s  ← bottleneck agent
  Stage 2 time saved:                   {seq_timings['accelerator_fit'] + seq_timings['user_personas'] + seq_timings['devils_advocate'] - max(parallel_result['stage_timings'][2]):.1f}s

  Stage 3 (vc_final_call):              {seq_timings['vc_final_call']:.1f}s (sequential only)
""")

    print("  ─── YOUR CV BULLET POINT NUMBERS ──────────────────")
    print(f"  'Reduced runtime from {seq_total:.0f}s to {par_total:.0f}s'")
    print(f"  'Cut latency by {pct_saved:.0f}% via {2}-stage parallel execution'")
    print(f"  'Cache hits reduce repeat-market runs to {cached_total:.0f}s ({speedup * cache_speedup:.1f}x total speedup)'")
    print()
    
    # Save raw results
    results = {
        "sequential_total_s": round(seq_total, 2),
        "parallel_total_s": round(par_total, 2),
        "cached_total_s": round(cached_total, 2),
        "speedup_x": round(speedup, 2),
        "pct_time_saved": round(pct_saved, 1),
        "cache_speedup_x": round(cache_speedup, 2),
        "per_agent_sequential_s": {k: round(v, 2) for k, v in seq_timings.items()},
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Full results saved to benchmark_results.json")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Seeding knowledge base...")
    seed_accelerator_knowledge()
    
    print(f"\nTest pitch: {TEST_PITCH[:80].strip()}...")
    
    # Run 1: Sequential (no parallelism baseline)
    seq_timings = run_sequential_benchmark(EMPTY_STATE)
    
    # Run 2: Real parallel pipeline (fresh — no cache)
    parallel_result = run_parallel_benchmark(EMPTY_STATE)
    
    # Run 3: Same pitch again — should hit pitch analysis cache
    cached_total = run_cached_benchmark(EMPTY_STATE)
    
    # Analysis + CV numbers
    print_analysis(seq_timings, parallel_result, cached_total)
