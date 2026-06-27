import time
import json
from dotenv import load_dotenv

load_dotenv()

from src.graph import venture_graph
from src.rag import seed_accelerator_knowledge

PITCH = """
A marketplace for renting out unused backyard spaces as private dog parks, 
complete with secure fencing verification, local pet insurance, and hourly booking. 
Solves the problem of crowded, unsafe public dog parks for reactive or untrained dogs.
"""

EMPTY_STATE = {
    "raw_pitch": PITCH,
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

if __name__ == "__main__":
    print("Seeding knowledge base...")
    seed_accelerator_knowledge()
    
    print("\n" + "="*60)
    print("NEW STARTUP IDEA:")
    print(PITCH.strip())
    print("="*60 + "\n")
    
    start_time = time.perf_counter()
    
    final_state = None
    # stream_mode="values" yields the full state after each step is processed
    for step in venture_graph.stream(EMPTY_STATE, stream_mode="values"):
        final_state = step
        stage = step.get("stage")
        print(f"[{time.perf_counter() - start_time:.2f}s] State updated to stage: {stage}")
        
    total_time = time.perf_counter() - start_time
    print(f"\nTOTAL PIPELINE EXECUTION TIME: {total_time:.2f}s\n")
    
    keys_to_print = [
        "refined_pitch", 
        "market_analysis", 
        "problem_validation", 
        "technical_feasibility", 
        "accelerator_fit", 
        "user_personas", 
        "devils_advocate", 
        "vc_verdict"
    ]
    
    with open("test_run_output.json", "w") as f:
        json.dump(final_state, f, indent=2)
        
    for key in keys_to_print:
        print(f"\n{'='*60}")
        print(f"AGENT OUTPUT: {key.upper()}")
        print(f"{'='*60}")
        print(json.dumps(final_state.get(key, {}), indent=2))
