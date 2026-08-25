"""
config.py
=========
Single source of truth for constants used across the project: model name,
retry/timeout policy, and agent display metadata for the UI.

DESIGN DECISION:
Nothing in this file is a secret (the API key lives in .env and is loaded via
python-dotenv in agents.py). Everything here is application logic that
happens to be configurable — it's checked into version control on purpose,
so changing the model or the retry policy is a one-line, reviewable diff
instead of a hunt through 8 agent functions.
"""

# ─────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile was retired by Groq Aug 2026

# Per-agent max_tokens. Kept modest (1500) because every agent returns a
# single structured JSON object, not free-form prose — there's no reason to
# pay for a larger budget than the schema requires. Bumping this is the
# first thing to try if an agent's JSON gets truncated mid-object.
MAX_TOKENS = 1500


# ─────────────────────────────────────────────────────────────────────────
# RETRY / RESILIENCE POLICY
# ─────────────────────────────────────────────────────────────────────────
# Implements the "pause → retry once → degrade gracefully" behavior designed
# in state.py's make_failed_result(). These numbers are deliberately small:
# this is a synchronous, user-facing UI flow (the user is staring at a
# Streamlit progress bar), not a background batch job, so we don't want a
# slow agent to make the whole pipeline feel broken.
MAX_RETRIES = 1            # one retry attempt, then give up and degrade
RETRY_BACKOFF_SECONDS = 3  # pause before the single retry
API_TIMEOUT_SECONDS = 30   # per-call timeout passed to the Anthropic client

# Groq free tier is 30 RPM — semaphore of 5 is safe
API_SEMAPHORE_LIMIT = 5


# ─────────────────────────────────────────────────────────────────────────
# AGENT METADATA — used by graph.py (node naming) and app.py (UI labels)
# ─────────────────────────────────────────────────────────────────────────
# Keeping internal key -> display name -> pipeline stage in one dict means
# the Streamlit progress UI and the LangGraph node names can never silently
# drift apart. If I rename an agent's state key, the UI label updates
# automatically because both pull from this same dict.
AGENT_METADATA = {
    "pitch_refiner": {
        "display_name": "Pitch Refiner",
        "stage": 0,
        "uses_web_search": False,
    },
    "market_intelligence": {
        "display_name": "Market Intelligence",
        "stage": 1,
        "uses_web_search": True,
    },
    "problem_validator": {
        "display_name": "Problem Validator",
        "stage": 1,
        "uses_web_search": True,
    },
    "technical_feasibility": {
        "display_name": "Technical Feasibility",
        "stage": 1,
        "uses_web_search": False,
    },
    "accelerator_fit": {
        "display_name": "Accelerator Fit",
        "stage": 2,
        "uses_web_search": False,  # grounded via RAG instead, see rag.py
    },
    "user_personas": {
        "display_name": "User Personas (x3)",
        "stage": 2,
        "uses_web_search": False,
    },
    "devils_advocate": {
        "display_name": "Devil's Advocate",
        "stage": 2,
        "uses_web_search": False,
    },
    "vc_final_call": {
        "display_name": "VC Final Call",
        "stage": 3,
        "uses_web_search": False,
    },
}

# Human-readable labels for the 4 progress indicators shown in app.py.
# Stage 0 and Stage 3 are sequential (single agent); Stage 1 and Stage 2 are
# the two parallel batches — this is the project's main interview talking
# point, so the UI is built to visually emphasize it.
STAGE_LABELS = {
    0: "Pitch Refined",
    1: "Market + Validation + Feasibility",
    2: "Fit + Personas + Risk",
    3: "VC Verdict",
}
