"""
agents.py
=========
All agent functions using Groq (LLM) + Tavily (search).

ARCHITECTURE:
  - Groq SDK: handles all LLM generation (Llama 3.3 70B)
  - Tavily (via tools.py): handles web search independently
  - Search results are injected into prompts BEFORE the LLM call
    (pre-retrieval injection pattern)

WHY GROQ:
  - 30 RPM free tier vs Gemini's 5 RPM
  - ~1s response latency vs Gemini's 5-8s
  - Llama 3.3 70B produces structured JSON reliably
  - No daily request caps that kill a demo mid-run

WHY PRE-RETRIEVAL INJECTION VS SERVER-SIDE GROUNDING:
  Server-side grounding (Gemini): search happens inside the API call,
  invisible to your code, coupled to the LLM provider.
  
  Pre-retrieval injection (our approach): your code calls Tavily,
  gets structured results, injects them as context into the prompt,
  then calls the LLM. Full observability, provider independence,
  ability to cache/filter search results before the LLM sees them.
"""

import json
import re
import time
import os
import threading
from groq import Groq

from config import (
    MODEL_NAME, MAX_TOKENS, MAX_RETRIES,
    RETRY_BACKOFF_SECONDS, API_SEMAPHORE_LIMIT
)
from src.tools import web_search, multi_search
from src.state import make_failed_result
from src.rag import (
    query_accelerator_knowledge,
    find_similar_pitch_analysis,
    store_pitch_analysis,
)

from dotenv import load_dotenv
load_dotenv()

# ─── GROQ CLIENT ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─── CONCURRENCY CONTROL ──────────────────────────────────────────────────────
# Groq free tier: 30 RPM. With 3 parallel agents, semaphore of 5 is
# very safe — we'll never hit rate limits even with retries.
_api_semaphore = threading.Semaphore(API_SEMAPHORE_LIMIT)


# ─── SHARED INFRASTRUCTURE ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Tolerant JSON extractor — unchanged from previous versions.
    Llama 3.3 70B is very good at JSON compliance but we keep the
    fallback for robustness.
    """
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No valid JSON found in response: {text[:200]}...")


def _call_groq_with_retry(
    agent_key: str,
    system_prompt: str,
    user_content: str,
) -> dict:
    """
    Groq API call with semaphore-based concurrency control and
    retry-then-degrade resilience.

    Groq API uses the OpenAI SDK format:
      - system message passed as {"role": "system", "content": system_prompt}
      - user message passed as {"role": "user", "content": user_content}
      - response at response.choices[0].message.content (string)

    This is the same interface as OpenAI's SDK — if you ever switch to
    OpenAI, only the client initialization line changes.
    """
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            with _api_semaphore:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                    # Groq supports response_format for JSON mode:
                    response_format={"type": "json_object"},
                )

            return _extract_json(response.choices[0].message.content)

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue

    return make_failed_result(agent_key, last_error)


# Alias for compatibility with function calls below
_call_claude_with_retry = _call_groq_with_retry


# ─────────────────────────────────────────────────────────────────────────
# AGENT 1 — pitch_refiner (sequential, runs first)
# ─────────────────────────────────────────────────────────────────────────

PITCH_REFINER_SYSTEM_PROMPT = """You are a legendary entrepreneur combining \
the product instincts of Steve Jobs and the strategic clarity of Sam Altman. \
You receive a rough startup idea and sharpen it ruthlessly.

Identify:
- The core insight the founder may not have articulated themselves
- The real target customer (a specific group, not "everyone")
- The unfair advantage that makes this defensible
- A single, sharp one-line value proposition

Return ONLY a JSON object with this exact shape, no other text:
{
  "refined_pitch": "2-3 sentence sharpened version of the idea",
  "core_insight": "the underlying insight, one sentence",
  "target_customer": "specific customer segment, not 'everyone'",
  "unfair_advantage": "what makes this hard to copy",
  "one_liner": "single sentence value proposition",
  "confidence_score": <integer 1-10, how strong is this pitch as given>
}"""


def pitch_refiner(state: dict) -> dict:
    """
    First node in the graph. Takes the user's raw, possibly messy pitch and
    runs it through a single Claude call to extract a sharpened version plus
    structured metadata (core insight, target customer, etc.) that every
    downstream agent will read instead of re-parsing the raw text themselves.

    WHY THIS RUNS FIRST AND ALONE (not parallelized):
    Every other agent depends on its output. There's nothing to parallelize
    here — it's the one genuinely sequential dependency at the start of the
    graph, which is exactly why the graph shape is "refine → parallel →
    parallel → synthesize" rather than "parallel from the start."
    """
    result = _call_claude_with_retry(
        agent_key="pitch_refiner",
        system_prompt=PITCH_REFINER_SYSTEM_PROMPT,
        user_content=f"Raw startup idea:\n\n{state['raw_pitch']}",
    )
    return {"refined_pitch": result, "stage": "pitch_refined"}


# ─────────────────────────────────────────────────────────────────────────
# AGENT 2 — market_intelligence (Stage 1, parallel; uses web_search OR cache)
# ─────────────────────────────────────────────────────────────────────────

MARKET_INTELLIGENCE_SYSTEM_PROMPT = """You are a McKinsey senior analyst \
specializing in competitive intelligence. Use web_search extensively to find \
REAL, current data — not estimates from your training data.

Search for:
1. Market size: TAM/SAM/SOM with sources, CAGR from recent reports
2. Market timing: what is driving growth RIGHT NOW? (new regulation, new \
   technology, demographic shift, post-COVID behavior change?)
3. Competitor deep-dive: for each of the top 3-5 competitors, find:
   - Their actual funding amount and investors (search Crunchbase/TechCrunch)
   - Their pricing model
   - What customers LOVE about them (read their positive reviews)
   - What customers HATE about them (read their negative reviews on G2, \
     Capterra, Reddit, Twitter — these are the gaps your startup can exploit)
   - Their go-to-market strategy
   - Their obvious weaknesses or blind spots
4. Recent news: any acquisitions, funding rounds, or major pivots in this \
   space in the last 12 months?

Return ONLY a JSON object with this exact shape, no other text:
{
  "tam_usd": <number>,
  "sam_usd": <number>,
  "som_usd": <number>,
  "cagr_pct": <number>,
  "market_timing": {
    "assessment": "growing|shrinking|stable",
    "key_driver": "what is specifically driving this right now",
    "window": "how long this opportunity window likely lasts"
  },
  "competitors": [
    {
      "name": "Company Name",
      "funding_usd": <number or null if unknown>,
      "investors": ["Investor 1", "Investor 2"],
      "pricing": "their pricing model",
      "where_they_excel": ["strength 1", "strength 2"],
      "where_they_lag": ["weakness 1", "weakness 2"],
      "customer_complaints": ["real complaint from reviews", "another complaint"],
      "gap_to_exploit": "specific opening this startup can target"
    }
  ],
  "market_gaps": ["gap 1 not served by any competitor", "gap 2"],
  "recent_news": ["relevant development 1", "relevant development 2"],
  "key_findings": ["most important insight 1", "insight 2", "insight 3"],
  "data_source": "live_web_search",
  "confidence_score": <integer 1-10>
}"""


def market_intelligence(state: dict) -> dict:
    """
    Cache-first market research agent.
    
    SEARCH PATTERN CHANGE:
    Old: pass tools=[WEB_SEARCH_TOOL] to Gemini, search happens inside API call
    New: call Tavily directly, inject results into prompt, call Groq for analysis
    
    We run 3 targeted searches instead of one broad search:
      1. Market size and TAM for the specific category
      2. Key competitors and their funding
      3. Market timing and recent trends
    This produces more structured, cited results than one broad query.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])
    one_liner = state["refined_pitch"].get("one_liner", refined_text[:100])

    # Cache check first — skip search entirely if we have recent results
    cached = find_similar_pitch_analysis(refined_text)
    if cached is not None:
        market_data = dict(cached["market_analysis"])
        market_data["data_source"] = "cached"
        market_data["cached_from_pitch"] = cached["matched_pitch_summary"]
        market_data["cached_age_days"] = cached["cached_age_days"]
        return {"market_analysis": market_data, "stage": "stage_1_parallel"}

    # Run targeted searches BEFORE the LLM call
    search_results = multi_search([
        f"{one_liner} market size TAM total addressable market 2024 2025",
        f"{one_liner} competitors funding raised investors",
        f"{one_liner} market growth trends recent news",
    ])

    # Inject search results into the prompt
    user_content = (
        f"LIVE MARKET RESEARCH DATA (from web search conducted just now):\n"
        f"{search_results}\n\n"
        f"STARTUP PITCH TO ANALYZE:\n{refined_text}\n\n"
        f"Use the search data above to ground your analysis in real numbers. "
        f"Cite specific figures from the search results where available."
    )

    result = _call_groq_with_retry(
        agent_key="market_intelligence",
        system_prompt=MARKET_INTELLIGENCE_SYSTEM_PROMPT,
        user_content=user_content,
    )

    if result.get("status") != "failed":
        result["data_source"] = "live_web_search"
        store_pitch_analysis(refined_text, result)

    return {"market_analysis": result, "stage": "stage_1_parallel"}


# ─────────────────────────────────────────────────────────────────────────
# AGENT 3 — problem_validator (Stage 1, parallel; uses web_search)
# ─────────────────────────────────────────────────────────────────────────

PROBLEM_VALIDATOR_SYSTEM_PROMPT = """You are a customer development expert \
who has personally conducted 10,000 customer interviews. Your job is to find \
REAL evidence that this problem exists and people suffer from it.

Use web_search to find evidence across these specific sources:
- Reddit: search r/entrepreneur, r/startups, and topic-specific subreddits \
  for complaints, frustration posts, and "how do I solve X" threads
- Product Hunt: look for similar products launched before and read the \
  comments to understand what pain they addressed
- G2 / Capterra / Trustpilot: find reviews of existing solutions in this \
  space and extract the most common complaints (these are real user pain)
- Twitter/X: find people actively complaining about this problem
- Quora / Stack Overflow / industry forums: find questions people ask \
  about solving this problem
- Blog posts and articles: find thought leaders writing about this pain
- App Store / Play Store reviews: if there are existing apps, read the 1-3 \
  star reviews — they are the most honest pain articulation that exists

For each source, you are looking for:
1. HOW MANY people express this pain (volume signal)
2. HOW LOUDLY they express it (severity — are they frustrated or just mildly \
   inconvenienced?)
3. HOW OFTEN it happens (frequency — daily pain vs. once a year pain)
4. WHETHER they have tried to solve it already (strongest signal of real pain \
   is people hacking together workarounds)

Return ONLY a JSON object with this exact shape, no other text:
{
  "problem_severity_score": <integer 1-10>,
  "evidence_sources": [
    {
      "source": "Reddit r/entrepreneur",
      "what_was_found": "description of evidence found",
      "volume": "how many posts/comments",
      "severity": "high/medium/low"
    }
  ],
  "estimated_people_affected": "rough order-of-magnitude with reasoning",
  "pain_frequency": "daily|weekly|monthly|rare — with specific context",
  "willingness_to_pay_signal": "strong|moderate|weak|none — cite specific evidence",
  "existing_workarounds": ["workaround 1 people are using", "workaround 2"],
  "key_quotes": [
    {
      "quote": "paraphrased real complaint from a forum or review",
      "source": "where this came from"
    }
  ],
  "verdict": "VALIDATED|PARTIALLY_VALIDATED|UNVALIDATED — one sentence why",
  "confidence_score": <integer 1-10>
}"""


def problem_validator(state: dict) -> dict:
    """
    Problem validation agent with pre-retrieved social proof data.
    
    Searches Reddit, forums, and review sites for evidence of the pain,
    injects that evidence into the prompt, then asks Groq to analyze it.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])
    one_liner = state["refined_pitch"].get("one_liner", refined_text[:100])

    # Search for real user pain evidence
    search_results = multi_search([
        f"{one_liner} problems complaints Reddit forum users frustrated",
        f"{one_liner} reviews G2 Capterra alternatives users want",
    ])

    user_content = (
        f"SOCIAL PROOF AND PAIN EVIDENCE (from web search):\n"
        f"{search_results}\n\n"
        f"STARTUP PITCH:\n{refined_text}\n\n"
        f"Analyze the search evidence above to validate whether this problem "
        f"is real, how severe it is, and how many people experience it."
    )

    result = _call_groq_with_retry(
        agent_key="problem_validator",
        system_prompt=PROBLEM_VALIDATOR_SYSTEM_PROMPT,
        user_content=user_content,
    )
    return {"problem_validation": result, "stage": "stage_1_parallel"}


# ─────────────────────────────────────────────────────────────────────────
# AGENT 4 — technical_feasibility (Stage 1, parallel; no tools needed)
# ─────────────────────────────────────────────────────────────────────────

TECHNICAL_FEASIBILITY_SYSTEM_PROMPT = """You are a CTO with 20 years of \
experience building products at scale. Assess: can this be built with \
current technology? What is the single hardest technical challenge? What \
is the minimum viable team to build an MVP? What's a realistic timeline?

Return ONLY a JSON object with this exact shape, no other text:
{
  "buildability_score": <integer 1-10>,
  "hardest_challenge": "the single hardest technical problem to solve",
  "min_team_size": <integer>,
  "mvp_timeline_months": <number>,
  "key_technical_risks": ["risk 1", "risk 2"],
  "confidence_score": <integer 1-10>
}"""


def technical_feasibility(state: dict) -> dict:
    """
    Runs in PARALLEL with market_intelligence and problem_validator
    (Stage 1). No web_search tool — engineering buildability is assessed
    from the model's own technical knowledge and the refined pitch, not
    from live external data, so giving it search access would only add
    latency without improving the answer.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])
    result = _call_claude_with_retry(
        agent_key="technical_feasibility",
        system_prompt=TECHNICAL_FEASIBILITY_SYSTEM_PROMPT,
        user_content=f"Startup pitch — assess technical feasibility:\n\n{refined_text}",
    )
    return {"technical_feasibility": result, "stage": "stage_1_parallel"}


# ─────────────────────────────────────────────────────────────────────────
# AGENT 5 — accelerator_fit (Stage 2, parallel; RAG-grounded, no web_search)
# ─────────────────────────────────────────────────────────────────────────

ACCELERATOR_FIT_SYSTEM_PROMPT = """You are a former YC partner who personally \
reviewed 5,000 applications. You will be given grounding notes retrieved \
from real YC/Sequoia/a16z criteria, followed by a startup pitch. Use the \
grounding notes to judge whether this startup has the hallmarks of a top \
accelerator company: founder-market fit, large or fast-growing market, a \
genuinely differentiated insight, and credible growth potential.

Return ONLY a JSON object with this exact shape, no other text:
{
  "fit_score_1_to_10": <integer>,
  "yc_similarity": "how closely this resembles a strong YC application, and why",
  "similar_funded_companies": ["company 1", "company 2"],
  "key_strengths_for_investors": ["strength 1", "strength 2"],
  "key_weaknesses": ["weakness 1", "weakness 2"],
  "confidence_score": <integer 1-10>
}"""


def accelerator_fit(state: dict) -> dict:
    """
    Runs in PARALLEL with user_personas and devils_advocate (Stage 2).

    THE RAG STEP HAPPENS HERE, BEFORE THE CLAUDE CALL:
    Rather than asking Claude to recall "what does YC look for" purely from
    training data (which risks confident-sounding but ungrounded claims),
    we retrieve the most relevant snippets from our seeded accelerator
    knowledge base (src/rag.py) and inject them directly into the prompt.
    The LLM call happens AFTER retrieval — that ordering is the entire
    definition of "RAG": retrieve, then generate.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])

    grounding_snippets = query_accelerator_knowledge(refined_text, n_results=4)
    grounding_block = "\n".join(f"- {s}" for s in grounding_snippets)

    user_content = (
        f"GROUNDING NOTES (retrieved from real accelerator criteria):\n"
        f"{grounding_block}\n\n"
        f"STARTUP PITCH TO EVALUATE:\n{refined_text}"
    )

    result = _call_claude_with_retry(
        agent_key="accelerator_fit",
        system_prompt=ACCELERATOR_FIT_SYSTEM_PROMPT,
        user_content=user_content,
    )
    return {"accelerator_fit": result, "stage": "stage_2_parallel"}


# ─────────────────────────────────────────────────────────────────────────
# AGENTS 6, 7, 8 — three user persona agents (Stage 2, parallel)
# ─────────────────────────────────────────────────────────────────────────
# WHY THREE SEPARATE PROMPTS INSTEAD OF ONE "GENERATE 3 PERSONAS" CALL:
# A single call asked to produce 3 distinct reactions tends to make them
# subtly similar in tone — the model anchors on its own first answer when
# generating the second and third in the same response. Three independent
# calls with three distinct, detailed personas (different age, tech comfort,
# budget authority, and even a different REQUIRED JSON SCHEMA per persona)
# produce genuinely more differentiated reactions, at the cost of 3 calls
# instead of 1. Since they're independent, they run in the same parallel
# graph stage as accelerator_fit and devils_advocate, so this costs latency
# only if it's the slowest branch — and in practice it isn't.

PERSONA_EARLY_ADOPTER_SYSTEM_PROMPT = """You ARE a tech-savvy early adopter: \
26 years old, you discover new tools on Product Hunt and Hacker News, and \
you have disposable income. React to the following pitch in FIRST PERSON, \
as yourself — not as an analyst describing this persona.

Return ONLY a JSON object with this exact shape, no other text:
{
  "persona_name": "Early Adopter",
  "would_use": <true|false>,
  "monthly_price_willing_to_pay_usd": <number>,
  "biggest_objection": "your honest, specific objection",
  "virality_trigger": "what would make you tell 5 friends about this",
  "excitement_score": <integer 1-10>
}"""

PERSONA_MAINSTREAM_SYSTEM_PROMPT = """You ARE a mainstream user: 42 years \
old, an office manager, not very tech-savvy, you only adopt new software if \
your IT department recommends it or a coworker shows you. React to the \
following pitch in FIRST PERSON, as yourself.

Return ONLY a JSON object with this exact shape, no other text:
{
  "persona_name": "Mainstream User",
  "would_use": <true|false>,
  "monthly_price_willing_to_pay_usd": <number>,
  "biggest_objection": "your honest, specific objection",
  "switching_trigger": "what would make you switch from your current solution",
  "excitement_score": <integer 1-10>
}"""

PERSONA_ENTERPRISE_SYSTEM_PROMPT = """You ARE a skeptical enterprise buyer: \
50 years old, a VP at a mid-size company, personally responsible for \
software budget. You evaluate every purchase on security, compliance, ROI, \
and vendor stability. React to the following pitch in FIRST PERSON, as \
yourself.

Return ONLY a JSON object with this exact shape, no other text:
{
  "persona_name": "Enterprise Buyer",
  "would_approve_budget": <true|false>,
  "annual_budget_willing_usd": <number>,
  "key_compliance_concerns": ["concern 1", "concern 2"],
  "roi_requirement": "what ROI threshold would justify this purchase",
  "excitement_score": <integer 1-10>
}"""


def _run_single_persona(system_prompt: str, agent_key: str, refined_text: str) -> dict:
    """Shared body for the 3 persona functions below — they differ only in
    which system prompt and agent_key they pass in."""
    return _call_claude_with_retry(
        agent_key=agent_key,
        system_prompt=system_prompt,
        user_content=f"Startup pitch:\n\n{refined_text}",
    )


def user_personas(state: dict) -> dict:
    """
    Runs all 3 persona reactions and combines them into the single
    `user_personas` list state key.

    NOTE ON GRAPH WIRING: even though there are 3 distinct prompts/calls
    happening here, this is implemented as ONE LangGraph node (not 3
    separate nodes) because the 3 calls are trivially independent of each
    other and don't need LangGraph's graph-level parallelism to coordinate
    them — a plain Python loop inside one node is simpler and equally fast.
    See graph.py for the discussion of where graph-level parallel nodes
    are actually worth it vs. where a loop suffices.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])

    early_adopter = _run_single_persona(
        PERSONA_EARLY_ADOPTER_SYSTEM_PROMPT, "persona_early_adopter", refined_text
    )
    mainstream = _run_single_persona(
        PERSONA_MAINSTREAM_SYSTEM_PROMPT, "persona_mainstream", refined_text
    )
    enterprise = _run_single_persona(
        PERSONA_ENTERPRISE_SYSTEM_PROMPT, "persona_enterprise", refined_text
    )

    return {
        "user_personas": [early_adopter, mainstream, enterprise],
        "stage": "stage_2_parallel",
    }


# ─────────────────────────────────────────────────────────────────────────
# AGENT 9 — devils_advocate (Stage 2, parallel)
# ─────────────────────────────────────────────────────────────────────────

DEVILS_ADVOCATE_SYSTEM_PROMPT = """You are a famously skeptical investor \
known for finding the fatal flaw in a pitch before anyone else does. Your \
job is to try to KILL this idea. Find every plausible reason it will fail: \
regulatory risk, existing incumbents with distribution advantages, market \
timing problems, team/execution gaps, and misconceptions about the market.

Return ONLY a JSON object with this exact shape, no other text:
{
  "fatal_flaw_risk_score": <integer 1-10, 10 = very likely to fail>,
  "regulatory_risks": ["risk 1", "risk 2"],
  "incumbent_threats": ["incumbent 1 and why they're a threat"],
  "timing_risks": "why now might be the wrong time, or 'none identified'",
  "team_gaps": "what capability gaps would sink execution",
  "most_likely_failure_mode": "the single most probable way this fails",
  "confidence_score": <integer 1-10>
}"""


def devils_advocate(state: dict) -> dict:
    """
    Runs in PARALLEL with accelerator_fit and user_personas (Stage 2).
    Deliberately adversarial system prompt — see vc_final_call below for
    why having one agent whose ENTIRE JOB is to disagree with the optimistic
    framing elsewhere in the pipeline produces a more balanced final memo
    than synthesizing only from agents that were never asked to find flaws.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])
    result = _call_claude_with_retry(
        agent_key="devils_advocate",
        system_prompt=DEVILS_ADVOCATE_SYSTEM_PROMPT,
        user_content=f"Startup pitch — find the fatal flaws:\n\n{refined_text}",
    )
    return {"devils_advocate": result, "stage": "stage_2_parallel"}


# ─────────────────────────────────────────────────────────────────────────
# AGENT 10 — vc_final_call (sequential, runs LAST — the synthesis agent)
# ─────────────────────────────────────────────────────────────────────────

VC_FINAL_CALL_SYSTEM_PROMPT = """You are a senior partner at a16z who also \
mentors early-stage founders. You are not here to just say yes or no — \
you are here to give the founder the most useful, honest, specific feedback \
they will ever receive. A one-word verdict with no reasoning helps nobody.

You have received independent analysis from 8 analysts. Some inputs may be \
marked UNAVAILABLE — reason around those gaps explicitly.

Your job has five parts:

PART 1 — WHERE THEY STAND NOW
Assess the current state honestly across: market opportunity, problem \
validation strength, technical feasibility, team gaps, and investor readiness. \
Be specific. "Market is large" is not useful. "TAM is $4B growing at 18% CAGR \
with 3 well-funded incumbents who all have weak mobile experiences" is useful.

PART 2 — WHAT VERDICT AND WHY
Assign one of four verdicts:
- Pass: fundamental issues that cannot be fixed without a different idea
- Watch: good bones but 2-3 specific things must be true before investing
- Invest: ready now with normal startup caveats
- Lead: exceptional — would compete to lead this round

The verdict must follow directly from Part 1. No surprises.

PART 3 — THE ROADMAP TO FUNDABLE
This is the most important part. If the verdict is Pass or Watch, give the \
founder a specific, ordered list of things they must do or prove to change \
that verdict. Be concrete: not "find product-market fit" but "get 50 paying \
customers at $X/month with less than Y% monthly churn, then come back."
If the verdict is Invest or Lead, give the conditions that would make you \
increase your conviction even further.

PART 4 — TEAM STRUCTURE RECOMMENDATION
Based on the pitch and the technical requirements, recommend:
- The critical first 3 hires (role, why this role specifically, what to look \
  for in that person)
- Any cofounder gaps that should be filled before fundraising
- Which functions should be hired vs. outsourced in the first 12 months

PART 5 — COMPARABLE COMPANIES AND WHAT TO LEARN FROM THEM
Use your knowledge to identify:
- 2-3 companies that went through a similar journey (ideally YC alumni or \
  known accelerator portfolio companies) — what did they do right in the \
  early days that this founder should copy?
- 1-2 companies that tried something similar and failed — what killed them \
  and how does this founder avoid that fate?
- If the market analysis identified specific competitors, name them and tell \
  the founder specifically which competitor's weakness is their biggest \
  near-term opportunity

Return ONLY a JSON object with this exact shape, no other text:
{
  "current_state_assessment": {
    "market_opportunity": "specific assessment with numbers from the analysis",
    "problem_validation_strength": "strong|moderate|weak — why",
    "technical_feasibility": "assessment of build complexity and team needs",
    "investor_readiness_score": <integer 1-10>,
    "biggest_current_strength": "the single strongest thing going for this",
    "biggest_current_gap": "the single most important thing missing"
  },
  "overall_signal_score_1_to_10": <integer>,
  "investment_verdict": "Pass|Watch|Invest|Lead",
  "verdict_reasoning": "2-3 sentences connecting the assessment to the verdict — no surprises",
  "key_bull_case": "strongest argument FOR investment",
  "key_bear_case": "strongest argument AGAINST investment",
  "roadmap_to_fundable": {
    "ordered_steps": [
      {
        "step": 1,
        "action": "specific thing to do or prove",
        "success_metric": "how you know this step is done",
        "timeline": "realistic timeframe"
      }
    ],
    "what_changes_the_verdict": "exactly what would make you upgrade Pass to Watch, Watch to Invest"
  },
  "team_structure_recommendation": {
    "critical_first_hires": [
      {
        "role": "specific role title",
        "why_critical": "why this role specifically, not another",
        "what_to_look_for": "the specific background or skill that matters"
      }
    ],
    "cofounder_gaps": "gaps that should be filled at cofounder level vs. hire level",
    "outsource_vs_hire": "which functions to outsource in first 12 months and why"
  },
  "comparable_companies": {
    "successful_parallels": [
      {
        "company": "Company Name",
        "accelerator_or_investor": "YC W21 / a16z / Sequoia etc.",
        "what_to_copy": "specific thing they did early that this founder should replicate"
      }
    ],
    "cautionary_tales": [
      {
        "company": "Company Name",
        "what_killed_them": "specific reason they failed",
        "how_to_avoid": "what this founder must do differently"
      }
    ],
    "competitor_opportunity": "which specific competitor weakness is the biggest near-term opening"
  },
  "critical_unknown": "the single biggest thing that needs to be true for this to work",
  "one_question_for_next_meeting": "sharpest question to ask the founder",
  "data_gaps_acknowledged": ["analysis that was unavailable, or empty list"]
}"""


def _format_agent_result(label: str, result: dict | None) -> str:
    """
    Renders one upstream agent's result as a labeled text block for the
    synthesis prompt — OR an explicit "UNAVAILABLE" marker if that agent
    failed after retry. This is what lets vc_final_call reason correctly
    about partial data instead of being confused by raw failure-marker JSON
    (status/reason/None fields) mixed in with real analysis fields.
    """
    if not result or result.get("status") == "failed":
        reason = (result or {}).get("reason", "unknown error")
        return f"### {label}\nUNAVAILABLE — this analysis failed after retry ({reason})."
    return f"### {label}\n{json.dumps(result, indent=2)}"


def vc_final_call(state: dict) -> dict:
    """
    Final node in the graph. Unlike every other agent, this one reads the
    ENTIRE state object rather than just refined_pitch, because synthesis
    is its whole job.

    WHY THIS NODE EXPLICITLY HANDLES PARTIAL FAILURE:
    Per the retry-then-degrade policy, any of the 7 upstream results could
    be a failure marker. _format_agent_result() converts each one into
    either real analysis text or an explicit "UNAVAILABLE" block, and the
    system prompt instructs Claude to acknowledge gaps rather than silently
    reasoning as if it had complete information. This is what makes
    "partial failure" a handled case throughout the system, not just a
    state.py concept that gets ignored by the time it matters.
    """
    refined_text = state["refined_pitch"].get("refined_pitch", state["raw_pitch"])

    personas = state.get("user_personas", [])
    personas_block = "\n".join(
        _format_agent_result(f"User Persona — {p.get('persona_name', 'Unknown')}", p)
        for p in personas
    ) if personas else "### User Personas\nUNAVAILABLE — no persona data collected."

    user_content = "\n\n".join([
        f"### Refined Pitch\n{refined_text}",
        _format_agent_result("Market Intelligence", state.get("market_analysis")),
        _format_agent_result("Problem Validation", state.get("problem_validation")),
        _format_agent_result("Technical Feasibility", state.get("technical_feasibility")),
        _format_agent_result("Accelerator Fit", state.get("accelerator_fit")),
        personas_block,
        _format_agent_result("Devil's Advocate", state.get("devils_advocate")),
    ])

    result = _call_claude_with_retry(
        agent_key="vc_final_call",
        system_prompt=VC_FINAL_CALL_SYSTEM_PROMPT,
        user_content=user_content,
    )
    return {"vc_verdict": result, "stage": "complete"}
