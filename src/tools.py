"""
tools.py
========
Web search via Tavily API — completely decoupled from the LLM provider.

ARCHITECTURE CHANGE FROM GEMINI VERSION:
Previously: Gemini's server-side grounding (search + generate in one API call)
Now:        Tavily search (separate call) → inject results → Groq generate

WHY THIS IS ACTUALLY BETTER:
1. No coupling between search provider and LLM provider
2. You can swap either independently (change LLM: edit agents.py;
   change search: edit tools.py — zero overlap)
3. Search results are visible in your code — you can log them, filter
   them, cache them, or transform them before the LLM sees them
4. Tavily returns structured JSON (title, url, content, score) not raw
   HTML — no parsing required
5. Search failures don't take down the LLM call and vice versa

INTERVIEW TALKING POINT:
"Server-side grounding couples your search provider to your LLM provider.
I decoupled them: Tavily handles retrieval, Groq handles generation.
This lets me swap either independently and gives me full observability
into what search results the model is reasoning over."
"""

import os
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient

# Initialize once at module level — same pattern as the LLM client
_tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def web_search(query: str, max_results: int = 5) -> str:
    """
    Executes a Tavily search and returns results as a formatted string
    ready to be injected into an LLM prompt.

    Returns a formatted string rather than a list of dicts because the
    output goes directly into an f-string prompt — no further processing
    needed by the caller.

    max_results=5 matches our previous max_uses=5 on the Gemini tool,
    keeping the search depth equivalent.
    """
    try:
        results = _tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",   # more thorough than "basic"
            include_answer=True,       # Tavily's own AI summary of results
        )

        # Format results as readable text for prompt injection
        formatted = []

        # Tavily's own answer synthesis (if available)
        if results.get("answer"):
            formatted.append(f"SEARCH SUMMARY: {results['answer']}\n")

        # Individual source results
        for i, result in enumerate(results.get("results", []), 1):
            formatted.append(
                f"SOURCE {i}: {result.get('title', 'Unknown')}\n"
                f"URL: {result.get('url', '')}\n"
                f"CONTENT: {result.get('content', '')}\n"
            )

        return "\n".join(formatted) if formatted else "No search results found."

    except Exception as e:
        # Search failure should not crash the agent — return empty string
        # The agent will reason from its own knowledge instead
        return f"Search unavailable ({str(e)[:100]}). Reasoning from training data."


def multi_search(queries: list[str]) -> str:
    """
    Runs multiple targeted searches and combines results.
    Used by market_intelligence which needs separate searches for:
      - market size/TAM
      - competitor funding
      - market timing/trends

    Running 3 focused queries produces better results than one broad query.
    """
    all_results = []
    for query in queries:
        result = web_search(query, max_results=3)
        all_results.append(f"=== SEARCH: {query} ===\n{result}")
    return "\n\n".join(all_results)


# Legacy compatibility — agents.py imported these names from the old tools.py
# Keep them defined so no import errors
WEB_SEARCH_TOOL = None         # No longer a dict — search is now in-code
WEB_SEARCH_TOOL_WITH_LIMIT = None
