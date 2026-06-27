# AI Venture Studio

A multi-agent startup evaluation system. Submit a pitch; 8 specialized
Claude agents analyze it across market, customer, technical, and investor
dimensions in two parallel stages; a final VC agent delivers the final
investment verdict.

Built with **Python · LangGraph · ChromaDB · Groq API (`llama-3.3-70b-versatile`) · Tavily Search**

---

## Architecture

```
START
  → pitch_refiner                                          [sequential]
  → [market_intelligence, problem_validator,
     technical_feasibility]                                [PARALLEL — Stage 1]
  → [accelerator_fit, user_personas, devils_advocate]      [PARALLEL — Stage 2]
  → vc_final_call                                          [sequential]
  → END
```

| Agent | Role |
|---|---|
| Pitch Refiner | Sharpens raw idea → core insight, target customer, one-liner |
| Market Intelligence | Live web search → TAM/SAM/SOM, competitors, CAGR |
| Problem Validator | Reddit/forum evidence → problem severity, willingness to pay |
| Technical Feasibility | CTO lens → buildability score, timeline, team size |
| Accelerator Fit | RAG-grounded → YC/Sequoia/a16z fit score |
| User Personas ×3 | Early adopter, mainstream user, enterprise buyer reactions |
| Devil's Advocate | Adversarial → regulatory risk, incumbent threats, failure modes |
| VC Final Call | Synthesizes all 7 → Pass / Watch / Invest / Lead verdict |

**Key numbers:** 8 agent nodes · 51 structured output fields · 4-verdict taxonomy · 128-document ChromaDB knowledge base · 2 parallel execution stages

---

## What makes this technically interesting

- **Pre-Retrieval Injection** — decoupled Tavily search from the Groq generation to eliminate vendor lock-in and avoid server-side grounding black boxes
- **Semantic pitch caching** — ChromaDB embedding similarity check before web search; reuses prior market analyses for semantically similar pitches
- **RAG-grounded accelerator judgment** — 128 curated YC/Sequoia/a16z documents retrieved before LLM call, not hallucinated
- **Retry-then-degrade policy with Concurrency Control** — uses Token Buckets and Semaphores to safely parallelize 8 agents while strictly adhering to API limits

---

## Project structure

```
ai-venture-studio/
├── .env                   ← GROQ_API_KEY, TAVILY_API_KEY (gitignored)
├── .env.example           ← template (safe to commit)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.py              ← model name, retry policy, agent metadata
├── README.md
├── src/
│   ├── __init__.py
│   ├── state.py           ← LangGraph TypedDict state schema
│   ├── agents.py          ← all 8 agent node functions using Groq API
│   ├── tools.py           ← Tavily web search decoupling
│   ├── rag.py             ← ChromaDB: 128-doc knowledge base + pitch cache
│   └── graph.py           ← StateGraph wiring (the parallel topology)
└── app.py                 ← Streamlit UI with live progress
```

---

## Local setup (without Docker)

```bash
git clone https://github.com/YOUR_USERNAME/ai-venture-studio.git
cd ai-venture-studio

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and set:
# GROQ_API_KEY=gsk_your_groq_key
# TAVILY_API_KEY=tvly-your_tavily_key
```

**Sanity-check graph topology (no API cost):**
```bash
python -m src.graph
```

**Run the app:**
```bash
streamlit run app.py
```
Open `http://localhost:8501`. On first run, ChromaDB seeds itself with the 128-document knowledge base automatically.

---

## Docker (local)

```bash
cp .env.example .env      # add your real ANTHROPIC_API_KEY

docker compose up --build  # first run; subsequent runs: docker compose up
```

Open `http://localhost:8501`

The ChromaDB data persists in a named Docker volume (`chroma_data`) — it survives container restarts and rebuilds. To wipe it: `docker compose down -v`

---

## Deploying to the cloud

### Option A — Railway (easiest, free tier available)

1. Push to GitHub (see below)
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Railway auto-detects the Dockerfile and builds it
4. Add environment variable: `ANTHROPIC_API_KEY` = your real key
5. Railway gives you a public HTTPS URL

Volume persistence on Railway: add a Railway Volume mounted at `/app/chroma_db`

### Option B — Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect repo
3. Runtime: Docker (auto-detected from Dockerfile)
4. Add env var: `ANTHROPIC_API_KEY`
5. For ChromaDB persistence: add a Render Disk mounted at `/app/chroma_db`

### Option C — Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ai-venture-studio
gcloud run deploy ai-venture-studio \
  --image gcr.io/YOUR_PROJECT/ai-venture-studio \
  --platform managed \
  --port 8501 \
  --set-env-vars ANTHROPIC_API_KEY=sk-ant-your-key \
  --allow-unauthenticated
```

Note: Cloud Run is stateless — ChromaDB will reseed on every cold start. For persistent ChromaDB on GCP, mount a Cloud Filestore volume or switch the ChromaDB client to a managed vector DB.

---

## Pushing to GitHub

```bash
# From inside the project directory
git init
git add .
git commit -m "Initial commit: AI Venture Studio"

# Create a new repo on github.com (no README, no .gitignore — you already have both)
git remote add origin https://github.com/YOUR_USERNAME/ai-venture-studio.git
git branch -M main
git push -u origin main
```

Your `.gitignore` already excludes `.env` and `chroma_db/` so your API key and local DB are never committed.

---

## Demo tip

To show the semantic cache working live in an interview:

1. Submit any startup pitch → full run with web search (~25-35s)
2. Submit a slightly different version of the same pitch (different wording, same market)
3. Market Intelligence card shows "⚡ Reused market analysis from a similar pitch analyzed X minutes ago" — zero web search, ~10s total runtime

This visually demonstrates the embedding similarity cache working in real time.
