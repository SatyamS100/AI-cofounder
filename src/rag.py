"""
rag.py
======
Sets up TWO separate ChromaDB collections that serve different purposes.
Conflating them would be a design mistake, so this file is deliberately
split into two clearly-separated halves:

  1. ACCELERATOR KNOWLEDGE (static, seeded once)
     Real-world-derived notes on what YC / Sequoia / a16z actually look for.
     Read-only at runtime. Queried by the `accelerator_fit` agent so it
     reasons from grounded text instead of hallucinating "what YC wants."

  2. PITCH ANALYSIS CACHE (dynamic, grows with every run)
     Every time this app fully analyzes a pitch, the refined pitch + its
     market_analysis result gets embedded and stored here. Before
     `market_intelligence` spends a web_search call on a NEW pitch, it
     first checks: "have we already researched a semantically similar
     market recently?" If yes (and the cached result isn't stale), reuse
     it instead of re-searching.

WHY THIS IS SEMANTIC CACHING, NOT JUST MEMOIZATION:
A plain dict cache (`cache[pitch_text] = result`) only hits on an EXACT
string match. Two founders describing the same underlying market in
different words — "hostel food delivery app" vs. "campus meal ordering
platform" — would never collide in a dict, but they SHOULD share a market
analysis. Embedding similarity catches this; exact-match caching can't.

WHY THIS ISN'T "NEVER SEARCH AGAIN":
Market data goes stale — funding rounds close, competitors launch, CAGR
estimates get revised. So a cache hit is gated by TWO conditions, not one:
  (a) similarity: is the cached pitch close enough in embedding space?
  (b) freshness: was that cached analysis run recently enough to still trust?
Both must pass, or we fall back to a live web_search.
"""

import time
import uuid
import json
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: a free, local sentence-transformers model (384-dim
# vectors). No API key, no per-call cost, runs on CPU fast enough for a
# single-user demo app. This is the standard "good enough" default for
# semantic similarity tasks that don't need state-of-the-art retrieval
# quality — we're comparing startup-pitch summaries, not doing legal
# document discovery.
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Persisted to disk so the cache (and the seeded knowledge) survives
# between Streamlit runs — re-seeding or re-embedding every time the app
# restarts would be wasteful and would defeat the point of caching.
_client = chromadb.PersistentClient(path="./chroma_db")


# ─────────────────────────────────────────────────────────────────────────
# PART 1 — STATIC ACCELERATOR KNOWLEDGE
# ─────────────────────────────────────────────────────────────────────────

ACCELERATOR_SEED_DOCS = [
    # ── Y COMBINATOR: FOUNDERS ───────────────────────────────────────────
    {"id": "yc_founder_market_fit",
     "text": "Y Combinator consistently emphasizes founder-market fit: founders who deeply understand the problem because they lived it, not because they researched a trend. Partners look for evidence the founder has unique insight or unfair access that competitors lack."},
    {"id": "yc_founder_obsession",
     "text": "YC partners repeatedly say the best founders are obsessed with the problem, not the startup. Founders who talk more about the customer pain than about their own product tend to build better companies than those who lead with the solution."},
    {"id": "yc_cofounder_importance",
     "text": "YC strongly prefers teams with at least two cofounders. Solo founders are accepted but flagged as a risk factor — a single point of failure for motivation, skills, and decision-making. The best cofounder pairs are friends with complementary skills who have worked together before."},
    {"id": "yc_technical_cofounder",
     "text": "For software startups, YC almost always wants at least one technical cofounder who can build the product. Outsourced development is treated as a warning sign because it implies the team can't iterate quickly and is dependent on an outside party for their core product."},
    {"id": "yc_resilience",
     "text": "YC partners look for evidence of founder resilience: have they overcome adversity before? Have they shipped something despite constraints? Startups hit multiple near-death moments and partners want confidence the founders won't quit when it gets hard."},
    {"id": "yc_execution_speed",
     "text": "A key YC signal is how fast founders ship. Partners often ask 'how long did it take you to build your first version?' A team that built an MVP in two weeks signals different execution speed than one that spent eight months planning before writing code."},
    {"id": "yc_founder_communication",
     "text": "YC values founders who communicate with exceptional clarity. If a founder cannot write a crisp application essay, YC partners worry they also can't write a crisp cold email to a potential customer, a clear job description for a hire, or a tight pitch to their next investor."},
    {"id": "yc_determination",
     "text": "Paul Graham's writing on YC criteria consistently places determination above intelligence as a founder trait. A highly determined founder who is smart enough will figure things out; a brilliant but uncommitted founder will quit when the first hard problem hits."},
    {"id": "yc_previous_startup_credit",
     "text": "Founders who have previously started a company — even one that failed — get significant credit at YC because they have already experienced the emotional rollercoaster of startup life and are less likely to be surprised or broken by it."},
    {"id": "yc_domain_expertise",
     "text": "YC gives extra credit to founders who have spent years in the industry they are now disrupting. A founder who worked in hospital billing for five years before starting a healthcare billing startup is far more credible than someone who read about the problem and decided to enter."},

    # ── Y COMBINATOR: MARKET ─────────────────────────────────────────────
    {"id": "yc_market_size",
     "text": "YC favors markets that are either already large or growing extremely fast. A common YC heuristic is preferring a small market growing 100% year over year over a large stagnant one, because rapid growth signals a real shift in user behavior worth riding early."},
    {"id": "yc_market_timing",
     "text": "YC partners frequently ask 'why is now the right time for this?' A startup in a market that has existed for 20 years with no major new enabler faces a much harder question than a startup riding a platform shift — new regulation, new API, new hardware — that just became possible."},
    {"id": "yc_large_market_required",
     "text": "YC's standard guidance is that the company must eventually be able to reach $100M+ in annual revenue to be worth the risk of venture funding. Ideas that are structurally capped at a small market size are passed on even if the team is excellent."},
    {"id": "yc_monopoly_thinking",
     "text": "YC partners, influenced by Peter Thiel's thinking, look for startups that have a plan to dominate a small market first rather than competing for share in a large crowded one. Winning 70% of a niche is more valuable at seed than 2% of an enormous market."},
    {"id": "yc_market_pull",
     "text": "The strongest YC applications show evidence of market pull: people already emailing the founders asking for the product before it is built, waitlist signups without advertising, or customers who found them through word of mouth. Pull is more convincing than any projection."},
    {"id": "yc_b2b_vs_b2c",
     "text": "YC funds both B2B and B2C companies, but the evaluation differs. B2B startups are expected to show customer conversations and letters of intent before funding. B2C startups are expected to show early user engagement data — even from a tiny sample — rather than projections."},
    {"id": "yc_international_markets",
     "text": "YC funds companies from all geographies, but expects founders outside the US to have thought clearly about whether they are building for a local market first or going global from day one. The go-to-market plan must match the geography of the actual product."},

    # ── Y COMBINATOR: PRODUCT & IDEA ─────────────────────────────────────
    {"id": "yc_simple_explanation",
     "text": "A recurring YC partner note: if a founder cannot explain the startup in one or two plain sentences a non-technical person understands immediately, the idea is usually not focused enough yet. Clarity of explanation is treated as a proxy for clarity of thinking."},
    {"id": "yc_do_things_that_dont_scale",
     "text": "YC's famous 'do things that don't scale' principle means early-stage startups should manually deliver value to a small number of customers rather than building automated systems for millions. Airbnb photographed listings themselves. Stripe manually signed up the first merchants. This is how you learn what actually matters."},
    {"id": "yc_product_market_fit_feeling",
     "text": "YC partner Marc Andreessen described PMF as when users are disappointed if the product is taken away. YC looks for evidence that early users have this visceral dependency — not just that they tried the product, but that they would miss it."},
    {"id": "yc_mvp_philosophy",
     "text": "YC's approach to MVPs is to build the smallest possible thing that delivers real value to a real customer. Not a demo, not a prototype — something a customer actually uses for a real purpose even if everything surrounding it is manual. Stripe's first version was two lines of JavaScript."},
    {"id": "yc_avoid_pivoting_too_early",
     "text": "YC advises founders not to pivot at the first sign of difficulty. Most successful YC companies went through a period where growth was flat or slow before finding the right position. Premature pivoting abandons the learning before it has had time to compound."},
    {"id": "yc_organic_growth",
     "text": "YC evaluates early growth carefully. Organic growth — from word of mouth, SEO, or referrals — is seen as signal of genuine product value. Paid growth in very early stages masks whether the product has real pull and delays the discovery of product-market fit."},
    {"id": "yc_user_interviews",
     "text": "YC's standard advice to any pre-product founder is to do 100 user interviews before writing code. The interviews should be about the customer's existing behavior and pain, not about the founder's proposed solution. 'Would you use X?' is a bad interview question; 'How do you currently do Y?' is a good one."},
    {"id": "yc_competition_response",
     "text": "YC partners expect founders to know their competitors deeply and have a specific, believable explanation for why their company will win. 'We'll do it better' is not an answer. 'We have a distribution advantage / a technical moat / we're starting in a niche competitors can't profitably serve' is an answer."},
    {"id": "yc_idea_maze",
     "text": "YC partner Balaji Srinivasan introduced the 'idea maze' concept: the best founders have spent so much time thinking about their space that they have already mentally navigated dozens of wrong turns and know why competitors failed. A founder who has traveled the idea maze is more fundable than one who just arrived at the entrance."},

    # ── Y COMBINATOR: FUNDRAISING & BATCH ────────────────────────────────
    {"id": "yc_batch_network",
     "text": "YC's primary value beyond capital is the network: access to thousands of YC alumni who have solved the same problems, investors who specifically source from YC Demo Day, and enterprise customers who buy from YC companies preferentially. The batch network compounds over time."},
    {"id": "yc_demo_day",
     "text": "YC Demo Day is a high-leverage fundraising moment — 200+ investors in one room, all pre-sold on the quality filter that being a YC company represents. Companies that have strong metrics at Demo Day routinely raise in days what might take months through cold outreach."},
    {"id": "yc_safe_notes",
     "text": "YC standardized the SAFE (Simple Agreement for Future Equity) as an alternative to convertible notes. SAFEs are now an industry standard because they are faster to close, have no interest rate or maturity date, and are founder-friendly — they were designed by YC to simplify early fundraising."},
    {"id": "yc_application_advice",
     "text": "YC application advice from alumni: be specific about what you've already built and what early users say, not what you plan to build. YC reads thousands of applications and partners can immediately tell the difference between a founder who has talked to customers and one who is hypothesizing."},
    {"id": "yc_post_yc_fundraising",
     "text": "YC companies that come out of the batch with strong metrics — typically $10K-$30K MRR and 10-20% weekly growth — routinely raise seed rounds at $10-20M valuations from top-tier VCs within weeks of Demo Day."},

    # ── A16Z: THESIS & PHILOSOPHY ─────────────────────────────────────────
    {"id": "a16z_thesis_software",
     "text": "a16z's public investment thesis repeatedly stresses backing software that becomes mission-critical infrastructure for its users rather than a nice-to-have tool, since mission-critical products show far lower churn and higher willingness to pay as the company scales."},
    {"id": "a16z_timing",
     "text": "a16z partners often frame the best opportunities as 'when this becomes possible for the first time' moments, driven by a platform shift (new model capability, new distribution channel, new regulation) that didn't exist 18-24 months earlier."},
    {"id": "a16z_software_eating_world",
     "text": "Marc Andreessen's 'software is eating the world' thesis holds that every large industry will eventually be restructured by software companies. a16z backs founders who are applying software to large industries — healthcare, finance, defense, real estate — that have been historically resistant to technology."},
    {"id": "a16z_founder_ceo",
     "text": "a16z has a strong stated preference for founder-CEOs over professional CEOs brought in to run the company. They argue that founders with product vision are harder to find than operational skill, and that founders who stay CEO build more durable companies than those who get replaced."},
    {"id": "a16z_network_effects",
     "text": "a16z specifically looks for businesses with network effects: the more users, the more valuable the product for each user. Marketplaces, communication tools, data businesses, and platforms are categories where network effects create winner-take-most dynamics that justify large valuations."},
    {"id": "a16z_crypto_thesis",
     "text": "a16z's crypto fund thesis is that decentralized protocols can create open, composable financial and ownership infrastructure. They back companies building core protocol infrastructure, not just applications on top of existing chains."},
    {"id": "a16z_bio_thesis",
     "text": "a16z Bio funds companies at the intersection of biology and software — computational biology, AI drug discovery, and synthetic biology. The thesis is that software is now the limiting factor in biology research, the same way it was the limiting factor in finance 30 years ago."},
    {"id": "a16z_american_dynamism",
     "text": "a16z's American Dynamism fund backs companies serving national security, defense, aerospace, manufacturing, and public safety — sectors where US strategic advantage depends on having strong private sector partners and where traditional VCs have historically avoided investing."},
    {"id": "a16z_cultural_capital",
     "text": "a16z explicitly tries to build what they call 'cultural capital' — a media presence, podcast network, and public intellectual community around technology — because they believe the best founders track which investors publish the most useful thinking, not just which ones have the most capital."},
    {"id": "a16z_series_a_signals",
     "text": "a16z's Series A investment signals include: clear evidence of PMF in at least one segment, a repeatable customer acquisition motion, an understanding of the unit economics even if they're not yet positive, and a founder who can articulate the path from current scale to a $1B+ outcome."},

    # ── A16Z: MARKET VIEWS ───────────────────────────────────────────────
    {"id": "a16z_ai_thesis",
     "text": "a16z's AI thesis as of their public writing is that foundation models are infrastructure, and the real value will accrue to application-layer companies that build specific workflows on top of models and own the customer relationship. Pure model providers face commoditization pressure."},
    {"id": "a16z_fintech_thesis",
     "text": "a16z fintech investing focuses on companies rebuilding financial services infrastructure: banking-as-a-service, embedded finance, payments infrastructure, and credit underwriting using alternative data. The thesis is that every large company will eventually have financial products embedded in their core offering."},
    {"id": "a16z_healthcare_thesis",
     "text": "a16z's healthcare investing focuses on three areas: companies using software to reduce healthcare costs at scale, companies giving patients more control over their own data and care, and companies using AI to accelerate drug discovery. They avoid pure services companies that can't scale software margins."},
    {"id": "a16z_consumer_thesis",
     "text": "a16z's consumer investing looks for products that create new categories of behavior rather than incrementally improving an existing behavior. Instagram didn't improve how people used Flickr; it created a new behavior pattern. TikTok didn't improve YouTube; it created a different relationship with video."},
    {"id": "a16z_infrastructure_thesis",
     "text": "a16z invests heavily in developer tools and infrastructure because developer-first products have proven to be some of the most durable businesses — developers adopt them bottoms-up, expand usage across their teams, and eventually become enterprise contracts. Stripe, Twilio, and Snowflake all followed this pattern."},

    # ── SEQUOIA: PRINCIPLES ──────────────────────────────────────────────
    {"id": "sequoia_pmf_signals",
     "text": "Sequoia's published guidance on product-market fit highlights organic, unprompted user pull as the clearest signal: users asking for the product before it is marketed to them, high week-over-week retention in a cohort, and word-of-mouth growth with no paid acquisition."},
    {"id": "sequoia_distribution",
     "text": "Sequoia frequently notes that differentiated distribution often matters more than a differentiated product in crowded markets — a slightly-better product with a proprietary acquisition channel tends to beat a much-better product fighting for the same expensive paid channels as everyone else."},
    {"id": "sequoia_arc_framework",
     "text": "Sequoia's Arc program for early-stage companies uses a framework they call the 'arc of the company' — what is the long-term narrative of value creation? Sequoia wants to back companies whose 10-year story is obvious and large, even if the first product is small."},
    {"id": "sequoia_market_leadership",
     "text": "Sequoia looks for companies positioned to be the category leader. They prefer not to back the 4th player in a market. If there is already a clear leader, Sequoia wants to understand specifically how the new entrant will displace them rather than coexist with them."},
    {"id": "sequoia_team_density",
     "text": "Sequoia's principle of 'team density' holds that a small team of exceptional people outperforms a large team of average people at every stage. They look for founders who hire people who are better than themselves in their functional area, not people they are comfortable managing."},
    {"id": "sequoia_company_building",
     "text": "Sequoia describes their role as company-building partners, not just capital providers. They have dedicated teams for recruiting, marketing, legal, and sales strategy — and they expect portfolio companies to actively use these resources, not just take the check."},
    {"id": "sequoia_global_scope",
     "text": "Sequoia invests across the US, India, China, Southeast Asia, and Europe through separate regional funds. They look for founders with global ambition even when starting locally, because the biggest outcomes they have backed — Google, Alibaba, WhatsApp — all had global reach."},
    {"id": "sequoia_enduring_company",
     "text": "Sequoia's stated goal is to back companies that will endure for decades, not just get acquired or go public in five years. They ask founders to articulate what the company looks like in 20 years and use that answer to evaluate whether the founder has genuine long-term conviction or is optimizing for a quick exit."},
    {"id": "sequoia_10x_better",
     "text": "Sequoia applies a 10x test to products: is this product 10x better than the alternative on at least one dimension that customers care deeply about? Incrementally better products rarely create category-defining companies. 10x products do."},
    {"id": "sequoia_entry_point",
     "text": "Sequoia looks for the right 'entry point' into a large market — a specific, underserved segment where the startup can win completely before expanding. Winning 100% of a tiny segment is more fundable than competing for 1% of an enormous one."},

    # ── SEQUOIA: SECTOR VIEWS ───────────────────────────────────────────
    {"id": "sequoia_saas_metrics",
     "text": "For SaaS companies, Sequoia expects founders to know their NRR (Net Revenue Retention), CAC (Customer Acquisition Cost), LTV (Lifetime Value), and payback period cold. An NRR above 120% — meaning existing customers spend 20% more each year without new customers — is a signal of exceptional product stickiness."},
    {"id": "sequoia_marketplace_metrics",
     "text": "Sequoia evaluates marketplace companies on take rate, GMV growth, supplier and buyer NPS, and leakage (transactions that start on-platform and complete off-platform). High leakage is a fatal signal that the platform is not providing enough value to justify its take."},
    {"id": "sequoia_deep_tech",
     "text": "Sequoia's deep tech investments (robotics, chips, biotech) are evaluated on a different timeline than software. They expect longer development cycles but also higher defensibility — a company with a genuine hardware or biology moat faces less competition than a software-only company."},

    # ── FRAMEWORKS: PRODUCT-MARKET FIT ───────────────────────────────────
    {"id": "framework_tam_sam_som",
     "text": "Standard market-sizing framework: TAM (Total Addressable Market) is total global demand for the category; SAM (Serviceable Addressable Market) is the portion reachable given the company's business model and geography; SOM (Serviceable Obtainable Market) is the realistic share capturable in the near term given competition and execution constraints."},
    {"id": "framework_go_to_market",
     "text": "A credible go-to-market plan names a SPECIFIC initial wedge customer segment rather than 'everyone,' identifies the cheapest channel to reach that exact segment, and has a believable explanation for why that channel won't be saturated or prohibitively expensive within 12 months."},
    {"id": "framework_pmf_40_percent",
     "text": "Sean Ellis's PMF benchmark: survey your users with 'How would you feel if you could no longer use this product?' If 40% or more say 'very disappointed,' you likely have product-market fit. Below 40%, you don't. This benchmark was validated across hundreds of early-stage companies."},
    {"id": "framework_retention_cohorts",
     "text": "Cohort retention analysis is the most honest way to evaluate PMF. Plot the percentage of users who signed up in a given week still active 1, 4, 8, 12, 26 weeks later. A retention curve that flattens rather than going to zero — even at 20% — is a PMF signal. A curve that reaches zero has no core users."},
    {"id": "framework_nps",
     "text": "Net Promoter Score (NPS) measures the percentage of users who would recommend the product (Promoters, score 9-10) minus the percentage who would not (Detractors, score 0-6). Consumer companies with NPS above 50 are exceptional. B2B companies with NPS above 30 are strong. NPS is a leading indicator of organic growth."},
    {"id": "framework_jtbd",
     "text": "The 'Jobs To Be Done' framework (Clay Christensen) says customers don't buy products — they hire products to do a job they need done. Understanding the real job (not the feature request) leads to better product decisions. Customers who 'hired' the milkshake for their commute needed something different from those who 'hired' it for a treat after dinner."},
    {"id": "framework_hook_model",
     "text": "Nir Eyal's Hook Model describes habit-forming products as four-stage cycles: Trigger (internal or external cue to use) → Action (behavior, motivated by reward expectation) → Variable Reward (unpredictable but satisfying outcome) → Investment (user contributes something that makes the next trigger more likely). Consumer apps with strong hooks have far higher DAU/MAU ratios than those without."},

    # ── FRAMEWORKS: BUSINESS MODEL ───────────────────────────────────────
    {"id": "framework_unit_economics",
     "text": "Unit economics is the profitability of the business at the level of one transaction or one customer. The key metrics: LTV (how much revenue one customer generates over their lifetime), CAC (how much it costs to acquire that customer), and payback period (how many months until CAC is recovered). LTV/CAC above 3x is the standard SaaS benchmark for a healthy business."},
    {"id": "framework_saas_revenue_model",
     "text": "SaaS revenue models are valued on ARR (Annual Recurring Revenue) multiples. Early-stage SaaS companies with strong growth (100%+ YoY) and high NRR (120%+) command 10-20x ARR multiples in fundraising. The key levers: expansion revenue from existing customers is more valuable than new customer acquisition because it has no CAC."},
    {"id": "framework_marketplace_model",
     "text": "Two-sided marketplace businesses require solving the chicken-and-egg problem: suppliers won't join without demand, buyers won't join without supply. The standard approach is to subsidize one side (usually supply) until there is enough supply to attract buyers organically. eBay subsidized sellers; Uber subsidized drivers; Airbnb subsidized hosts."},
    {"id": "framework_freemium",
     "text": "Freemium works when: (1) the product delivers genuine value in the free tier so users become dependent on it, (2) the paid tier adds features that specific high-value users need enough to pay for, and (3) the conversion rate from free to paid is high enough that the free user infrastructure costs are covered by paying users. Dropbox and Slack are canonical examples."},
    {"id": "framework_enterprise_sales",
     "text": "Enterprise sales cycle characteristics: average deal size above $50K ARR typically requires a dedicated sales rep; above $250K requires an enterprise sales organization with SDRs, AEs, SEs, and legal review. The rule of thumb is that 1 enterprise sales rep should generate $1M-$2M ARR annually in a healthy enterprise software company."},
    {"id": "framework_bottom_up_gtm",
     "text": "Bottom-up go-to-market (product-led growth) means individual users adopt the product for free or cheaply, use it at work, and pull the product into their company's purchasing process. Slack, Figma, and Notion all entered enterprises this way. It is cheaper than top-down sales but requires a product experience that is compelling enough for individual adoption without IT involvement."},
    {"id": "framework_top_down_gtm",
     "text": "Top-down go-to-market means selling to executives (CTO, CIO, CFO) who then mandate product use across the organization. It produces larger contracts and faster rollout but requires a significant sales team, longer sales cycles, and more work on compliance/security before deals close. Workday, ServiceNow, and Salesforce built primarily top-down."},

    # ── FRAMEWORKS: COMPETITIVE MOATS ────────────────────────────────────
    {"id": "framework_moats_overview",
     "text": "Competitive moats are durable advantages that prevent competitors from eroding a company's position. The main categories: network effects (value grows with users), switching costs (painful to leave), economies of scale (unit costs fall with volume), proprietary data (unique dataset competitors cannot replicate), and brand (trust accumulated over time)."},
    {"id": "framework_network_effects_types",
     "text": "Network effects come in several forms: direct (each new user makes the product more valuable for all existing users — telephone, WhatsApp), indirect (more users on one side attract more supply on the other — Uber, Airbnb), and data network effects (more users generate more data that trains better models that attract more users — Google Search, TikTok algorithm)."},
    {"id": "framework_switching_costs",
     "text": "Switching costs are the friction a customer faces when moving to a competitor. High-switching-cost products include: anything with significant data migration complexity, anything embedded deeply in workflow, anything that has trained a large team of users, and anything with strong integration dependencies. Enterprise software has high switching costs; consumer apps often do not."},
    {"id": "framework_data_moat",
     "text": "A data moat exists when a company has accumulated a dataset that: (1) competitors cannot easily replicate because it requires real user activity over time, (2) produces meaningfully better outputs (recommendations, models, predictions) than a smaller dataset would, and (3) gets better as more users generate more data. Classic examples: Google's search click data, Amazon's purchase history, LinkedIn's professional graph."},
    {"id": "framework_brand_moat",
     "text": "Brand moats develop when customers trust a product so deeply that they won't consider switching even when a cheaper or objectively comparable alternative exists. Brand moats are most powerful in categories where trust is the primary purchase criterion: financial services, healthcare, food safety. Building a brand moat typically takes 5-10 years of consistent customer experience."},

    # ── FRAMEWORKS: FUNDRAISING ──────────────────────────────────────────
    {"id": "framework_seed_round",
     "text": "Seed round characteristics: typically $500K-$3M, from angel investors or seed funds, valuation $5-20M pre-money, used for building initial product and finding first customers. The key milestone that seed capital is meant to reach is proof of PMF or strong evidence that a specific go-to-market approach works."},
    {"id": "framework_series_a",
     "text": "Series A characteristics: typically $5-15M, from institutional VCs, valuation $15-60M pre-money. The key question Series A investors ask is: 'Is there a repeatable, scalable way to acquire customers at a cost that makes sense relative to their lifetime value?' Pre-Series-A companies should have clear evidence of this repeatability."},
    {"id": "framework_series_b",
     "text": "Series B characteristics: typically $20-50M+, valuation $60-200M+ pre-money. By Series B, investors expect the company to have proven the go-to-market motion, understand which customer segments are most valuable, and have a specific plan for how Series B capital will accelerate a mechanism that is already working."},
    {"id": "framework_valuation_methods",
     "text": "Early-stage startup valuation methods: (1) Comparable transactions — what have similar-stage companies raised at recently? (2) Revenue multiples — ARR × industry multiple, typically 10-20x for high-growth SaaS. (3) Discounted cash flow — rarely used at early stages because there are no reliable cash flows to discount. Seed-stage valuations are most heavily influenced by team strength and market size, since there is rarely product revenue to use as a baseline."},
    {"id": "framework_due_diligence",
     "text": "Investor due diligence for a Series A typically covers: customer reference calls (3-5 customers), technical review of the product and codebase architecture, financial model review, competitive landscape analysis, legal review of cap table and contracts, and founder background checks. The process takes 4-8 weeks on average."},

    # ── FRAMEWORKS: EXECUTION & OPERATIONS ──────────────────────────────
    {"id": "framework_okrs",
     "text": "OKRs (Objectives and Key Results) originated at Intel and were popularized by Google. Objectives are qualitative goals; Key Results are quantitative measures of progress toward each objective. The standard cadence is quarterly OKRs with weekly check-ins. OKRs work best when the company has enough clarity about what it is trying to achieve that it can commit to measurable outcomes in advance."},
    {"id": "framework_board_management",
     "text": "Effective startup board management: send board materials 5-7 days before the meeting, include a 'state of the company' memo with honest assessment of what's working and what isn't, use board time for strategic discussion rather than reporting, and make specific asks of board members rather than general updates. The best boards are partnerships, not oversight committees."},
    {"id": "framework_hiring_first_10",
     "text": "The first 10 hires at a startup are disproportionately important because they set the cultural and quality bar for every subsequent hire. Founders should personally interview every candidate until the company reaches 30-50 people. Hiring someone 'good enough' early creates a quality ceiling that is very difficult to raise later."},
    {"id": "framework_firing_fast",
     "text": "Nearly every startup founder reports that their biggest hiring mistake was waiting too long to fire a bad fit. The warning signs are usually present in the first 30-60 days but founders wait 6-12 months, during which the underperformer affects team morale, occupies a role that a better person could fill, and delays progress on critical work."},
    {"id": "framework_runway_management",
     "text": "Startup runway is the number of months of cash remaining at the current burn rate. The standard rule is to start fundraising when you have 6-9 months of runway remaining, because fundraising takes longer than expected and you don't want to negotiate from desperation. Burn rate should be actively managed: every increase in monthly expense should be tied to a specific expected return."},

    # ── FRAMEWORKS: PRODUCT DEVELOPMENT ─────────────────────────────────
    {"id": "framework_agile_sprints",
     "text": "Agile development uses short development cycles (sprints, typically 2 weeks) with daily standups, sprint reviews, and retrospectives. The goal is to ship working software frequently and respond to new information quickly rather than building to a fixed specification that may be wrong. Most successful software startups operate on some form of agile methodology."},
    {"id": "framework_technical_debt",
     "text": "Technical debt is the accumulated cost of shortcuts taken to ship faster. All startups accumulate technical debt intentionally (speed matters early) but must consciously schedule debt repayment before it becomes a ceiling on engineering velocity. The classic failure mode is letting debt accumulate until new features take 10x longer to build than they should."},
    {"id": "framework_data_driven_product",
     "text": "Data-driven product development means using quantitative usage data (funnel drop-off, session duration, feature adoption rates, A/B test results) alongside qualitative user research to make product decisions. Neither alone is sufficient: data tells you what is happening, qualitative research tells you why."},
    {"id": "framework_api_first",
     "text": "API-first product design means building the product as a service that other products can consume before building the UI layer on top of that service. Stripe, Twilio, and Plaid all took this approach. It forces engineers to build clean abstractions, makes the product easier to integrate with partners, and often produces faster-loading frontends."},
    {"id": "framework_zero_to_one",
     "text": "Peter Thiel's Zero to One thesis: going from zero to one (creating something that didn't exist) is more valuable and more defensible than going from one to N (copying something that already exists). Thiel argues that most competitive business thinking focuses on 1 to N improvements, which produces commoditized outcomes, while the biggest companies are always doing something fundamentally new."},

    # ── STARTUP FAILURE MODES ────────────────────────────────────────────
    {"id": "common_rejection_reasons",
     "text": "Common reasons top accelerators pass on applications: founders solving a problem they don't personally experience, market size claims that don't survive a bottom-up sanity check, no evidence of any user demand prior to the application, and teams missing the technical capability to build their own MVP."},
    {"id": "failure_no_market_need",
     "text": "CB Insights' startup post-mortem analysis consistently shows 'no market need' as the #1 reason startups fail, cited in ~42% of post-mortems. Founders built something people didn't want badly enough to pay for or change behavior for. This is why customer discovery before product development is the single most important early-stage activity."},
    {"id": "failure_ran_out_of_cash",
     "text": "Running out of cash (#2 cause of startup failure per CB Insights, ~29%) is almost always a symptom of a deeper problem — usually no PMF, leading to slow growth, leading to inability to raise the next round. The rare case where a company with genuine PMF runs out of cash is usually a fundraising timing problem, not a fundamental product problem."},
    {"id": "failure_wrong_team",
     "text": "Team problems (#3 cause, ~23%) include: cofounders who split acrimoniously over equity or direction, critical skill gaps (no technical cofounder for a software company), founders who can't adapt their role as the company grows, and cultural dysfunction that causes key hires to leave."},
    {"id": "failure_out_competed",
     "text": "Being outcompeted (#4, ~19%) is more often a symptom of insufficient differentiation than of a bad product. Companies that get crushed by competition were usually building something a well-funded incumbent could copy without much difficulty. The question to ask before starting: 'If Google/Amazon/Meta built this in 18 months, would our company survive?'"},
    {"id": "failure_pricing",
     "text": "Pricing problems — either pricing too low (undervaluing the product, leaving revenue on the table, attracting the wrong customers) or too high (missing the entry-level market entirely) — cause startup failure more often than founders expect. The most common mistake is pricing on cost rather than on the value delivered to the customer."},
    {"id": "failure_pivot_too_late",
     "text": "Pivoting too late is the mirror failure mode of pivoting too early. Companies that had strong signals their original direction wasn't working but kept executing on it anyway until they ran out of runway. Instagram pivoted from a location app to photo sharing; Slack pivoted from a gaming company to messaging; YouTube pivoted from a dating video site to general video. All of those pivots came early enough that there was runway to find the right direction."},
    {"id": "failure_regulatory",
     "text": "Regulatory failure is especially common in healthcare, fintech, legal tech, and any company touching personal data. Founders often underestimate the time and cost of regulatory compliance and overestimate the pace at which regulators will approve novel products. The safest approach is to hire regulatory counsel before entering a regulated market, not after the product is already built."},

    # ── INDIA / EMERGING MARKET SPECIFIC ────────────────────────────────
    {"id": "india_startup_ecosystem",
     "text": "India's startup ecosystem produced 100+ unicorns by 2024. The most successful Indian startups address: financial inclusion (Paytm, PhonePe, Razorpay), commerce (Flipkart, Meesho, Zepto), edtech (BYJU's, Unacademy), logistics (Delhivery, Shiprocket), and B2B SaaS for global markets (Freshworks, Zoho). The common thread is large Indian market problems solved with mobile-first products."},
    {"id": "india_b2b_saas",
     "text": "India has become a major exporter of B2B SaaS to US and European markets. The model — build in India, sell globally — works because engineering cost structures allow Indian SaaS companies to undercut US competitors by 30-50% while maintaining comparable product quality. Freshworks, BrowserStack, Chargebee, and Postman all followed this model."},
    {"id": "india_consumer_market",
     "text": "India's consumer market is characterized by extreme price sensitivity, high smartphone penetration with relatively low data costs (post-Jio), rapid urbanization creating new consumer needs, and a young demographic skewed heavily toward mobile-first behavior. Products that work for Tier 2 and Tier 3 cities alongside metros have 10x the addressable market of metro-only products."},
    {"id": "india_fintech_opportunity",
     "text": "India's fintech opportunity is driven by UPI (Unified Payments Interface), which processed over 10 billion transactions per month by 2024 and created a real-time payment infrastructure that most developed markets don't have. The layer above UPI — lending, insurance, wealth management, business finance — is still early and large."},
    {"id": "india_iit_founder_signal",
     "text": "IIT and IIM alumni networks are high-signal in Indian startup investing. IIT founder status is used as a quality filter for technical capability, and the alumni network provides early customer access, co-founder sourcing, and referral investment. Indian VCs explicitly weight IIT/IIM credentials in early-stage decisions when there is little product data to go on."},

    # ── METRICS & BENCHMARKS ─────────────────────────────────────────────
    {"id": "benchmark_saas_growth_rates",
     "text": "SaaS growth rate benchmarks by stage: pre-seed/seed companies should target 10-15% weekly growth on a small base; Series A companies should show 100%+ ARR growth year-over-year; Series B companies should be growing at 80%+; late-stage (Series D+) companies can show 40%+ and still be healthy. Companies below these benchmarks at each stage struggle to raise the next round at expected valuations."},
    {"id": "benchmark_gross_margins",
     "text": "Gross margin benchmarks: pure software SaaS companies should target 70-80%+ gross margins at scale; marketplaces typically run 40-60% gross margins; hardware companies often run 40-50%; healthcare services companies often run 20-40%. Gross margin determines how much revenue can flow to operating profit or growth investment as the company scales."},
    {"id": "benchmark_payback_period",
     "text": "CAC payback period benchmarks: under 12 months is excellent for SaaS, 12-24 months is acceptable, above 24 months is a warning sign that either CAC is too high or LTV is too low. Consumer companies with subscription revenue should target under 6-month payback periods because churn is typically higher than enterprise."},
    {"id": "benchmark_magic_number",
     "text": "The SaaS Magic Number measures sales efficiency: (new ARR in a quarter) / (sales and marketing spend in the prior quarter). A Magic Number above 0.75 means the business is efficient at growing ARR. Above 1.5 means the business should aggressively accelerate sales hiring. Below 0.5 is a warning sign that the go-to-market is inefficient."},
    {"id": "benchmark_rule_of_40",
     "text": "The Rule of 40 says a healthy SaaS company's revenue growth rate plus profit margin should equal or exceed 40%. A company growing at 80% with -40% margins is healthy; a company growing at 20% with 25% margins is healthy; a company growing at 30% with -20% margins scores 10 — below the threshold and concerning to late-stage investors."},
    {"id": "benchmark_nrr",
     "text": "Net Revenue Retention (NRR) measures how much revenue from existing customers grows or shrinks over time, including expansion, contraction, and churn. NRR above 130% (best-in-class: Snowflake was 158% at IPO) means existing customers generate enough new revenue that the company would grow even without adding a single new customer. NRR below 100% means churn exceeds expansion."},
    {"id": "benchmark_churn_rates",
     "text": "Monthly churn rate benchmarks: below 1% monthly is excellent for B2B SaaS (12% annual); 1-2% is acceptable (12-24% annual); above 2% is concerning. Consumer subscription churn is typically much higher — 5-7% monthly is common. High churn is the clearest signal that PMF is not achieved because customers who get value don't leave."},
    {"id": "benchmark_seed_metrics",
     "text": "Metrics that make a strong seed-stage company: 10-20 paying customers even at low prices (shows willingness to pay), 3+ customer interviews per week (shows founder commitment to learning), some form of weekly or monthly growth over at least 8 weeks (shows product is improving), and a founder who can articulate what they've learned that has changed their original hypothesis."},

    # ── AI / GENAI SPECIFIC ──────────────────────────────────────────────
    {"id": "genai_startup_thesis",
     "text": "The GenAI startup investment thesis in 2024-2025: the first wave of companies built thin wrappers on foundation models that were quickly replicated by OpenAI/Anthropic building the feature natively. The durable AI companies will own proprietary data, a unique distribution channel, or a workflow so deeply embedded in customer operations that switching costs are high."},
    {"id": "genai_moat_question",
     "text": "The core investor question for any AI startup in 2024+: 'What happens to this company when the foundation model gets 10x better and cheaper?' If the answer is 'we also get 10x better' and the customer relationship stays with the startup, the business is durable. If the answer is 'the foundation model provider bundles our feature,' the business is not."},
    {"id": "genai_data_flywheel",
     "text": "The most defensible GenAI companies are building data flywheels: product usage generates proprietary data, that data trains better models, better models attract more users, more users generate more data. Scale AI, Harvey, and Glean are examples of companies building real data moats because their products generate labeled data that competitors don't have access to."},
    {"id": "genai_enterprise_adoption",
     "text": "Enterprise GenAI adoption patterns as of 2024: large enterprises are spending but cautiously — most have AI pilots rather than full deployments, security and compliance requirements are the primary blockers, and the ROI evidence base is still thin. Startups that can show measurable, auditable productivity improvements with clear security posture close deals faster than those selling on potential."},
    {"id": "genai_vertical_saas",
     "text": "Vertical AI SaaS — AI built specifically for one industry (legal, healthcare, construction, finance) — is one of the more fundable GenAI categories because: domain-specific models outperform general models on domain-specific tasks, switching costs are higher when a product is deeply embedded in industry-specific workflow, and competitors must develop domain expertise to replicate, not just API access."},

    # ── PRODUCT DESIGN & USER EXPERIENCE ────────────────────────────────
    {"id": "design_simplicity",
     "text": "The best-designed startup products have fewer features than competitors, not more. Simplicity is the hardest design achievement because it requires knowing which features to exclude. Every feature added is a decision made for the user; every feature excluded is a decision deferred to the user. The best products make the right decisions, not all the decisions."},
    {"id": "design_onboarding",
     "text": "Onboarding is the highest-leverage part of the user experience because activation rate (% of new users who experience core value) directly determines whether all marketing spend pays off. A product with 40% activation that doubles to 80% effectively doubles the output of every acquisition dollar without changing the acquisition cost."},
    {"id": "design_mobile_first",
     "text": "Mobile-first design means designing for the smallest screen and slowest connection first, then expanding to desktop — not the reverse. Products designed desktop-first often produce awkward, high-friction mobile experiences because they were not constrained by mobile limitations during the critical early design decisions."},

    # ── SECTOR-SPECIFIC NOTES ────────────────────────────────────────────
    {"id": "sector_edtech",
     "text": "EdTech companies face a specific structural challenge: the person paying (parent, employer, institution) is often not the person learning, creating misaligned incentives between engagement metrics and payment. The strongest EdTech companies either sell directly to learners with high enough willingness to pay (coding bootcamps, professional certification) or have strong evidence of measurable outcome improvement that justifies institutional budgets."},
    {"id": "sector_healthtech",
     "text": "Health tech startups face three distinct market segments: (1) direct-to-consumer health and wellness, which operates like any consumer subscription; (2) digital health with clinical claims, which requires clinical evidence and FDA clearance; (3) health system software, which requires hospital procurement processes. Each segment has different timelines, sales cycles, and evidence requirements."},
    {"id": "sector_legaltech",
     "text": "Legal tech startups face adoption challenges because the legal profession is conservative and risk-averse, billing structures (hourly billing) create perverse incentives to resist efficiency tools, and professional responsibility rules create liability concerns around AI-generated legal content. The most successful legal tech companies either sell to in-house legal teams (more price-sensitive than law firms) or serve specific high-volume, repetitive legal workflows where accuracy can be validated."},
    {"id": "sector_proptech",
     "text": "Property technology (proptech) covers a wide range of applications: real estate marketplaces, property management software, construction tech, mortgage technology, and smart building infrastructure. The sector is highly fragmented because real estate is a local industry with enormous geographic variation in regulations, practices, and market dynamics."},
    {"id": "sector_climatetech",
     "text": "Climate tech investing has grown dramatically since 2020, with VCs deploying billions annually into energy storage, grid software, carbon capture, electric transportation, and sustainable agriculture. The best climate tech businesses combine technology innovation with large, addressable markets and have a credible path to cost parity with fossil fuel alternatives without ongoing subsidy."},
    {"id": "sector_agritech",
     "text": "Agritech faces a uniquely challenging customer: farmers are sophisticated buyers who have seen many technology promises fail, operate on thin margins that make upfront investment difficult, and have highly variable needs depending on geography, crop type, and scale. Successful agritech companies often use a distribution-first strategy — partnering with input suppliers, cooperatives, or government agricultural programs rather than selling directly to individual farmers."},
]


def seed_accelerator_knowledge(force_reseed: bool = False):
    """
    Populate the static accelerator_knowledge collection. Idempotent by
    default — if the collection already has documents, we skip re-seeding
    (re-embedding identical static text on every Streamlit rerun would be
    pure wasted compute). Pass force_reseed=True to wipe and rebuild it,
    e.g. after editing ACCELERATOR_SEED_DOCS.
    """
    collection = _client.get_or_create_collection(
        name="accelerator_knowledge",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if force_reseed:
        existing_ids = collection.get(include=[])["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    if collection.count() > 0:
        return collection  # already seeded, nothing to do

    collection.add(
        ids=[doc["id"] for doc in ACCELERATOR_SEED_DOCS],
        documents=[doc["text"] for doc in ACCELERATOR_SEED_DOCS],
        metadatas=[{"source": "accelerator_seed"} for _ in ACCELERATOR_SEED_DOCS],
    )
    return collection


def query_accelerator_knowledge(refined_pitch_text: str, n_results: int = 4) -> list[str]:
    """
    Called by the accelerator_fit agent BEFORE its Claude API call. Returns
    the top-N most relevant grounding snippets as plain strings, ready to be
    inserted into the agent's prompt context.

    WHY RETRIEVAL HAPPENS BEFORE THE LLM CALL (the actual "RAG" part):
    The retrieved text becomes part of the prompt Claude reasons over, so
    the model's judgment is grounded in these specific notes rather than
    whatever it recalls (or invents) from training data about "what YC
    wants." This is the core RAG pattern: retrieve, then generate.
    """
    collection = _client.get_or_create_collection(
        name="accelerator_knowledge",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        seed_accelerator_knowledge()

    results = collection.query(query_texts=[refined_pitch_text], n_results=n_results)
    # results["documents"] is a list of lists (one inner list per query text);
    # we only passed one query, so we take index [0].
    return results["documents"][0] if results["documents"] else []


# ─────────────────────────────────────────────────────────────────────────
# PART 2 — DYNAMIC PITCH ANALYSIS CACHE (the "don't re-search the same
# market twice" feature)
# ─────────────────────────────────────────────────────────────────────────

# Cache hit requires BOTH gates to pass:
SIMILARITY_DISTANCE_THRESHOLD = 0.25   # cosine distance; lower = stricter match
CACHE_FRESHNESS_SECONDS = 7 * 24 * 60 * 60  # 7 days — market data older than
                                             # this is treated as a cache miss


def _cache_collection():
    return _client.get_or_create_collection(
        name="pitch_analysis_cache",
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def find_similar_pitch_analysis(refined_pitch_text: str) -> dict | None:
    """
    Checks whether a semantically similar pitch has already been analyzed
    recently. Returns the cached market_analysis dict on a hit, or None on
    a miss (forcing the caller to fall back to a live web_search).

    GATE 1 — similarity: nearest neighbor's cosine DISTANCE must be below
    SIMILARITY_DISTANCE_THRESHOLD. (ChromaDB's `query` returns distance,
    where 0 = identical embedding; we want SMALL distance = closely related.)

    GATE 2 — freshness: the cached entry's stored timestamp must be within
    CACHE_FRESHNESS_SECONDS of now, otherwise the market data is considered
    too stale to trust even though the pitch matched semantically.
    """
    collection = _cache_collection()
    if collection.count() == 0:
        return None  # cold start — nothing cached yet, guaranteed miss

    results = collection.query(
        query_texts=[refined_pitch_text],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return None

    distance = results["distances"][0][0]
    if distance > SIMILARITY_DISTANCE_THRESHOLD:
        return None  # nearest match isn't actually similar enough — miss

    metadata = results["metadatas"][0][0]
    cached_at = metadata.get("cached_at_unix", 0)
    age_seconds = time.time() - cached_at
    if age_seconds > CACHE_FRESHNESS_SECONDS:
        return None  # similar enough, but the data is too old to trust — miss

    return {
        "market_analysis": json.loads(metadata["market_analysis_json"]),
        "matched_pitch_summary": results["documents"][0][0],
        "similarity_distance": distance,
        "cached_age_days": round(age_seconds / 86400, 1),
    }


def store_pitch_analysis(refined_pitch_text: str, market_analysis: dict) -> None:
    """
    Called after a FRESH market_intelligence run (i.e. one that actually
    used web_search) to add this pitch + its result to the cache for future
    runs. Not called on a cache hit — we don't want to keep re-stamping the
    same cached entry's timestamp forward, since that would let a single
    stale-but-frequently-matched entry live forever.
    """
    collection = _cache_collection()
    collection.add(
        ids=[f"pitch_{uuid.uuid4().hex[:12]}"],
        documents=[refined_pitch_text],
        metadatas=[{
            "market_analysis_json": json.dumps(market_analysis),
            "cached_at_unix": time.time(),
        }],
    )
