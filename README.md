# TalentIQ — Intelligent Candidate Discovery & Ranking

Redrob Hackathon submission: AI-powered candidate ranker that intelligently ranks 100K candidates for a Senior AI Engineer role in under 60 seconds on CPU.

## Quick Reproduce

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Requirements:** Python 3.10+ (no external dependencies needed)

**Runtime:** ~47 seconds on 8-core CPU with 16GB RAM

## Architecture

```
candidates.jsonl (100K)
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: Fast Pre-filter           │  Eliminates ~50% candidates lacking
│  (Title + Skills + Career scan)     │  any AI/ML signal
└─────────────────────────────────────┘
    │ ~49K candidates
    ▼
┌─────────────────────────────────────┐
│  Stage 2: Multi-Signal Scoring      │
│  ┌─────────────────────────────┐    │
│  │ Title Relevance      (22%)  │    │  Strong/Moderate/Non-engineering
│  │ Career History        (23%) │    │  Product company + AI roles
│  │ Skill Match + Trust   (18%) │    │  Proficiency × endorsements × duration
│  │ Text Semantic         (12%) │    │  JD concept matching in descriptions
│  │ Behavioral Signals    (10%) │    │  Recency, response rate, GitHub
│  │ Experience Fit         (5%) │    │  5-9 year sweet spot
│  │ Location               (3%) │    │  India Tier-1 cities
│  └─────────────────────────────┘    │
│  + Honeypot Detection               │  Expert+0duration, title mismatch
│  + Fine-grained Differentiators     │  Assessments, saves, notice period
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 3: Sort + Normalize + Output │  Top 100 → submission.csv
└─────────────────────────────────────┘
```

## Key Design Decisions

1. **Title is decisive** — The JD explicitly warns against keyword stuffers. A "Marketing Manager" with 9 AI skills is penalized; an "ML Engineer" with 4 relevant skills is promoted.

2. **Career over skills** — We weight career history (actual AI roles at product companies) higher than self-reported skills, with trust multipliers (endorsements × duration) to validate skill claims.

3. **Behavioral signals as availability filter** — A perfect-on-paper candidate who hasn't logged in for 6 months and has 5% response rate is downweighted 50%.

4. **Honeypot detection** — Identifies impossible profiles (expert proficiency with 0 months usage, non-engineering titles claiming 10+ AI skills with no career evidence).

5. **Zero dependencies** — Pure Python stdlib. No ML libraries, no embeddings model, no network. Runs anywhere Python runs.

## Scoring Details

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Title Relevance | 22% | Is current title an AI/ML engineering role? |
| Career History | 23% | Years in AI roles at product (not consulting) companies |
| Skill Match | 18% | AI/ML skills with trust weighting (endorsements × duration) |
| Text Semantic | 12% | JD-relevant concepts in summary/descriptions |
| Behavioral | 10% | Recency, response rate, GitHub, interview completion |
| Experience Fit | 5% | 5-9 years = optimal per JD |
| Location | 3% | India Tier-1 cities, willingness to relocate |

**Bonus signals:** GitHub activity, platform assessment scores, recruiter saves, notice period

## Files

| File | Purpose |
|------|---------|
| `rank.py` | Main ranking script (single command) |
| `submission.csv` | Final ranked output (100 candidates) |
| `submission_metadata.yaml` | Portal metadata mirror |
| `requirements_ranker.txt` | Dependencies (none needed) |

## Validation

```bash
python validate_submission.py submission.csv
# Output: "Submission is valid."
```

## Results Summary

- **Runtime:** 47.4 seconds (budget: 300s)
- **Top candidates:** Senior ML Engineers, AI Engineers, NLP Engineers at Zomato, Paytm, Razorpay, Flipkart, Netflix, Google, Meta, Apple
- **Experience range:** 5.2–8.9 years (sweet spot per JD)
- **Honeypots in top 100:** 0
- **Score range:** 0.40–0.99 (100 unique scores)

---

## Original Platform (TalentIQ AI)

AI-powered semantic candidate ranking platform. Uses LangGraph 6-agent pipeline for evaluation, BGE embeddings for semantic matching, and Groq for free LLM inference.

### Platform Architecture

```
Frontend (Next.js 15) → Backend (FastAPI) → Neon PostgreSQL + pgvector
                                          → Groq (LLM)
                                          → BGE embeddings (local)
                                          → Cloudinary (file storage)
                                          → Upstash Redis (cache)
```

### Platform Setup

1. Clone and copy env: `cp .env.example .env`
2. Fill in API keys (all free-tier): Groq, Neon, Clerk, Cloudinary, Upstash
3. Run: `docker compose up --build`
4. Seed demo data: `docker exec -it ptojects-backend-1 python -m app.seed`
5. Open http://localhost:3000
