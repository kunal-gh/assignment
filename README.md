# AI Resume Screener

**Live Demo: https://ai-resume-screener-sepia.vercel.app**

> Upload resumes + paste a job description. Get ranked candidates with scores, skill analysis, and AI explanations in seconds.

---

## What It Does

Paste a job description, upload PDF/DOCX resumes, and the system:

1. Extracts text and skills from each resume
2. Scores every candidate 0-100 using a hybrid formula
3. Ranks them with plain-English AI explanations
4. Flags potential bias and hidden gems
5. Exports results to CSV

No signup. No API key needed. Hit **LOAD DEMO** to see it instantly.

---

## Tech Stack

### Running on Vercel right now

| Layer | Technology | Detail |
|-------|-----------|--------|
| Frontend | Next.js 15 + React 19 | App Router, Tailwind CSS, Framer Motion |
| State | Zustand 5 | Client-side state management |
| Charts | Recharts 2 | Bar chart + Radar comparison |
| API | Next.js Route Handler | `/app/api/screen/route.ts` on Vercel Node runtime |
| Scoring | TF-IDF cosine similarity | Lightweight, no ML deps, runs in under 1s |
| Skill matching | IDF-weighted coverage | 200+ skills, rarer skills worth more |
| Hidden gem detection | Score gap analysis | Flags semantic >> keyword candidates |
| Deployment | Vercel | Auto-deploy on push to main |

### Scoring formula

```
hybrid_score = 0.7 x tfidf_similarity(resume, jd) + 0.3 x idf_skill_coverage

idf_skill_coverage = sum(IDF[skill] for matched skills)
                     / sum(IDF[skill] for all required skills)

IDF weights: faiss=3.2, pytorch=2.4, python=1.5, git=1.2
Rarer / more specialised skills count more.
```

---

## AI and LLM Integration

The live demo uses TF-IDF + IDF-weighted skill scoring — no API key needed.

To upgrade to real LLM-powered explanations, add one environment variable in Vercel:

```bash
# Vercel Dashboard -> Settings -> Environment Variables
OPENAI_API_KEY=sk-...   # enables GPT-4o explanations

# Or swap to any other provider in app/api/screen/route.ts:
# Anthropic Claude  -> import Anthropic from '@anthropic-ai/sdk'
# Google Gemini     -> import { GoogleGenerativeAI } from '@google/generative-ai'
# HuggingFace free  -> fetch('https://api-inference.huggingface.co/...')
```

The scoring pipeline is model-agnostic. The LLM only generates the explanation text.
Swap the provider in one function, everything else stays the same.

---

## The Hidden Gem Problem

The core value of semantic scoring over keyword matching:

```
JD says:                    Candidate writes:
"resume screening"    ->    "document understanding pipeline"
"rank candidates"     ->    "semantic similarity ranking system"
"bias detection"      ->    "algorithmic fairness, parity evaluation"

Keyword match score:  12%   <- buried at the bottom
Semantic score:       78%   <- correctly surfaced near the top
```

The system detects these automatically and flags them with a star badge.

---

## Project Structure

```
/
|-- app/
|   |-- api/screen/route.ts   <- scoring engine + API endpoint
|   |-- page.tsx              <- main UI + demo mode
|   |-- layout.tsx
|   `-- globals.css
|-- components/
|   |-- CandidateCard.tsx     <- ranked result card with scores
|   |-- ResultsView.tsx       <- results page + CSV export
|   |-- AnalyticsCharts.tsx   <- bar chart + radar comparison
|   |-- FileUpload.tsx        <- drag-and-drop PDF/DOCX upload
|   |-- JobDescriptionForm.tsx
|   `-- LoadingScreen.tsx
|-- store/
|   `-- screeningStore.ts     <- Zustand state
|-- data/sample_resumes/      <- 6 synthetic candidates for demo
|-- package.json
|-- next.config.ts
`-- vercel.json
```

---

## Sample Data

Six synthetic candidates in `data/sample_resumes/` demonstrate the system:

| Candidate | Role | Expected Score | Why |
|-----------|------|----------------|-----|
| Priya Sharma | Data Scientist | ~91% | Strong ML + NLP + fairness expertise |
| Aisha Rodriguez | MLOps Engineer | ~87% | Expert in embeddings + FAISS + DevOps |
| Alex Chen | Software Engineer | ~80% | Solid Python/ML all-rounder |
| Dr. Sarah Okonkwo | NLP Scientist | ~74% | **Hidden gem** - deep NLP, different vocab |
| Marcus Johnson | Full Stack Dev | ~38% | Growing ML interest, not ready for senior ML |
| James Whitfield | HR Manager | ~9% | Non-technical, correctly ranked last |

---

## What Works Right Now

- Resume upload (PDF + DOCX), drag-and-drop
- Job description parsing + skill extraction (200+ skills)
- Hybrid scoring: TF-IDF semantic + IDF-weighted skill coverage
- Candidate ranking with plain-English explanations
- Hidden gem detection (semantic >> keyword candidates)
- Score breakdown: semantic / skill / overall per candidate
- Matched vs missing skills per candidate
- Analytics: bar chart + radar comparison for top 3
- Fairness summary with bias flags
- CSV export
- Demo mode (LOAD DEMO - no upload needed)
- Configurable semantic/skill weight slider

---

## Future Scope

- Add `OPENAI_API_KEY` for GPT-4o explanations (one env var, zero code change)
- Swap to `sentence-transformers` + FAISS for real semantic embeddings (Docker deploy)
- ATS integrations (Greenhouse, Lever)

---

**Built by [Kunal Saini](https://github.com/kunal-gh)**

---

## Performance Note

The AI backend runs on **Render free tier** which spins down after 15 minutes of inactivity.

- **First request after inactivity:** ~30-60 seconds (model loading + cold start)
- **Subsequent requests:** ~3-8 seconds (model cached in memory)
- **What loads on first request:** sentence-transformers all-MiniLM-L6-v2 (~90MB) + spaCy en_core_web_sm

The frontend shows a warning banner during the cold start. Just wait — it will complete.

To eliminate cold starts, upgrade Render to a paid instance (\/month) or self-host via Docker.

