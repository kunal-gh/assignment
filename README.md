# AI Resume Screener

**Live Demo: https://ai-resume-screener-sepia.vercel.app**

> Upload resumes + paste a job description. Get ranked candidates with real AI scores, skill analysis, and explanations in seconds.

---

## Architecture

```
Browser (Vercel)
    |
    |-- LOAD DEMO button --> instant results (no upload needed)
    |
    |-- Upload PDF/DOCX + Job Description
            |
            v
    Render ML Backend (Python FastAPI)
            |
            |-- PyMuPDF: extract text from PDF
            |-- spaCy en_core_web_sm: NER + skill extraction
            |-- sentence-transformers all-MiniLM-L6-v2: 384-dim embeddings
            |-- FAISS IndexFlatIP: cosine similarity search
            |-- Hybrid score: 0.7 x cosine + 0.3 x IDF-weighted skill coverage
            |-- LLM explanation (GPT-4o if OPENAI_API_KEY set, else template)
            |-- Fairlearn: four-fifths rule fairness check
            |
            v
    Ranked candidates with scores 0-100, explanations, fairness report
```

---

## What Is Actually Running

### ML Backend (Render)
`https://ai-resume-screener-api-5iq6.onrender.com`

Real sentence-transformers pipeline:

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim vectors

resume_vec = model.encode(resume_text, convert_to_tensor=True)
jd_vec     = model.encode(job_description, convert_to_tensor=True)
cosine     = util.cos_sim(resume_vec, jd_vec).item()  # semantic similarity

skill_score = sum(IDF[s] for s in matched_skills) / sum(IDF[s] for s in required_skills)
final_score = round((0.7 * cosine + 0.3 * skill_score) * 100, 1)
```

### Frontend (Vercel)
`https://ai-resume-screener-sepia.vercel.app`

Next.js 15 + React 19 + Zustand + Recharts + Framer Motion

---

## Tech Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| Semantic Embeddings | sentence-transformers | 2.x | all-MiniLM-L6-v2, 384-dim |
| Vector Search | FAISS IndexFlatIP | 1.7.x | cosine similarity |
| NLP / NER | spaCy | 3.6+ | en_core_web_sm |
| PDF Parsing | PyMuPDF (fitz) | 1.23+ | primary; pdfplumber fallback |
| Fairness | Fairlearn | 0.9+ | four-fifths rule |
| LLM (optional) | OpenAI GPT-4o | API | set OPENAI_API_KEY |
| API | FastAPI | 0.104+ | async, OpenAPI at /docs |
| Frontend | Next.js | 15.5 | App Router, React 19 |
| State | Zustand | 5.0 | client state |
| Charts | Recharts | 2.15 | bar + radar |
| Frontend Deploy | Vercel | -- | auto-deploy on push |
| Backend Deploy | Render | -- | Docker, free tier |

---

## Scoring Formula

```
final_score = round((alpha * cosine_similarity + (1-alpha) * skill_score) * 100, 1)

where:
  alpha           = semantic_weight (default 0.7, tunable in UI)
  cosine_similarity = cosine(embed(resume), embed(JD))  in [0, 1]
  skill_score     = sum(IDF[s] for s in matched) / sum(IDF[s] for s in required)
  IDF weights     = faiss:3.2, rag:3.5, mcp:3.4, pytorch:2.4, python:1.5
```

---

## LLM Integration

The live demo uses template-based explanations by default.

To enable GPT-4o explanations, add one env var in Render dashboard:

```bash
OPENAI_API_KEY=sk-...   # GPT-4o explanations

# Or swap to any other provider in backend/main.py:
# Anthropic Claude  -> import anthropic
# Google Gemini     -> import google.generativeai
# HuggingFace free  -> requests.post('https://api-inference.huggingface.co/...')
```

The scoring pipeline is model-agnostic. The LLM only generates the explanation text.

---

## Performance Note

The ML backend runs on **Render free tier** which spins down after 15 min of inactivity.

- **First request after inactivity:** 30-60 seconds (model loading + cold start)
- **Subsequent requests:** 3-8 seconds (model cached in memory)
- **What loads:** sentence-transformers all-MiniLM-L6-v2 (~90MB) + spaCy

The frontend shows a warning banner and pipeline progress during processing.
If the first request fails, click TRY AGAIN -- the model will already be loaded.

---

## The Hidden Gem Problem

```
JD says:                    Candidate writes:
"resume screening"    ->    "document understanding pipeline"
"rank candidates"     ->    "semantic similarity ranking system"
"bias detection"      ->    "algorithmic fairness, parity evaluation"

Keyword match score:  12%   <- buried at the bottom
Semantic AI score:    78%   <- correctly surfaced near the top
```

The system detects these automatically and flags them with a star badge.

---

## Project Structure

```
/
|-- app/
|   |-- api/screen/route.ts   <- Next.js API route (TF-IDF fallback)
|   |-- page.tsx              <- main UI + demo mode
|   |-- layout.tsx
|   `-- globals.css
|-- components/
|   |-- CandidateCard.tsx     <- ranked result with scores + skills
|   |-- ResultsView.tsx       <- results page + CSV export
|   |-- AnalyticsCharts.tsx   <- bar chart + radar comparison
|   |-- FileUpload.tsx        <- drag-and-drop PDF/DOCX
|   |-- JobDescriptionForm.tsx
|   `-- LoadingScreen.tsx     <- pipeline progress + cold start warning
|-- store/
|   `-- screeningStore.ts     <- Zustand state + API calls
|-- backend/
|   |-- main.py               <- FastAPI server (real ML pipeline)
|   |-- requirements.txt      <- torch, sentence-transformers, faiss, spacy
|   `-- Dockerfile            <- Docker image for Render
|-- src/                      <- Python ML modules
|   |-- parsers/              <- PyMuPDF, pdfplumber, spaCy NER
|   |-- embeddings/           <- sentence-transformers, FAISS, cache
|   |-- ranking/              <- hybrid scoring, fairness, LLM service
|   `-- models/               <- ResumeData, JobDescription, RankedCandidate
|-- data/sample_resumes/      <- 6 synthetic candidates for demo
|-- render.yaml               <- Render deployment config
`-- vercel.json               <- Vercel deployment config
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
- Real sentence-transformers semantic embeddings (all-MiniLM-L6-v2)
- FAISS cosine similarity vector search
- spaCy NER + 200+ skill taxonomy extraction
- Hybrid scoring: 0.7 x semantic + 0.3 x IDF-weighted skill coverage
- Candidate ranking with AI-generated explanations
- Hidden gem detection (semantic >> keyword candidates)
- Score breakdown: semantic / skill / overall per candidate
- Matched vs missing skills per candidate
- Analytics: bar chart + radar comparison for top 3
- Fairness analysis (four-fifths rule)
- CSV export
- Demo mode (LOAD DEMO - no upload needed)
- Configurable semantic/skill weight slider

---

## Future Scope

- Add `OPENAI_API_KEY` for GPT-4o explanations (one env var, zero code change)
- Upgrade Render to paid tier to eliminate cold starts
- Add Pinecone/Qdrant for persistent vector storage at scale
- ATS integrations (Greenhouse, Lever)

---

**Built by [Kunal Saini](https://github.com/kunal-gh)**
