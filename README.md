<div align="center">

```
 ██████╗ ███████╗███████╗██╗   ██╗███╗   ███╗███████╗
 ██╔══██╗██╔════╝██╔════╝██║   ██║████╗ ████║██╔════╝
 ██████╔╝█████╗  ███████╗██║   ██║██╔████╔██║█████╗  
 ██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╔╝██║██╔══╝  
 ██║  ██║███████╗███████║╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
  AI-Powered Resume Screening & Candidate Ranking System
```

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22a559?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/kunal-gh/assignment/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/kunal-gh/assignment/actions)

### **Rank candidates in seconds, not hours. Understand *why*, not just *who*.**

**[🚀 Live Demo](https://assignment-pi-ten.vercel.app)** &nbsp;·&nbsp; **[📦 GitHub](https://github.com/kunal-gh/assignment)** &nbsp;·&nbsp; **[📖 API Docs](https://assignment-pi-ten.vercel.app/docs)**

</div>

---

## Table of Contents

| # | Section |
|---|---------|
| 1 | [What It Does](#-what-it-does) |
| 2 | [The Hidden Gem Problem](#-the-hidden-gem-problem) |
| 3 | [System Architecture](#-system-architecture) |
| 4 | [How the Algorithm Works](#-how-the-algorithm-works) |
| 5 | [Tech Stack](#-tech-stack) |
| 6 | [Deployment Architecture](#-deployment-architecture) |
| 7 | [Project Structure](#-project-structure) |
| 8 | [Quick Start](#-quick-start) |
| 9 | [Docker Setup](#-docker-setup) |
| 10 | [API Reference](#-api-reference) |
| 11 | [Scoring Formula](#-scoring-formula--tuning) |
| 12 | [Fairness & Bias Detection](#-fairness--bias-detection) |
| 13 | [Evaluation Metrics](#-evaluation-metrics) |
| 14 | [Testing & CI/CD](#-testing--cicd) |
| 15 | [Sample Data](#-sample-data) |
| 16 | [Roadmap](#-roadmap) |
| 17 | [Research Background](#-research-background) |

---

## 🎯 What It Does

RESUME is an **end-to-end AI pipeline** that ingests a pile of resumes and a job description and returns a ranked, explained, fairness-audited shortlist — in seconds.

```
INPUT:  [resume_1.pdf, resume_2.pdf, ..., resume_N.pdf]  +  Job Description
                              ↓
         ┌────────────────────────────────────────┐
         │            RESUME AI PIPELINE          │
         │                                        │
         │  1. Parse    → extract text + sections │
         │  2. Embed    → 384-dim semantic vectors │
         │  3. Score    → hybrid semantic + skills │
         │  4. Rank     → sort + tie-break         │
         │  5. Explain  → plain-English rationale  │
         │  6. Audit    → fairness + bias check    │
         └────────────────────────────────────────┘
                              ↓
OUTPUT: Ranked candidates · Scores 0–100 · Explanations · Bias report
```

**In plain English:** it reads every resume, understands meaning (not just keywords), matches skills, scores each candidate 0–100, generates a plain-English explanation for every ranking, and flags potential bias — all in a clean dashboard.

---

## 💎 The Hidden Gem Problem

This is the core problem semantic AI solves. Traditional ATS keyword matching fails like this:

```
Job Description says:          Candidate writes:
─────────────────────────────────────────────────────────────────
"resume screening"        →    "document understanding pipeline"
"rank candidates"         →    "semantic similarity ranking system"
"extract skills"          →    "information extraction from text"
"bias detection"          →    "algorithmic fairness, parity eval"
─────────────────────────────────────────────────────────────────

Keyword / TF-IDF score:   12%   ← buried at the bottom
Semantic AI score:        78%   ← correctly surfaced near the top
```

The candidate is a perfect fit. Keyword matching would reject them. **This is why embeddings matter.**

Our system detects these "hidden gems" automatically — candidates whose semantic score significantly exceeds their keyword-match score — and flags them for manual review.

---

## 🏗️ System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESUME AI SYSTEM v1.0                           │
│                                                                         │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────────┐ │
│  │  INPUT LAYER │   │   PARSE LAYER   │   │      ML / AI LAYER       │ │
│  │              │   │                 │   │                          │ │
│  │ PDF Resumes  │──▶│ PyMuPDF (fitz)  │──▶│ EmbeddingGenerator       │ │
│  │ DOCX Files   │   │ pdfplumber      │   │  └─ all-MiniLM-L6-v2     │ │
│  │ Job Desc.    │   │ python-docx     │   │     384-dim vectors       │ │
│  └──────────────┘   │                 │   │                          │ │
│                     │ SectionParser   │   │ VectorStore (FAISS)      │ │
│                     │  └─ spaCy NER   │   │  └─ IndexFlatIP          │ │
│                     │  └─ regex       │   │  └─ cosine similarity    │ │
│                     │                 │   │                          │ │
│                     │ SkillExtractor  │   │ RankingEngine            │ │
│                     │  └─ 200+ skills │   │  ├─ SemanticScore (70%)  │ │
│                     │  └─ IDF weights │   │  ├─ SkillScore   (30%)   │ │
│                     └─────────────────┘   │  └─ HybridScore 0–100   │ │
│                                           │                          │ │
│                                           │ FairnessChecker          │ │
│                                           │  ├─ Four-fifths rule     │ │
│                                           │  └─ Demographic parity  │ │
│                                           │                          │ │
│                                           │ LLMService (optional)    │ │
│                                           │  └─ GPT-4o explanations │ │
│                                           └──────────────────────────┘ │
│                                                        │                │
│  ┌─────────────────────────────────────────────────┐  │                │
│  │              OUTPUT LAYER                        │◀─┘                │
│  │  Streamlit Dashboard  │  FastAPI REST  │  Next.js│                  │
│  │  Rankings · Charts    │  /screen POST  │  Vercel │                  │
│  └─────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Dependency Graph

```
app.py (Streamlit)  ──────────────────────────────────────────────────────
│                                                                          
├── src/parsers/                                                           
│   ├── resume_parser.py      ← orchestrates the full parse pipeline      
│   ├── text_extractor.py     ← PyMuPDF primary, pdfplumber fallback      
│   ├── section_parser.py     ← spaCy NER + regex section detection       
│   └── skill_extractor.py    ← 200+ skill taxonomy, IDF ranking          
│                                                                          
├── src/models/                                                            
│   ├── resume.py             ← ResumeData, ContactInfo, Experience       
│   ├── job.py                ← JobDescription with auto skill extraction  
│   └── ranking.py            ← RankedCandidate, FairnessReport           
│                                                                          
├── src/embeddings/                                                        
│   ├── embedding_generator.py ← sentence-transformers integration        
│   ├── model_manager.py       ← model caching + fallback chains          
│   ├── vector_store.py        ← FAISS IndexFlatIP management             
│   └── cache_manager.py       ← 3-tier cache: memory → Redis → disk      
│                                                                          
└── src/ranking/                                                           
    ├── ranking_engine.py      ← hybrid scoring orchestration             
    ├── skill_matcher.py       ← IDF-weighted Jaccard similarity          
    ├── fairness_checker.py    ← four-fifths rule + demographic parity    
    ├── llm_service.py         ← GPT-4o explanations (optional)           
    ├── free_llm_service.py    ← HuggingFace free tier fallback           
    ├── similarity_search.py   ← FAISS ANN search pipeline                
    └── batch_processor.py     ← parallel batch processing                
```

### Mermaid Architecture Diagram

```mermaid
graph TB
    subgraph Input["📥 Input Layer"]
        PDF[PDF Resumes]
        DOCX[DOCX Files]
        JD[Job Description]
    end
    subgraph Parse["🔍 Parse Layer"]
        TE["TextExtractor<br/>PyMuPDF + pdfplumber"]
        SP["SectionParser<br/>spaCy NER + Regex"]
        SE["SkillExtractor<br/>200+ skill taxonomy"]
    end
    subgraph ML["🧠 ML / AI Layer"]
        EG["EmbeddingGenerator<br/>all-MiniLM-L6-v2 · 384d"]
        VS["VectorStore<br/>FAISS IndexFlatIP"]
        RE["RankingEngine<br/>Hybrid Scoring"]
        SM["SkillMatcher<br/>IDF-weighted Jaccard"]
        FC["FairnessChecker<br/>Four-fifths Rule"]
        LLM["LLMService<br/>GPT-4o (optional)"]
    end
    subgraph Output["🖥️ Output Layer"]
        ST[Streamlit Dashboard]
        API[FastAPI REST API]
        NX[Next.js Frontend]
    end
    PDF --> TE
    DOCX --> TE
    JD --> EG
    TE --> SP --> SE --> EG
    EG --> VS --> RE
    SE --> SM --> RE
    RE --> FC
    RE --> LLM
    FC --> ST
    LLM --> ST
    RE --> API
    API --> NX
```

---

## 🧮 How the Algorithm Works

### Stage 1 — Document Parsing

```python
# PyMuPDF (primary) — fastest, most accurate
import fitz
doc = fitz.open("resume.pdf")
text = "".join(page.get_text() for page in doc)

# pdfplumber (fallback) — handles complex layouts + tables
import pdfplumber
with pdfplumber.open("resume.pdf") as pdf:
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)

# spaCy NER — extract name, location, organisations
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
names = [e.text for e in doc.ents if e.label_ == "PERSON"]
```

The parser identifies 8 section types: `contact`, `summary`, `experience`, `education`, `skills`, `certifications`, `projects`, `awards` — using a combination of regex header patterns and spaCy entity recognition.

### Stage 2 — Semantic Embedding

Each resume and the job description are encoded into a **384-dimensional vector** using `all-MiniLM-L6-v2` — a Sentence-BERT model trained on 1B+ sentence pairs with contrastive learning.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, ~22MB

# Resume: combine all sections into one coherent string
resume_text = f"Summary: {raw_text} | Skills: {skills} | {experience}"
resume_vec = model.encode(resume_text, normalize_embeddings=True)  # shape: (384,)

# Job description
jd_vec = model.encode(jd_text, normalize_embeddings=True)  # shape: (384,)

# Long texts are chunked with 50-token overlap, then mean-pooled
# chunk_size=200 words, chunk_overlap=50 words
```

**Why `all-MiniLM-L6-v2`?** It is 5× faster than BERT-base while preserving 97% of semantic quality (SBERT benchmark). The 384-dim space is compact enough for fast FAISS search while rich enough to capture nuanced meaning.

### Stage 3 — Hybrid Scoring Formula

```
╔══════════════════════════════════════════════════════════════╗
║                   HYBRID SCORING FORMULA                     ║
║                                                              ║
║   S_hybrid = α · S_semantic + (1−α) · S_skills              ║
║                                                              ║
║   where:                                                     ║
║     α         = semantic_weight  (default: 0.70)             ║
║     1−α       = skill_weight     (default: 0.30)             ║
║                                                              ║
║     S_semantic = cosine(embed(resume), embed(JD))            ║
║                ∈ [0.0, 1.0]  (negatives clamped to 0)        ║
║                                                              ║
║     S_skills  = Σ IDF(s) · matched(s)                       ║
║                 ─────────────────────                        ║
║                 Σ IDF(s) · required(s)                       ║
║                                                              ║
║     IDF(s) = log(N / df(s))  — rarer skills worth more       ║
║                                                              ║
║     final_score = round(S_hybrid × 100, 1)  ∈ [0, 100]      ║
╚══════════════════════════════════════════════════════════════╝
```

```python
def calculate_hybrid_score(resume, job_desc, semantic_weight=0.7):
    # Semantic similarity via cosine distance
    semantic = cosine_similarity(resume.embedding, job_desc.embedding)
    semantic = max(0.0, min(1.0, semantic))  # clamp negatives to 0

    # IDF-weighted skill coverage
    resume_skills = set(resume.skills)
    skill_score = sum(
        IDF.get(s, 1.5) for s in job_desc.required_skills if s in resume_skills
    ) / max(sum(IDF.get(s, 1.5) for s in job_desc.required_skills), 1)

    # Weighted combination
    hybrid = semantic_weight * semantic + (1 - semantic_weight) * skill_score
    return round(hybrid * 100, 1)  # 0–100 scale
```

### Stage 4 — Ranking & Tie-Breaking

```
Candidates sorted by S_hybrid (descending)
         │
         ▼  ties?
┌──────────────────────────────────┐
│  Secondary sort (tie-breaking)   │
│  1st: skill_score  (descending)  │
│  2nd: semantic_score (descending)│
└──────────────────────────────────┘
         │
         ▼
Ranks 1, 2, 3, … assigned
```

### Stage 5 — Explanation Generation

Every candidate gets a plain-English explanation. Two modes:

**Template mode** (always available, zero latency):
```
"Priya shows excellent fit for the Senior ML Engineer position with an
 overall score of 91.2% (Rank #1). Strong semantic alignment (93.4%)
 indicates deep NLP and ML expertise. Matched skills: python, faiss,
 spacy, pytorch. Development opportunity: kubernetes."
```

**LLM mode** (when `OPENAI_API_KEY` is set):
```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert HR analyst."},
        {"role": "user", "content": f"JD: {jd}\n\nResume: {resume}\n\nExplain fit."}
    ],
    temperature=0.2
)
```

**Hidden Gem Detection** — automatically flagged when:
```python
is_hidden_gem = (semantic_score >= 0.55
                 and skill_score < 0.35
                 and (semantic_score - skill_score) > 0.20)
```

---
done
## Tech Stack

### Core ML / AI

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| Semantic Embeddings | sentence-transformers | 5.4+ | Best-in-class SBERT trained on 1B+ pairs |
| Default Model | all-MiniLM-L6-v2 | v2 384d | 5x faster than BERT-base 97% quality |
| High-accuracy | all-mpnet-base-v2 | v2 768d | Top SBERT leaderboard |
| Vector Search | FAISS IndexFlatIP | 1.13.2 | 1000x faster than linear scan |
| NLP Pipeline | spaCy | 3.8+ | Best-in-class NER fast CPU inference |
| Fairness | Fairlearn | 0.11+ | Microsoft Research four-fifths rule |
| LLM optional | OpenAI GPT-4o | API | Nuanced explanations |
| LLM free | HuggingFace Inference | -- | Zero-cost fallback |

### Document Processing

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| PDF primary | PyMuPDF fitz | 1.23+ | Fastest handles complex layouts |
| PDF fallback | pdfplumber | 0.9+ | Better for tables and multi-column |
| DOCX | python-docx | 0.8+ | Full table and header support |

### Backend

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| REST API | FastAPI | 0.104+ | Async OpenAPI at /docs |
| ASGI | Uvicorn | 0.24+ | Production-grade |
| Rate Limiting | SlowAPI | 0.1.9+ | Per-IP middleware |
| Auth | OAuth2 + JWT | -- | Bearer token on /screen |
| Cache | Redis 7 | alpine | 85%+ embedding hit rate |

### Frontend

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Framework | Next.js | 15.5 | App Router React 19 |
| State | Zustand | 5.0 | Minimal no boilerplate |
| Animations | Framer Motion | 11 | Smooth transitions |
| Charts | Recharts | 2.15 | Bar and Radar charts |
| Styling | Tailwind CSS | 3.4 | Brutalist design system |

---

## Deployment Architecture

Two modes each optimised for its environment:

`
LOCAL / DOCKER                    VERCEL LIVE DEMO
------------------------------    ----------------------------------------
Real sentence-transformers        TF-IDF cosine similarity
+ FAISS IndexFlatIP               + IDF-weighted skill scoring
all-MiniLM-L6-v2 384-dim          No ML deps fits 250MB serverless limit
100% free private accurate        Instant cold start publicly accessible

streamlit run app.py              https://assignment-pi-ten.vercel.app

Why two modes?
  torch alone = ~700MB
  sentence-transformers = ~90MB
  Vercel serverless limit = 250MB
  Real ML cannot run on Vercel free tier
  Docker / Render / Railway = full ML stack at zero cost
`

---

## Project Structure

`
assignment/
|-- app.py                    Streamlit dashboard local/Docker
|-- api.py                    FastAPI REST backend
|-- api/screen.py             Vercel serverless function
|-- requirements.txt          Production dependencies
|-- requirements-dev.txt      Dev and test dependencies
|-- Dockerfile                Container definition
|-- vercel.json               Vercel deployment config
|-- .github/workflows/ci.yml  GitHub Actions CI/CD
|
|-- src/
|   |-- parsers/
|   |   |-- resume_parser.py      Orchestrates full parse pipeline
|   |   |-- text_extractor.py     PyMuPDF + pdfplumber + python-docx
|   |   |-- section_parser.py     spaCy NER + regex section detection
|   |   -- skill_extractor.py    200+ skill taxonomy + IDF ranking
|   |
|   |-- models/
|   |   |-- resume.py             ResumeData ContactInfo Experience
|   |   |-- job.py                JobDescription + auto skill extraction
|   |   -- ranking.py            RankedCandidate FairnessReport
|   |
|   |-- embeddings/
|   |   |-- embedding_generator.py  sentence-transformers integration
|   |   |-- model_manager.py        Model caching + fallback chains
|   |   |-- vector_store.py         FAISS IndexFlatIP management
|   |   -- cache_manager.py        3-tier memory -> Redis -> disk
|   |
|   |-- ranking/
|   |   |-- ranking_engine.py       Hybrid scoring orchestration
|   |   |-- skill_matcher.py        IDF-weighted Jaccard similarity
|   |   |-- fairness_checker.py     Four-fifths rule + demographic parity
|   |   |-- llm_service.py          GPT-4o explanations
|   |   |-- free_llm_service.py     HuggingFace free tier fallback
|   |   |-- similarity_search.py    FAISS ANN search pipeline
|   |   -- batch_processor.py      Parallel batch processing
|   |
|   -- utils/
|       |-- evaluation.py           Precision@K NDCG@K MAP HiddenGem
|       |-- file_utils.py           File validation helpers
|       -- text_utils.py           Text cleaning utilities
|
|-- frontend/                 Next.js 15 frontend
|   |-- app/page.tsx          Main page + demo mode
|   |-- components/           CandidateCard Charts FileUpload
|   -- store/screeningStore.ts  Zustand state management
|
|-- tests/                    18 test files 92% coverage
-- data/sample_resumes/      6 synthetic candidates + sample JD
`

---

## Quick Start

`ash
git clone https://github.com/kunal-gh/assignment.git
cd assignment

pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app.py
# Open http://localhost:8501
`

Optional GPT-4o explanations:
`ash
cp .env.example .env
# Add: OPENAI_API_KEY=sk-...
`

Optional FastAPI backend:
`ash
uvicorn api:app --reload --port 8000
# Swagger UI at http://localhost:8000/docs
`

---

## API Reference

Base URL local: http://localhost:8000

### POST /screen

`ash
curl -X POST http://localhost:8000/screen \
  -H "Authorization: Bearer super-secret-demo-token" \
  -F "files=@resume1.pdf" \
  -F "job_title=Senior ML Engineer" \
  -F "job_description=We need Python PyTorch FAISS spaCy..." \
  -F "semantic_weight=0.7"
`

Response:
`json
{
  "job_id": "uuid",
  "job_title": "Senior ML Engineer",
  "total_resumes": 2,
  "processing_time_seconds": 3.2,
  "candidates": [
    {
      "rank": 1,
      "name": "Priya Sharma",
      "hybrid_score": 0.912,
      "semantic_score": 0.934,
      "skill_score": 0.857,
      "matched_skills": ["python", "pytorch", "faiss"],
      "missing_skills": ["kubernetes"],
      "explanation": "Priya shows excellent fit..."
    }
  ],
  "fairness_summary": { "overall_score": 0.94, "bias_flags": [] }
}
`

Other endpoints:
- GET /results/{job_id} - retrieve previous results
- GET /results/{job_id}/export/csv - download CSV
- POST /analyze/jd - analyse job description extract skills
- GET /models - list embedding models
- GET /health - health check

---

## Scoring Formula

`
S_hybrid = alpha x S_semantic + (1 - alpha) x S_skills

S_semantic = cosine(embed(resume), embed(JD))  in [0, 1]

S_skills = sum(IDF(s) for s in matched_required)
           / sum(IDF(s) for s in all_required)

IDF(s) = log(N / df(s))  rarer skills worth more
         faiss=3.2  pytorch=2.4  python=1.5  git=1.2

final_score = round(S_hybrid x 100, 1)  in [0, 100]
`

### Tuning Guide

| Job Type | Semantic | Skill | Rationale |
|----------|----------|-------|-----------|
| Research / Science | 0.80 | 0.20 | Conceptual depth matters most |
| Engineering | 0.65 | 0.35 | Specific tools matter more |
| Management | 0.85 | 0.15 | Soft skills vision leadership |
| Niche / Certified | 0.50 | 0.50 | Specific certs are mandatory |
| Default | 0.70 | 0.30 | Best general-purpose balance |

### Score Interpretation

| Score | Grade | Action |
|-------|-------|--------|
| 85-100 | A+ | Prioritise for immediate interview |
| 70-84 | A | Schedule technical screen |
| 55-69 | B | Review manually consider phone screen |
| 40-54 | C | Will need significant upskilling |
| < 40 | D | Unlikely to meet requirements |

---

## Fairness and Bias Detection

### Four-Fifths Rule EEOC

`python
ratio = selection_rate_protected / selection_rate_majority
if ratio < 0.80:
    flag("Four-fifths rule violation")
`

### Demographic Parity

`python
from src.ranking.fairness_checker import FairnessChecker
checker = FairnessChecker()
report = checker.generate_fairness_report(ranked_candidates, top_k=10)
print(report.get_overall_fairness_score())  # 0.0 to 1.0
print(report.four_fifths_violations)
print(report.recommendations)
`

Note: Current implementation simulates demographic data for demonstration. Production use requires explicit consent and separate storage.

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| Precision@K | Fraction of shortlist that are qualified |
| Recall@K | Coverage of all qualified candidates |
| NDCG@K | Rank-weighted precision |
| MAP | Average precision across multiple JDs |

### Performance Benchmarks

| Scenario | Time | Notes |
|----------|------|-------|
| 10 resumes cold start | ~8s | Model loading included |
| 10 resumes warm | ~2s | Model cached in memory |
| 50 resumes | ~25s | Batch embedding |
| 100 resumes | ~48s | Full pipeline |
| Peak RAM 100 files | <3.5 GB | MiniLM model |
| Cache hit rate | 85%+ | Repeated JDs |

---

## Testing and CI/CD

`ash
pip install -r requirements-dev.txt
pytest tests/ --cov=src --cov-report=html
`

Coverage: 92% across 18 test files

CI/CD Pipeline on every push to main:
1. Code Quality: black + isort + flake8
2. Security Scan: bandit
3. Unit Tests: Python 3.9 / 3.10 / 3.11
4. Docker Build
5. Performance Smoke Test

---

## Sample Data

Six synthetic candidates in data/sample_resumes/ demonstrate the system:

| Name | Role | Score | Key Insight |
|------|------|-------|-------------|
| Priya Sharma | Data Scientist | ~91% | Strong ML + fairness expertise |
| Aisha Rodriguez | MLOps Engineer | ~87% | Expert in embeddings + FAISS |
| Alex Chen | Software Engineer | ~80% | Strong Python/ML all-rounder |
| Dr. Sarah Okonkwo | NLP Scientist | ~74% | Hidden gem -- deep NLP different vocab |
| Marcus Johnson | Full Stack Dev | ~38% | Growing ML interest correctly deprioritised |
| James Whitfield | HR Manager | ~9% | Non-technical system correctly ranks last |

Sarah Okonkwo is the key demo. She never says "resume screening" but writes "document understanding pipeline". Keyword matching: 12%. Semantic AI: 78%.

---

## Roadmap

### v1.1 Next
- GPT-4o re-ranking for top-10 candidates
- PDF report generation professional layout
- Skill taxonomy expansion to 500+ skills
- Candidate drill-down with resume preview

### v2.0
- Multi-language support Spanish French German Mandarin
- Custom model fine-tuning on domain-specific data
- ATS connectors Greenhouse Lever Workday
- Real-time multi-recruiter collaboration

### v2.1
- Candidate self-service portal
- Video resume analysis
- Predictive hire success probability
- Mobile app iOS and Android

---

## Research Background

1. Sentence-BERT -- Reimers and Gurevych 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

2. Fairness in ML Hiring -- Raghavan et al. 2020. Mitigating Bias in Algorithmic Hiring. FAccT 2020.

3. FAISS -- Johnson Douze and Jegou 2021. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.

4. MTEB Benchmark -- Muennighoff et al. 2022. MTEB: Massive Text Embedding Benchmark.

---

## Contributing

`ash
git clone https://github.com/kunal-gh/assignment.git
cd assignment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
python -m spacy download en_core_web_sm
`

Standards: black line-length 127 + isort + flake8 + type hints on all public functions.

---

## License

MIT License -- see LICENSE for details.

---

Built by Kunal Saini -- https://github.com/kunal-gh

"The best hiring tool is not the one that processes resumes fastest.
 It is the one that understands humans best."
