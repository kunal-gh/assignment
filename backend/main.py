"""
AI Resume Screener — FastAPI Backend
Real ML pipeline: sentence-transformers + FAISS + spaCy + LLM explanations

Deploy on Render / Railway / any Docker host (not Vercel — needs 2GB+ RAM for ML models)
"""

import io
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Resume Screener API",
    description=(
        "Real ML pipeline: sentence-transformers (all-MiniLM-L6-v2) + FAISS + spaCy NER. "
        "Hybrid scoring: 70% semantic cosine similarity + 30% IDF-weighted skill coverage."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-load ML components (avoid cold-start timeout) ───────────────────────

_parser = None
_engine = None
_model_loaded = False


def get_parser():
    global _parser
    if _parser is None:
        from src.parsers.resume_parser import ResumeParser
        _parser = ResumeParser()
        logger.info("ResumeParser loaded")
    return _parser


def get_engine(semantic_weight: float = 0.7):
    global _engine, _model_loaded
    if _engine is None:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.ranking.ranking_engine import RankingEngine

        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        generator = EmbeddingGenerator(model_name=model_name)
        _engine = RankingEngine(
            semantic_weight=semantic_weight,
            skill_weight=1.0 - semantic_weight,
            embedding_generator=generator,
        )
        _model_loaded = True
        logger.info("RankingEngine ready")
    return _engine


# ─── Response models ──────────────────────────────────────────────────────────


class CandidateResult(BaseModel):
    rank: int
    name: str
    email: Optional[str]
    hybrid_score: float
    semantic_score: float
    skill_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    years_experience: int
    explanation: Optional[str]


class ScreeningResult(BaseModel):
    job_id: str
    job_title: str
    total_resumes: int
    successfully_parsed: int
    processing_time_seconds: float
    candidates: List[CandidateResult]
    fairness_summary: Optional[Dict[str, Any]]
    created_at: str
    model_used: str


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model_loaded,
        "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "version": "2.0.0",
    }


@app.get("/")
async def root():
    return {
        "message": "AI Resume Screener API — Real ML Backend",
        "docs": "/docs",
        "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "pipeline": "sentence-transformers → FAISS cosine similarity → IDF skill scoring → LLM explanations",
    }


@app.post("/screen", response_model=ScreeningResult)
async def screen_resumes(
    files: List[UploadFile] = File(...),
    job_title: str = Form(default="Software Engineer"),
    job_description: str = Form(...),
    semantic_weight: float = Form(default=0.7),
    include_fairness: bool = Form(default=True),
    embedding_model: str = Form(default="all-MiniLM-L6-v2"),
):
    """
    Screen resumes against a job description using real ML.

    Pipeline:
    1. PyMuPDF / pdfplumber text extraction
    2. spaCy NER + 200+ skill taxonomy extraction
    3. sentence-transformers all-MiniLM-L6-v2 embeddings (384-dim)
    4. FAISS cosine similarity search
    5. Hybrid score: 0.7 * semantic + 0.3 * IDF-weighted skill coverage
    6. LLM explanation (if OPENAI_API_KEY set, else template)
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file required")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description required")

    t0 = time.time()
    job_id = str(uuid.uuid4())

    # Save uploaded files to temp dir
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for upload in files:
        fname = upload.filename or f"resume_{len(file_paths)}.pdf"
        if not fname.lower().endswith((".pdf", ".docx")):
            continue
        dest = os.path.join(temp_dir, fname)
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        file_paths.append(dest)

    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid PDF/DOCX files")

    try:
        from src.models.job import JobDescription
        from src.ranking.skill_matcher import SkillMatcher

        parser = get_parser()
        engine = get_engine(semantic_weight)

        # Parse resumes
        logger.info(f"Parsing {len(file_paths)} resumes...")
        resumes = parser.batch_parse(file_paths)

        # Create job description — auto-extracts skills from text
        job_desc = JobDescription(
            title=job_title,
            description=job_description,
        )

        # Run full ML pipeline
        logger.info("Running ranking pipeline...")
        batch_result = engine.process_batch(
            resumes=resumes,
            job_desc=job_desc,
            include_fairness=include_fairness,
        )

        # Build response
        matcher = SkillMatcher()
        candidates_out = []
        for c in batch_result.ranked_candidates:
            resume = c.resume
            name = resume.contact_info.name if resume.contact_info else "Unknown"
            email = resume.contact_info.email if resume.contact_info else None

            analysis = matcher.analyze_skill_match(
                resume.skills,
                job_desc.required_skills,
                job_desc.preferred_skills,
            )

            candidates_out.append(
                CandidateResult(
                    rank=c.rank,
                    name=name,
                    email=email,
                    hybrid_score=round(c.hybrid_score, 4),
                    semantic_score=round(c.semantic_score, 4),
                    skill_score=round(c.skill_score, 4),
                    matched_skills=analysis.get("matched_required", [])[:10],
                    missing_skills=analysis.get("missing_required", [])[:10],
                    years_experience=resume.get_years_of_experience(),
                    explanation=c.explanation,
                )
            )

        fairness_summary = None
        if batch_result.fairness_report:
            fr = batch_result.fairness_report
            fairness_summary = {
                "overall_score": fr.get_overall_fairness_score(),
                "bias_flags": fr.bias_flags,
                "recommendations": fr.recommendations[:3],
            }

        return ScreeningResult(
            job_id=job_id,
            job_title=job_title,
            total_resumes=batch_result.total_resumes,
            successfully_parsed=batch_result.successfully_parsed,
            processing_time_seconds=round(time.time() - t0, 3),
            candidates=candidates_out,
            fairness_summary=fairness_summary,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_used=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )

    finally:
        for p in file_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@app.get("/models")
async def list_models():
    """Available embedding models."""
    return {
        "models": [
            {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "speed": "fast",
                "description": "Default — 5x faster than BERT, 97% quality. Best for most use cases.",
                "is_default": True,
            },
            {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "speed": "medium",
                "description": "Higher accuracy — top of SBERT leaderboard. Use when quality > speed.",
                "is_default": False,
            },
            {
                "name": "multi-qa-MiniLM-L6-cos-v1",
                "dimensions": 384,
                "speed": "fast",
                "description": "Optimised for question-answer style job descriptions.",
                "is_default": False,
            },
        ]
    }
