"""FastAPI backend for AI Resume Screener — REST API endpoints."""

import io
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.job import JobDescription
from src.parsers.resume_parser import ResumeParser
from src.ranking.ranking_engine import RankingEngine

logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Resume Screener API",
    description=(
        "REST API for AI-powered resume screening and ranking. "
        "Combines semantic embeddings with skill matching to rank candidates."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# In-memory job store (replace with Redis/DB in production)
_job_store: Dict[str, Any] = {}

# Security setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    # In a real app, decode JWT and verify against DB
    if token != "super-secret-demo-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "admin"}


# ─── Pydantic Models ──────────────────────────────────────────────────────────


class JobDescriptionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    description: str = Field(..., min_length=10, description="Full job description text")
    required_skills: Optional[List[str]] = Field(default=None, description="Explicit required skills list")
    preferred_skills: Optional[List[str]] = Field(default=None, description="Nice-to-have skills")
    experience_level: Optional[str] = Field(default=None, description="junior/mid/senior/lead")


class ScoringConfig(BaseModel):
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    skill_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    include_fairness: bool = Field(default=True)
    max_candidates: int = Field(default=50, ge=1, le=500)

    def model_post_init(self, __context: Any) -> None:
        # Ensure weights sum to 1
        total = self.semantic_weight + self.skill_weight
        if abs(total - 1.0) > 1e-6:
            self.skill_weight = round(1.0 - self.semantic_weight, 10)


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


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ─── Dependency: shared components ────────────────────────────────────────────

_parser_cache: Optional[ResumeParser] = None
_engine_cache: Dict[str, RankingEngine] = {}


def get_parser() -> ResumeParser:
    global _parser_cache
    if _parser_cache is None:
        _parser_cache = ResumeParser()
    return _parser_cache


def get_engine(semantic_weight: float = 0.7, skill_weight: float = 0.3, model_name: str = "all-MiniLM-L6-v2") -> RankingEngine:
    key = f"{model_name}_{semantic_weight}_{skill_weight}"
    if key not in _engine_cache:
        generator = EmbeddingGenerator(model_name=model_name)
        _engine_cache[key] = RankingEngine(
            semantic_weight=semantic_weight,
            skill_weight=skill_weight,
            embedding_generator=generator,
        )
    return _engine_cache[key]


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/token", tags=["Auth"])
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Dummy auth for demo purposes
    if form_data.username == "admin" and form_data.password == "password":
        return {"access_token": "super-secret-demo-token", "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/", tags=["System"])
@limiter.limit("60/minute")
async def root(request: Request):
    """API root — redirects to docs."""
    return {"message": "AI Resume Screener API", "docs": "/docs", "version": "1.0.0"}


@app.post("/screen", response_model=ScreeningResult, tags=["Screening"])
@limiter.limit("10/minute")
async def screen_resumes(
    request: Request,
    files: List[UploadFile] = File(..., description="Resume files (PDF or DOCX)"),
    job_title: str = "Software Engineer",
    job_description: str = "We are looking for a skilled engineer...",
    semantic_weight: float = 0.7,
    include_fairness: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    current_user: dict = Depends(get_current_user),
):
    """
    Upload resumes and screen them against a job description.

    Returns a ranked list of candidates with scores and explanations.
    Scores are on a 0.0–1.0 scale (multiply by 100 for percentage).
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file is required")

    skill_weight = round(1.0 - semantic_weight, 10)
    job_id = str(uuid.uuid4())

    # Save uploaded files to temp directory
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for upload in files:
        if not upload.filename.lower().endswith((".pdf", ".docx")):
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {upload.filename}. Only PDF and DOCX accepted."
            )
        dest = os.path.join(temp_dir, upload.filename)
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        file_paths.append(dest)

    try:
        parser = get_parser()
        engine = get_engine(semantic_weight, skill_weight, embedding_model)

        job_desc = JobDescription(title=job_title, description=job_description)
        resumes = parser.batch_parse(file_paths)

        batch_result = engine.process_batch(resumes, job_desc, include_fairness=include_fairness)

        # Build response
        candidates_out = []
        for c in batch_result.ranked_candidates:
            resume = c.resume
            name = resume.contact_info.name if resume.contact_info else "Unknown"
            email = resume.contact_info.email if resume.contact_info else None

            # Get skill analysis
            try:
                from src.ranking.skill_matcher import SkillMatcher

                matcher = SkillMatcher()
                analysis = matcher.analyze_skill_match(
                    resume.skills,
                    job_desc.required_skills,
                    job_desc.preferred_skills,
                )
                matched = analysis.get("matched_required", [])
                missing = analysis.get("missing_required", [])
            except Exception:
                matched, missing = [], []

            candidates_out.append(
                CandidateResult(
                    rank=c.rank,
                    name=name,
                    email=email,
                    hybrid_score=round(c.hybrid_score, 4),
                    semantic_score=round(c.semantic_score, 4),
                    skill_score=round(c.skill_score, 4),
                    matched_skills=matched[:10],
                    missing_skills=missing[:10],
                    years_experience=resume.get_years_of_experience(),
                    explanation=c.explanation,
                )
            )

        # Fairness summary
        fairness_summary = None
        if batch_result.fairness_report:
            fr = batch_result.fairness_report
            fairness_summary = {
                "overall_score": fr.get_overall_fairness_score(),
                "bias_flags": fr.bias_flags,
                "recommendations": fr.recommendations[:3],
                "violations": fr.four_fifths_violations,
            }

        result = ScreeningResult(
            job_id=job_id,
            job_title=job_title,
            total_resumes=batch_result.total_resumes,
            successfully_parsed=batch_result.successfully_parsed,
            processing_time_seconds=round(batch_result.processing_time, 3),
            candidates=candidates_out,
            fairness_summary=fairness_summary,
            created_at=datetime.utcnow().isoformat(),
        )

        _job_store[job_id] = result.model_dump()
        return result

    finally:
        # Cleanup temp files
        for p in file_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass


@app.get("/results/{job_id}", response_model=ScreeningResult, tags=["Screening"])
@limiter.limit("30/minute")
async def get_results(request: Request, job_id: str):
    """Retrieve screening results for a previous job by ID."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")
    return _job_store[job_id]


@app.get("/results/{job_id}/export/csv", tags=["Export"])
@limiter.limit("10/minute")
async def export_csv(request: Request, job_id: str):
    """Export screening results as a CSV file download."""
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")

    result = _job_store[job_id]
    candidates = result["candidates"]

    rows = []
    for c in candidates:
        rows.append(
            {
                "Rank": c["rank"],
                "Name": c["name"],
                "Email": c["email"] or "",
                "Overall Score (%)": round(c["hybrid_score"] * 100, 1),
                "Semantic Score (%)": round(c["semantic_score"] * 100, 1),
                "Skill Score (%)": round(c["skill_score"] * 100, 1),
                "Years Experience": c["years_experience"],
                "Matched Skills": ", ".join(c["matched_skills"]),
                "Missing Skills": ", ".join(c["missing_skills"]),
                "Explanation": c["explanation"] or "",
            }
        )

    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    filename = f"screening_results_{job_id[:8]}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([csv_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/analyze/jd", tags=["Analysis"])
@limiter.limit("20/minute")
async def analyze_job_description(request: Request, body: JobDescriptionRequest):
    """
    Analyze a job description — extract required skills, experience level,
    and provide a preview of what the system will match against.
    """
    job_desc = JobDescription(
        title=body.title,
        description=body.description,
        required_skills=body.required_skills or [],
        preferred_skills=body.preferred_skills or [],
        experience_level=body.experience_level or "mid",
    )

    return {
        "title": job_desc.title,
        "extracted_required_skills": job_desc.required_skills[:20],
        "extracted_preferred_skills": job_desc.preferred_skills[:20],
        "experience_level": job_desc.experience_level,
        "skill_count": len(job_desc.required_skills),
        "description_length": len(job_desc.description),
        "tip": ("The more specific your job description, " "the better the semantic matching will perform."),
    }


@app.get("/models", tags=["Configuration"])
@limiter.limit("60/minute")
async def list_models(request: Request):
    """List available embedding models with their characteristics."""
    return {
        "models": [
            {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "speed": "fast",
                "quality": "good",
                "recommended_for": "most use cases, fast processing",
                "is_default": True,
            },
            {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "speed": "medium",
                "quality": "excellent",
                "recommended_for": "when highest quality matters more than speed",
                "is_default": False,
            },
            {
                "name": "multi-qa-MiniLM-L6-cos-v1",
                "dimensions": 384,
                "speed": "fast",
                "quality": "good",
                "recommended_for": "question-answer style job descriptions",
                "is_default": False,
            },
        ]
    }


@app.get("/metrics", tags=["System"])
@limiter.limit("60/minute")
async def system_metrics(request: Request):
    """System metrics — cached jobs, engine instances."""
    return {
        "cached_jobs": len(_job_store),
        "cached_engines": len(_engine_cache),
        "timestamp": datetime.utcnow().isoformat(),
    }
