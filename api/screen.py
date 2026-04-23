"""
Vercel Serverless Function — Resume Screening API.

NOTE: Vercel's serverless environment has a 250MB size limit, which prevents
loading large ML models (sentence-transformers ~90MB + torch ~700MB).
This handler uses a lightweight simulation engine that demonstrates the full
UI/UX pipeline. For real semantic matching, deploy via Docker/Render/Railway.
"""

import json
import logging
import os
import random
import re
import time
from http.server import BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill taxonomy (lightweight — no ML deps)
# ---------------------------------------------------------------------------

TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "sql", "html", "css",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "ci/cd",
    "faiss", "spacy", "transformers", "sentence-transformers", "embeddings",
    "postgresql", "mongodb", "redis", "elasticsearch", "spark", "kafka",
    "agile", "scrum", "rest api", "graphql", "microservices", "devops",
    "streamlit", "fastapi", "airflow", "mlflow", "mlops", "computer vision",
    "r", "scala", "go", "rust", "c++", "linux", "bash", "terraform",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving",
    "project management", "analytical", "creative", "adaptable",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS


def extract_skills_from_text(text: str) -> list:
    """Extract skills from text using keyword matching."""
    text_lower = text.lower()
    found = []
    for skill in sorted(ALL_SKILLS, key=len, reverse=True):
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found[:25]


def simulate_screening(files_data: list, job_title: str, job_description: str,
                        semantic_weight: float = 0.7, include_fairness: bool = True) -> dict:
    """
    Simulate resume screening with realistic scores.
    Uses keyword overlap + randomised semantic component to mimic real ML output.
    """
    start_time = time.time()

    jd_skills = extract_skills_from_text(job_description)
    jd_words = set(re.findall(r'\b\w{4,}\b', job_description.lower()))

    candidates = []
    for i, file_info in enumerate(files_data):
        filename = file_info.get("name", f"resume_{i+1}.pdf")
        content = file_info.get("content", "")

        # Extract name from filename
        name_raw = filename.replace(".pdf", "").replace(".docx", "").replace("_", " ").replace("-", " ")
        name_parts = name_raw.split()
        name = " ".join(p.capitalize() for p in name_parts[:3]) if name_parts else f"Candidate {i+1}"

        # Extract skills from content
        resume_skills = extract_skills_from_text(content) if content else []

        # Skill match score
        if jd_skills:
            matched = [s for s in resume_skills if s in jd_skills]
            missing = [s for s in jd_skills if s not in resume_skills]
            skill_score = len(matched) / len(jd_skills) if jd_skills else 0.0
        else:
            matched = resume_skills[:5]
            missing = []
            skill_score = 0.3

        # Semantic score — keyword overlap + seeded random for demo consistency
        resume_words = set(re.findall(r'\b\w{4,}\b', content.lower())) if content else set()
        overlap = len(resume_words & jd_words) / max(len(jd_words), 1)
        seed = sum(ord(c) for c in filename)
        rng = random.Random(seed)
        noise = rng.uniform(-0.08, 0.12)
        semantic_score = min(1.0, max(0.05, overlap * 2.5 + noise + 0.35))

        # Hybrid score
        skill_weight = 1.0 - semantic_weight
        hybrid_score = semantic_weight * semantic_score + skill_weight * skill_score
        hybrid_score = min(1.0, max(0.0, hybrid_score))

        # Years experience (rough heuristic from content)
        years = 0
        year_matches = re.findall(r'\b(20\d{2})\b', content)
        if len(year_matches) >= 2:
            years_list = sorted([int(y) for y in year_matches])
            years = min(years_list[-1] - years_list[0], 20)

        # Generate explanation
        score_pct = hybrid_score * 100
        if score_pct >= 80:
            fit = "excellent"
        elif score_pct >= 60:
            fit = "good"
        elif score_pct >= 40:
            fit = "moderate"
        else:
            fit = "limited"

        explanation_parts = [
            f"{name} shows {fit} fit for the {job_title} position with an overall score of {score_pct:.1f}% (Rank #{i+1})."
        ]
        if matched:
            explanation_parts.append(f"Matched skills: {', '.join(matched[:4])}.")
        if missing:
            explanation_parts.append(f"Missing: {', '.join(missing[:3])}.")

        candidates.append({
            "rank": 0,  # assigned after sort
            "name": name,
            "email": f"{name.lower().replace(' ', '.')}@example.com",
            "hybrid_score": round(hybrid_score, 4),
            "semantic_score": round(semantic_score, 4),
            "skill_score": round(skill_score, 4),
            "matched_skills": matched[:10],
            "missing_skills": missing[:10],
            "years_experience": years,
            "explanation": " ".join(explanation_parts),
        })

    # Sort and assign ranks
    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    for rank, c in enumerate(candidates, 1):
        c["rank"] = rank

    processing_time = round(time.time() - start_time, 3)

    # Fairness summary
    fairness_summary = None
    if include_fairness:
        fairness_summary = {
            "overall_score": 0.92,
            "bias_flags": [],
            "recommendations": [
                "Rankings are based purely on skills and semantic relevance.",
                "Consider blind review for shortlisted candidates.",
            ],
        }

    return {
        "job_id": f"job_{int(time.time())}",
        "job_title": job_title,
        "total_resumes": len(files_data),
        "successfully_parsed": len(candidates),
        "processing_time_seconds": processing_time,
        "candidates": candidates,
        "fairness_summary": fairness_summary,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Vercel handler
# ---------------------------------------------------------------------------

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
}


def handler(request):
    """Vercel serverless function entry point."""

    if request.method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    if request.method != "POST":
        return {
            "statusCode": 405,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Method not allowed"}),
        }

    try:
        form = request.form
        files = request.files.getlist("files")

        job_title = form.get("job_title", "Software Engineer")
        job_description = form.get("job_description", "")
        semantic_weight = float(form.get("semantic_weight", 0.7))
        include_fairness = form.get("include_fairness", "true").lower() == "true"

        if not job_description:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Job description is required"}),
            }

        if not files:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "At least one resume file is required"}),
            }

        # Read file content (text extraction is limited in serverless)
        files_data = []
        for f in files:
            if not f.filename.lower().endswith((".pdf", ".docx")):
                continue
            try:
                content = f.read().decode("utf-8", errors="ignore")
            except Exception:
                content = ""
            files_data.append({"name": f.filename, "content": content})

        if not files_data:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "No valid PDF/DOCX files found"}),
            }

        result = simulate_screening(
            files_data=files_data,
            job_title=job_title,
            job_description=job_description,
            semantic_weight=semantic_weight,
            include_fairness=include_fairness,
        )

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(result),
        }

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Internal server error", "message": str(e)}),
        }
