"""API endpoint tests using FastAPI TestClient."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We patch the heavy ML components before importing the app
with patch("src.parsers.resume_parser.ResumeParser"), patch("src.embeddings.embedding_generator.EmbeddingGenerator"), patch(
    "src.ranking.ranking_engine.RankingEngine"
):
    from api import app

client = TestClient(app)


class TestHealthEndpoints:

    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "timestamp" in body


class TestModelsEndpoint:

    def test_models_returns_list(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 3

    def test_default_model_flagged(self):
        resp = client.get("/models")
        models = resp.json()["models"]
        defaults = [m for m in models if m.get("is_default")]
        assert len(defaults) == 1
        assert defaults[0]["name"] == "all-MiniLM-L6-v2"


class TestMetricsEndpoint:

    def test_metrics_returns_counts(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "cached_jobs" in body
        assert "timestamp" in body


class TestAnalyzeJobDescription:

    def test_analyze_valid_jd(self):
        payload = {
            "title": "ML Engineer",
            "description": (
                "We need a Python developer with experience in machine learning, "
                "Docker, Kubernetes, and FastAPI. Must know PyTorch or TensorFlow."
            ),
        }
        resp = client.post("/analyze/jd", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "ML Engineer"
        assert "extracted_required_skills" in body
        assert "tip" in body

    def test_analyze_empty_description_rejected(self):
        payload = {"title": "Engineer", "description": "x"}
        resp = client.post("/analyze/jd", json=payload)
        # min_length=10 on description → 422
        assert resp.status_code == 422

    def test_analyze_missing_title_rejected(self):
        payload = {"description": "Valid description with enough text here."}
        resp = client.post("/analyze/jd", json=payload)
        assert resp.status_code == 422


class TestResultsEndpoint:

    def test_nonexistent_job_id_returns_404(self):
        resp = client.get("/results/does-not-exist-12345")
        assert resp.status_code == 404

    def test_csv_export_nonexistent_returns_404(self):
        resp = client.get("/results/nonexistent/export/csv")
        assert resp.status_code == 404
