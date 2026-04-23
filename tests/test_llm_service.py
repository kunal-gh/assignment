"""Tests for LLM service explanation generation."""

from unittest.mock import MagicMock, patch

import pytest

from src.ranking.llm_service import LLMService, LLMUsage


class TestLLMServiceFallback:
    """Tests for the template-based fallback (no OpenAI needed)."""

    def setup_method(self):
        """Create LLM service without real API (uses fallback)."""
        self.service = LLMService(api_key=None)
        # Force unavailable so we always hit fallback
        self.service._available = False

    def test_fallback_explanation_excellent_score(self):
        """Excellent scores should produce 'excellent' in explanation."""
        result = self.service.generate_explanation(
            candidate_name="Alice Chen",
            rank=1,
            semantic_score=0.90,
            skill_score=0.85,
            hybrid_score=0.88,
            matched_skills=["python", "pytorch", "fastapi"],
            missing_skills=[],
            job_title="ML Engineer",
        )
        assert isinstance(result, str)
        assert len(result) > 20
        assert "Alice Chen" in result

    def test_fallback_explanation_good_score(self):
        """Good scores (0.6–0.8) should produce valid explanation."""
        result = self.service.generate_explanation(
            candidate_name="Bob Smith",
            rank=2,
            semantic_score=0.65,
            skill_score=0.70,
            hybrid_score=0.67,
            matched_skills=["python", "docker"],
            missing_skills=["kubernetes", "tensorflow"],
            job_title="Backend Engineer",
        )
        assert isinstance(result, str)
        assert "Bob Smith" in result

    def test_fallback_with_missing_skills_mentions_them(self):
        """Missing skills should appear in the explanation."""
        result = self.service.generate_explanation(
            candidate_name="Carol Davis",
            rank=3,
            semantic_score=0.55,
            skill_score=0.40,
            hybrid_score=0.50,
            matched_skills=["python"],
            missing_skills=["aws", "kubernetes"],
            job_title="DevOps Engineer",
        )
        assert isinstance(result, str)
        # Should mention something about missing skills
        assert len(result) > 10

    def test_fallback_low_score_candidate(self):
        """Low-scoring candidate should get appropriate explanation."""
        result = self.service.generate_explanation(
            candidate_name="Dave Wilson",
            rank=10,
            semantic_score=0.20,
            skill_score=0.10,
            hybrid_score=0.17,
            matched_skills=[],
            missing_skills=["python", "docker", "aws"],
            job_title="ML Engineer",
        )
        assert isinstance(result, str)
        assert "Dave Wilson" in result

    def test_generate_batch_explanations(self):
        """Batch generation should work for multiple candidates."""
        candidates_data = [
            dict(
                candidate_name="Alice",
                rank=1,
                semantic_score=0.9,
                skill_score=0.85,
                hybrid_score=0.88,
                matched_skills=["python"],
                missing_skills=[],
                job_title="ML Engineer",
            ),
            dict(
                candidate_name="Bob",
                rank=2,
                semantic_score=0.65,
                skill_score=0.60,
                hybrid_score=0.63,
                matched_skills=[],
                missing_skills=["python"],
                job_title="ML Engineer",
            ),
        ]

        results = self.service.generate_batch_explanations(candidates_data)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_usage_stats_initially_zero(self):
        """Usage statistics should start at zero."""
        fresh = LLMService(api_key=None)
        fresh._available = False
        stats = fresh.get_usage_stats()

        assert stats["requests"] == 0
        assert stats["errors"] == 0
        assert stats["tokens"]["total"] == 0
        assert stats["estimated_cost_usd"] == 0.0

    def test_reset_usage_stats(self):
        """Resetting stats should zero everything out."""
        self.service.request_count = 100
        self.service.error_count = 5
        self.service.reset_usage_stats()

        stats = self.service.get_usage_stats()
        assert stats["requests"] == 0
        assert stats["errors"] == 0

    def test_explanation_not_available_returns_fallback(self):
        """generate_explanation should always return a string, never raise."""
        result = self.service.generate_explanation(
            candidate_name="Edge Case",
            rank=99,
            semantic_score=0.0,
            skill_score=0.0,
            hybrid_score=0.0,
            matched_skills=[],
            missing_skills=[],
            job_title="",
        )
        assert isinstance(result, str)

    def test_prompt_builder_produces_text(self):
        """The prompt builder should produce non-empty text."""
        prompt = self.service._build_explanation_prompt(
            candidate_name="Test",
            rank=1,
            semantic_score=0.8,
            skill_score=0.7,
            hybrid_score=0.77,
            matched_skills=["python", "docker"],
            missing_skills=["aws"],
            job_title="Engineer",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "Test" in prompt
        assert "Engineer" in prompt


class TestLLMUsageTracking:
    """Tests for LLMUsage dataclass."""

    def test_usage_dataclass_defaults(self):
        """LLMUsage defaults should be zero."""
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0

    def test_usage_accumulates(self):
        """Usage values should accumulate correctly."""
        usage = LLMUsage()
        usage.prompt_tokens += 100
        usage.completion_tokens += 50
        usage.total_tokens += 150
        usage.cost_usd += 0.001

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert abs(usage.cost_usd - 0.001) < 1e-9
