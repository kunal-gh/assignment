"""LLM service for generating ranking explanations."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. Install with: pip install openai")


@dataclass
class LLMUsage:
    """Track LLM API usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class LLMService:
    """Service for generating explanations using LLM APIs."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 200,
                 temperature: float = 0.3,
                 rate_limit_delay: float = 1.0):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            rate_limit_delay: Delay between requests to avoid rate limits
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        
        # Usage tracking
        self.usage = LLMUsage()
        self.request_count = 0
        self.error_count = 0
        
        # Initialize OpenAI client
        self.client = None
        self._available = False
        
        if OPENAI_AVAILABLE:
            try:
                import openai as openai_module
                self.client = openai_module.OpenAI(api_key=api_key) if api_key else openai_module.OpenAI()
                self._available = True
                logger.info(f"LLMService initialised with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialise OpenAI client: {str(e)}")
                self._available = False
        else:
            logger.warning("OpenAI not available - explanations will use fallback templates")

    @property
    def available(self) -> bool:
        """Check if LLM service is available."""
        return self._available

    def generate_explanation(self,
                             candidate_name: str,
                             rank: int,
                             semantic_score: float,
                             skill_score: float,
                             hybrid_score: float,
                             matched_skills: List[str],
                             missing_skills: List[str],
                             job_title: str) -> str:
        """
        Generate ranking explanation using LLM.

        Falls back to template-based explanation when OpenAI is unavailable,
        so this method is always safe to call regardless of API status.
        """
        if not self._available:
            return self._generate_fallback_explanation(
                candidate_name, rank, semantic_score, skill_score,
                hybrid_score, matched_skills, missing_skills, job_title
            )

        try:
            prompt = self._build_explanation_prompt(
                candidate_name, rank, semantic_score, skill_score,
                hybrid_score, matched_skills, missing_skills, job_title
            )

            response = self._make_api_request(prompt)

            if response:
                self._update_usage(response)
                content = response.choices[0].message.content
                return content.strip() if content else self._generate_fallback_explanation(
                    candidate_name, rank, semantic_score, skill_score,
                    hybrid_score, matched_skills, missing_skills, job_title
                )
            return self._generate_fallback_explanation(
                candidate_name, rank, semantic_score, skill_score,
                hybrid_score, matched_skills, missing_skills, job_title
            )

        except Exception as e:
            logger.error(f"Error generating LLM explanation: {str(e)}")
            self.error_count += 1
            return self._generate_fallback_explanation(
                candidate_name, rank, semantic_score, skill_score,
                hybrid_score, matched_skills, missing_skills, job_title
            )

    def generate_batch_explanations(self, candidates_data: List[Dict[str, Any]]) -> List[str]:
        """Generate explanations for multiple candidates efficiently."""
        explanations = []
        for i, data in enumerate(candidates_data):
            if i > 0:
                time.sleep(self.rate_limit_delay)
            explanation = self.generate_explanation(**data)
            explanations.append(explanation)
        return explanations

    def _build_explanation_prompt(self,
                                  candidate_name: str,
                                  rank: int,
                                  semantic_score: float,
                                  skill_score: float,
                                  hybrid_score: float,
                                  matched_skills: List[str],
                                  missing_skills: List[str],
                                  job_title: str) -> str:
        """Build prompt for explanation generation."""
        matched_str = ", ".join(matched_skills[:5]) if matched_skills else "none"
        missing_str = ", ".join(missing_skills[:5]) if missing_skills else "none"
        return (
            f"Generate a concise, professional explanation for why a candidate ranked #{rank} "
            f"for a {job_title} position.\n\n"
            f"Candidate: {candidate_name}\n"
            f"Overall Score: {hybrid_score:.1%} (Rank #{rank})\n"
            f"Semantic Match: {semantic_score:.1%}\n"
            f"Skill Match: {skill_score:.1%}\n"
            f"Matched Skills: {matched_str}\n"
            f"Missing Skills: {missing_str}\n\n"
            "Write a 2-3 sentence explanation that summarises their overall fit, "
            "highlights key strengths, and mentions significant gaps (if any). "
            "Keep it professional, specific, and actionable."
        )

    def _make_api_request(self, prompt: str) -> Optional[Any]:
        """Make API request to OpenAI using the v1.0+ client."""
        try:
            self.request_count += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst providing candidate evaluation explanations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30,
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return None

    def _update_usage(self, response: Any) -> None:
        """Update usage statistics from API response."""
        try:
            usage = response.usage
            if usage:
                self.usage.prompt_tokens += usage.prompt_tokens
                self.usage.completion_tokens += usage.completion_tokens
                self.usage.total_tokens += usage.total_tokens
                if "gpt-3.5-turbo" in self.model:
                    self.usage.cost_usd += (usage.prompt_tokens * 0.0015 + usage.completion_tokens * 0.002) / 1000
                elif "gpt-4" in self.model:
                    self.usage.cost_usd += (usage.prompt_tokens * 0.03 + usage.completion_tokens * 0.06) / 1000
        except Exception as e:
            logger.debug(f"Could not update usage stats: {str(e)}")
    
    def _generate_fallback_explanation(self, 
                                     candidate_name: str,
                                     rank: int,
                                     semantic_score: float,
                                     skill_score: float,
                                     hybrid_score: float,
                                     matched_skills: List[str],
                                     missing_skills: List[str],
                                     job_title: str) -> str:
        """Generate explanation using template when LLM is not available."""
        
        # Determine overall assessment
        if hybrid_score >= 0.8:
            assessment = "excellent"
        elif hybrid_score >= 0.6:
            assessment = "good"
        elif hybrid_score >= 0.4:
            assessment = "moderate"
        else:
            assessment = "limited"
        
        explanation_parts = []
        
        # Overall assessment
        explanation_parts.append(
            f"{candidate_name} shows {assessment} fit for the {job_title} position "
            f"with an overall score of {hybrid_score:.1%} (Rank #{rank})."
        )
        
        # Strengths
        strengths = []
        if semantic_score >= 0.7:
            strengths.append("strong relevant experience")
        if skill_score >= 0.7:
            strengths.append("excellent skill alignment")
        if matched_skills:
            key_skills = matched_skills[:3]
            strengths.append(f"proficiency in {', '.join(key_skills)}")
        
        if strengths:
            explanation_parts.append(f"Key strengths include {' and '.join(strengths)}.")
        
        # Areas for improvement
        if missing_skills:
            top_missing = missing_skills[:3]
            explanation_parts.append(
                f"Development opportunities exist in {', '.join(top_missing)}."
            )
        
        return " ".join(explanation_parts)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "available": self._available,
            "model": self.model,
            "requests": self.request_count,
            "errors": self.error_count,
            "tokens": {
                "prompt": self.usage.prompt_tokens,
                "completion": self.usage.completion_tokens,
                "total": self.usage.total_tokens
            },
            "estimated_cost_usd": round(self.usage.cost_usd, 4),
            "error_rate": self.error_count / max(1, self.request_count)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage = LLMUsage()
        self.request_count = 0
        self.error_count = 0
        logger.info("Usage statistics reset")
    
    def set_rate_limit(self, delay: float):
        """Update rate limit delay."""
        self.rate_limit_delay = max(0.1, delay)
        logger.info(f"Rate limit delay set to {self.rate_limit_delay}s")
    
    def test_connection(self) -> bool:
        """Test if the LLM service is working."""
        if not self._available:
            return False
        
        try:
            test_response = self._make_api_request("Test connection. Respond with 'OK'.")
            return test_response is not None and "OK" in test_response.upper()
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False