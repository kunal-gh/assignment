"""LLM service for generating ranking explanations."""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

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
                if api_key:
                    openai.api_key = api_key
                
                # Test API availability with a simple request
                self.client = openai
                self._available = True
                logger.info(f"LLMService initialized with model: {model}")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
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
        
        Args:
            candidate_name: Name of the candidate
            rank: Candidate's rank
            semantic_score: Semantic similarity score
            skill_score: Skill matching score
            hybrid_score: Combined score
            matched_skills: Skills that match job requirements
            missing_skills: Skills missing from candidate
            job_title: Job title
            
        Returns:
            Generated explanation text
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
                return response.strip()
            else:
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
        """
        Generate explanations for multiple candidates efficiently.
        
        Args:
            candidates_data: List of candidate data dictionaries
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        for i, data in enumerate(candidates_data):
            if i > 0:  # Rate limiting
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
        
        matched_skills_str = ", ".join(matched_skills[:5]) if matched_skills else "none"
        missing_skills_str = ", ".join(missing_skills[:5]) if missing_skills else "none"
        
        prompt = f"""Generate a concise, professional explanation for why a candidate ranked #{rank} for a {job_title} position.

Candidate: {candidate_name}
Overall Score: {hybrid_score:.1%} (Rank #{rank})
Semantic Match: {semantic_score:.1%}
Skill Match: {skill_score:.1%}
Matched Skills: {matched_skills_str}
Missing Skills: {missing_skills_str}

Write a 2-3 sentence explanation that:
1. Summarizes their overall fit for the role
2. Highlights their key strengths
3. Mentions any significant gaps (if applicable)

Keep it professional, specific, and actionable. Focus on the most important factors that influenced their ranking."""
        
        return prompt
    
    def _make_api_request(self, prompt: str) -> Optional[str]:
        """Make API request to OpenAI."""
        try:
            self.request_count += 1
            
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst providing candidate evaluation explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            return None
    
    def _update_usage(self, response):
        """Update usage statistics from API response."""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                self.usage.prompt_tokens += usage.prompt_tokens
                self.usage.completion_tokens += usage.completion_tokens
                self.usage.total_tokens += usage.total_tokens
                
                # Estimate cost (approximate pricing for gpt-3.5-turbo)
                if "gpt-3.5-turbo" in self.model:
                    prompt_cost = usage.prompt_tokens * 0.0015 / 1000
                    completion_cost = usage.completion_tokens * 0.002 / 1000
                    self.usage.cost_usd += prompt_cost + completion_cost
                elif "gpt-4" in self.model:
                    prompt_cost = usage.prompt_tokens * 0.03 / 1000
                    completion_cost = usage.completion_tokens * 0.06 / 1000
                    self.usage.cost_usd += prompt_cost + completion_cost
                    
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