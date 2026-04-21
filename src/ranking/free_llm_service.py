"""Free LLM service using Hugging Face Inference API."""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class FreeLLMService:
    """
    Free LLM service using Hugging Face Inference API.

    Uses Qwen2.5-72B-Instruct model (free tier: 30k tokens/month).
    Falls back to template-based explanations if API fails or quota exceeded.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize free LLM service.

        Args:
            api_key: Hugging Face API key (optional, uses env var if not provided)
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.model = "Qwen/Qwen2.5-72B-Instruct"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.max_retries = 2
        self.timeout = 30.0

        # Track if API is available
        self.api_available = bool(self.api_key)

        if not self.api_available:
            logger.warning("No Hugging Face API key found. Using template-based explanations only.")
        else:
            logger.info(f"FreeLLMService initialized with model: {self.model}")

    async def generate_explanation(
        self,
        candidate_name: str,
        rank: int,
        hybrid_score: float,
        semantic_score: float,
        skill_score: float,
        matched_skills: list,
        missing_skills: list,
        job_title: str,
        years_experience: int,
    ) -> str:
        """
        Generate ranking explanation using free LLM or template fallback.

        Args:
            candidate_name: Candidate's name
            rank: Ranking position
            hybrid_score: Overall match score
            semantic_score: Semantic similarity score
            skill_score: Skill matching score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            job_title: Job title
            years_experience: Years of experience

        Returns:
            Human-readable explanation string
        """
        # Try LLM if API is available
        if self.api_available:
            try:
                llm_explanation = await self._call_huggingface_api(
                    candidate_name=candidate_name,
                    rank=rank,
                    hybrid_score=hybrid_score,
                    semantic_score=semantic_score,
                    skill_score=skill_score,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    job_title=job_title,
                    years_experience=years_experience,
                )

                if llm_explanation:
                    return llm_explanation

            except Exception as e:
                logger.warning(f"LLM API call failed: {str(e)}. Using template fallback.")

        # Fallback to template-based explanation
        return self._generate_template_explanation(
            candidate_name=candidate_name,
            rank=rank,
            hybrid_score=hybrid_score,
            semantic_score=semantic_score,
            skill_score=skill_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            job_title=job_title,
            years_experience=years_experience,
        )

    async def _call_huggingface_api(
        self,
        candidate_name: str,
        rank: int,
        hybrid_score: float,
        semantic_score: float,
        skill_score: float,
        matched_skills: list,
        missing_skills: list,
        job_title: str,
        years_experience: int,
    ) -> Optional[str]:
        """Call Hugging Face Inference API."""

        # Construct prompt
        prompt = self._build_prompt(
            candidate_name=candidate_name,
            rank=rank,
            hybrid_score=hybrid_score,
            semantic_score=semantic_score,
            skill_score=skill_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            job_title=job_title,
            years_experience=years_experience,
        )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(self.api_url, headers=headers, json=payload)

                    if response.status_code == 200:
                        result = response.json()

                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            generated_text = result.get("generated_text", "")
                        else:
                            generated_text = ""

                        if generated_text:
                            # Clean up the response
                            explanation = generated_text.strip()
                            # Remove any prompt repetition
                            if "Explanation:" in explanation:
                                explanation = explanation.split("Explanation:")[-1].strip()
                            return explanation

                    elif response.status_code == 503:
                        # Model is loading, wait and retry
                        logger.info(f"Model loading, attempt {attempt + 1}/{self.max_retries}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                            continue

                    else:
                        logger.warning(f"HF API returned status {response.status_code}: {response.text}")
                        break

                except httpx.TimeoutException:
                    logger.warning(f"HF API timeout, attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        continue
                    break

                except Exception as e:
                    logger.error(f"HF API error: {str(e)}")
                    break

        return None

    def _build_prompt(
        self,
        candidate_name: str,
        rank: int,
        hybrid_score: float,
        semantic_score: float,
        skill_score: float,
        matched_skills: list,
        missing_skills: list,
        job_title: str,
        years_experience: int,
    ) -> str:
        """Build prompt for LLM."""

        matched_str = ", ".join(matched_skills[:5]) if matched_skills else "none"
        missing_str = ", ".join(missing_skills[:3]) if missing_skills else "none"

        prompt = f"""You are an expert recruiter. Write a concise 2-3 sentence explanation for why this candidate ranked #{rank} for a {job_title} position.

Candidate: {candidate_name}
Overall Score: {hybrid_score:.1%}
Semantic Match: {semantic_score:.1%}
Skill Match: {skill_score:.1%}
Experience: {years_experience} years
Matched Skills: {matched_str}
Missing Skills: {missing_str}

Explanation:"""

        return prompt

    def _generate_template_explanation(
        self,
        candidate_name: str,
        rank: int,
        hybrid_score: float,
        semantic_score: float,
        skill_score: float,
        matched_skills: list,
        missing_skills: list,
        job_title: str,
        years_experience: int,
    ) -> str:
        """Generate template-based explanation (fallback)."""

        explanation_parts = []

        # Overall assessment
        if hybrid_score >= 0.8:
            assessment = "excellent fit"
        elif hybrid_score >= 0.6:
            assessment = "strong candidate"
        elif hybrid_score >= 0.4:
            assessment = "potential match"
        else:
            assessment = "limited alignment"

        explanation_parts.append(
            f"{candidate_name} ranks #{rank} with {hybrid_score:.1%} overall match - {assessment} for the {job_title} position."
        )

        # Semantic analysis
        if semantic_score >= 0.7:
            explanation_parts.append(
                f"Strong semantic alignment ({semantic_score:.1%}) indicates relevant experience and background."
            )
        elif semantic_score >= 0.5:
            explanation_parts.append(f"Moderate semantic match ({semantic_score:.1%}) shows some relevant experience.")
        else:
            explanation_parts.append(f"Limited semantic alignment ({semantic_score:.1%}) suggests different background focus.")

        # Skill analysis
        if matched_skills:
            skill_list = ", ".join(matched_skills[:3])
            if len(matched_skills) > 3:
                skill_list += f" and {len(matched_skills) - 3} more"
            explanation_parts.append(f"Possesses key skills: {skill_list}.")

        if missing_skills:
            missing_list = ", ".join(missing_skills[:2])
            if len(missing_skills) > 2:
                missing_list += f" and {len(missing_skills) - 2} more"
            explanation_parts.append(f"Development areas: {missing_list}.")

        # Experience
        if years_experience >= 5:
            explanation_parts.append(f"Brings {years_experience} years of solid experience.")
        elif years_experience >= 2:
            explanation_parts.append(f"Has {years_experience} years of relevant experience.")
        else:
            explanation_parts.append(f"Early career with {years_experience} year(s) experience.")

        return " ".join(explanation_parts)


# Synchronous wrapper for backward compatibility
class FreeLLMServiceSync:
    """Synchronous wrapper for FreeLLMService."""

    def __init__(self, api_key: Optional[str] = None):
        self.service = FreeLLMService(api_key)

    def generate_explanation(self, **kwargs) -> str:
        """Generate explanation synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.service.generate_explanation(**kwargs))
