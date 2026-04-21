"""Main ranking engine that combines semantic and skill-based scoring."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from ..models.resume import ResumeData
from ..models.job import JobDescription
from ..models.ranking import RankedCandidate, BatchProcessingResult
from ..embeddings.embedding_generator import EmbeddingGenerator
from .skill_matcher import SkillMatcher
from .fairness_checker import FairnessChecker

logger = logging.getLogger(__name__)


class RankingEngine:
    """Main ranking engine that combines semantic similarity with skill matching."""
    
    def __init__(self, 
                 semantic_weight: float = 0.7,
                 skill_weight: float = 0.3,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize ranking engine.
        
        Args:
            semantic_weight: Weight for semantic similarity score (0-1)
            skill_weight: Weight for skill matching score (0-1)
            embedding_generator: Pre-initialized embedding generator
        """
        # Validate weights
        if not (0 <= semantic_weight <= 1 and 0 <= skill_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(semantic_weight + skill_weight - 1.0) > 1e-6:
            raise ValueError("Semantic and skill weights must sum to 1.0")
        
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight
        
        # Initialize components
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.skill_matcher = SkillMatcher()
        self.fairness_checker = FairnessChecker()
        
        logger.info(f"RankingEngine initialized with weights: semantic={semantic_weight}, skill={skill_weight}")
    
    def calculate_semantic_score(self, resume_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors, normalized to [0, 1].

        Handles edge cases:
        - Zero vectors return 0.0
        - Identical vectors return 1.0
        - Orthogonal vectors return 0.0

        Args:
            resume_embedding: Embedding vector for the resume
            job_embedding: Embedding vector for the job description

        Returns:
            Cosine similarity score in [0.0, 1.0]
        """
        if not isinstance(resume_embedding, np.ndarray):
            resume_embedding = np.array(resume_embedding, dtype=np.float64)
        if not isinstance(job_embedding, np.ndarray):
            job_embedding = np.array(job_embedding, dtype=np.float64)

        # Dimension mismatch
        if resume_embedding.shape != job_embedding.shape:
            logger.error(
                f"Embedding dimension mismatch: {resume_embedding.shape} vs {job_embedding.shape}"
            )
            return 0.0

        norm_resume = np.linalg.norm(resume_embedding)
        norm_job = np.linalg.norm(job_embedding)

        # Zero vector edge case
        if norm_resume == 0.0 or norm_job == 0.0:
            return 0.0

        similarity = np.dot(resume_embedding, job_embedding) / (norm_resume * norm_job)

        # Clamp to [0, 1] — cosine similarity for normalized embeddings is already
        # in [-1, 1]; we clamp negatives to 0 since a negative similarity has no
        # meaningful interpretation in the context of resume matching.
        return float(max(0.0, min(1.0, similarity)))

    def calculate_hybrid_score(self, resume: ResumeData, job_desc: JobDescription) -> Dict[str, float]:
        """
        Calculate combined semantic + skill matching score.
        
        Args:
            resume: Resume data
            job_desc: Job description
            
        Returns:
            Dictionary with score breakdown
        """
        try:
            # Ensure embeddings exist
            if resume.embedding is None:
                resume.embedding = self.embedding_generator.encode_resume(resume)
            
            if job_desc.embedding is None:
                job_desc.embedding = self.embedding_generator.encode_job_description(job_desc)
            
            # Calculate semantic similarity
            semantic_score = self.calculate_semantic_score(
                resume.embedding, job_desc.embedding
            )
            
            # Calculate skill matching score
            skill_score = self.skill_matcher.calculate_skill_match(
                resume.skills, job_desc.required_skills, job_desc.preferred_skills
            )
            
            # Calculate hybrid score
            hybrid_score = (self.semantic_weight * semantic_score + 
                          self.skill_weight * skill_score)
            
            # Ensure scores are in valid range
            semantic_score = max(0.0, min(1.0, semantic_score))
            skill_score = max(0.0, min(1.0, skill_score))
            hybrid_score = max(0.0, min(1.0, hybrid_score))
            
            return {
                'semantic_score': semantic_score,
                'skill_score': skill_score,
                'hybrid_score': hybrid_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating hybrid score: {str(e)}")
            return {
                'semantic_score': 0.0,
                'skill_score': 0.0,
                'hybrid_score': 0.0
            }
    
    def rank_candidates(self, resumes: List[ResumeData], job_desc: JobDescription) -> List[RankedCandidate]:
        """
        Rank all candidates against job requirements.
        
        Args:
            resumes: List of resume data
            job_desc: Job description
            
        Returns:
            List of ranked candidates sorted by score
        """
        if not resumes:
            logger.warning("No resumes provided for ranking")
            return []
        
        logger.info(f"Ranking {len(resumes)} candidates against job: {job_desc.title}")
        
        candidates = []
        
        # Generate embeddings in batch for efficiency
        self._ensure_embeddings(resumes, job_desc)
        
        # Calculate scores for each candidate
        for resume in resumes:
            try:
                scores = self.calculate_hybrid_score(resume, job_desc)
                
                candidate = RankedCandidate(
                    resume=resume,
                    semantic_score=scores['semantic_score'],
                    skill_score=scores['skill_score'],
                    hybrid_score=scores['hybrid_score'],
                    rank=0  # Will be set after sorting
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.error(f"Error processing candidate {resume.candidate_id}: {str(e)}")
                # Add candidate with zero scores to maintain list consistency
                error_candidate = RankedCandidate(
                    resume=resume,
                    semantic_score=0.0,
                    skill_score=0.0,
                    hybrid_score=0.0,
                    rank=0
                )
                candidates.append(error_candidate)
        
        # Sort by hybrid score (descending) and assign ranks
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Handle ties with secondary sorting criteria
        candidates = self._handle_ties(candidates)
        
        # Assign ranks
        for i, candidate in enumerate(candidates, 1):
            candidate.rank = i
        
        logger.info(f"Ranking completed. Top score: {candidates[0].hybrid_score:.3f}")
        
        return candidates
    
    def rank_with_explanations(self, resumes: List[ResumeData], job_desc: JobDescription) -> List[RankedCandidate]:
        """
        Rank candidates and generate explanations for each ranking.
        
        Args:
            resumes: List of resume data
            job_desc: Job description
            
        Returns:
            List of ranked candidates with explanations
        """
        # Get basic rankings
        candidates = self.rank_candidates(resumes, job_desc)
        
        # Generate explanations
        for candidate in candidates:
            candidate.explanation = self.explain_ranking(candidate, job_desc)
        
        return candidates
    
    def explain_ranking(self, candidate: RankedCandidate, job_desc: JobDescription) -> str:
        """
        Generate human-readable ranking explanation.
        
        Args:
            candidate: Ranked candidate
            job_desc: Job description
            
        Returns:
            Explanation string
        """
        try:
            resume = candidate.resume
            
            # Get skill analysis
            skill_analysis = self.skill_matcher.analyze_skill_match(
                resume.skills, job_desc.required_skills, job_desc.preferred_skills
            )
            
            # Build explanation
            explanation_parts = []
            
            # Overall score
            explanation_parts.append(
                f"Overall match score: {candidate.hybrid_score:.1%} "
                f"(Rank #{candidate.rank})"
            )
            
            # Semantic similarity
            if candidate.semantic_score > 0.8:
                explanation_parts.append(
                    f"Excellent semantic match ({candidate.semantic_score:.1%}) - "
                    "resume content strongly aligns with job requirements."
                )
            elif candidate.semantic_score > 0.6:
                explanation_parts.append(
                    f"Good semantic match ({candidate.semantic_score:.1%}) - "
                    "resume shows relevant experience and background."
                )
            else:
                explanation_parts.append(
                    f"Limited semantic match ({candidate.semantic_score:.1%}) - "
                    "resume may not closely align with job requirements."
                )
            
            # Skill matching
            if skill_analysis['matched_required']:
                explanation_parts.append(
                    f"Matches {len(skill_analysis['matched_required'])} required skills: "
                    f"{', '.join(skill_analysis['matched_required'][:3])}{'...' if len(skill_analysis['matched_required']) > 3 else ''}"
                )
            
            if skill_analysis['matched_preferred']:
                explanation_parts.append(
                    f"Also has {len(skill_analysis['matched_preferred'])} preferred skills: "
                    f"{', '.join(skill_analysis['matched_preferred'][:3])}{'...' if len(skill_analysis['matched_preferred']) > 3 else ''}"
                )
            
            if skill_analysis['missing_required']:
                explanation_parts.append(
                    f"Missing {len(skill_analysis['missing_required'])} required skills: "
                    f"{', '.join(skill_analysis['missing_required'][:3])}{'...' if len(skill_analysis['missing_required']) > 3 else ''}"
                )
            
            # Experience level
            years_exp = resume.get_years_of_experience()
            if job_desc.matches_experience_level(years_exp):
                explanation_parts.append(
                    f"Experience level ({years_exp} years) matches job requirements."
                )
            else:
                explanation_parts.append(
                    f"Experience level ({years_exp} years) may not match {job_desc.experience_level} level requirement."
                )
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Score: {candidate.hybrid_score:.1%} (Rank #{candidate.rank})"
    
    def process_batch(self, resumes: List[ResumeData], job_desc: JobDescription, 
                     include_fairness: bool = True) -> BatchProcessingResult:
        """
        Process a batch of resumes with comprehensive analysis.
        
        Args:
            resumes: List of resume data
            job_desc: Job description
            include_fairness: Whether to include fairness analysis
            
        Returns:
            Batch processing result with rankings and analysis
        """
        start_time = datetime.now()
        
        try:
            # Rank candidates with explanations
            ranked_candidates = self.rank_with_explanations(resumes, job_desc)
            
            # Generate fairness report if requested
            fairness_report = None
            if include_fairness and ranked_candidates:
                fairness_report = self.fairness_checker.generate_fairness_report(
                    ranked_candidates, top_k=min(10, len(ranked_candidates))
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Count successful parses (resumes with valid data)
            successfully_parsed = sum(
                1 for resume in resumes 
                if resume.contact_info and resume.contact_info.name != "Unknown"
            )
            
            result = BatchProcessingResult(
                job_id=job_desc.job_id,
                total_resumes=len(resumes),
                successfully_parsed=successfully_parsed,
                failed_parses=len(resumes) - successfully_parsed,
                ranked_candidates=ranked_candidates,
                fairness_report=fairness_report,
                processing_time=processing_time
            )
            
            logger.info(f"Batch processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BatchProcessingResult(
                job_id=job_desc.job_id,
                total_resumes=len(resumes),
                successfully_parsed=0,
                failed_parses=len(resumes),
                processing_time=processing_time,
                errors=[str(e)]
            )
    
    def _ensure_embeddings(self, resumes: List[ResumeData], job_desc: JobDescription):
        """Ensure all resumes and job description have embeddings."""
        # Generate job description embedding if needed
        if job_desc.embedding is None:
            job_desc.embedding = self.embedding_generator.encode_job_description(job_desc)
        
        # Find resumes without embeddings
        resumes_without_embeddings = [r for r in resumes if r.embedding is None]
        
        if resumes_without_embeddings:
            logger.info(f"Generating embeddings for {len(resumes_without_embeddings)} resumes")
            self.embedding_generator.batch_encode_resumes(resumes_without_embeddings)
    
    def _handle_ties(self, candidates: List[RankedCandidate]) -> List[RankedCandidate]:
        """Handle tied scores with secondary sorting criteria."""
        # Group by hybrid score
        score_groups = {}
        for candidate in candidates:
            score = round(candidate.hybrid_score, 6)  # Round to avoid floating point issues
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(candidate)
        
        # Sort each group by secondary criteria
        sorted_candidates = []
        for score in sorted(score_groups.keys(), reverse=True):
            group = score_groups[score]
            
            if len(group) > 1:
                # Sort by skill score first, then semantic score
                group.sort(key=lambda x: (x.skill_score, x.semantic_score), reverse=True)
            
            sorted_candidates.extend(group)
        
        return sorted_candidates
    
    def get_ranking_stats(self, candidates: List[RankedCandidate]) -> Dict[str, Any]:
        """Get statistics about the ranking results."""
        if not candidates:
            return {}
        
        scores = [c.hybrid_score for c in candidates]
        semantic_scores = [c.semantic_score for c in candidates]
        skill_scores = [c.skill_score for c in candidates]
        
        return {
            'total_candidates': len(candidates),
            'hybrid_scores': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'semantic_scores': {
                'mean': np.mean(semantic_scores),
                'median': np.median(semantic_scores),
                'std': np.std(semantic_scores)
            },
            'skill_scores': {
                'mean': np.mean(skill_scores),
                'median': np.median(skill_scores),
                'std': np.std(skill_scores)
            },
            'weights': {
                'semantic_weight': self.semantic_weight,
                'skill_weight': self.skill_weight
            }
        }
    
    def update_weights(self, semantic_weight: float, skill_weight: float):
        """Update scoring weights."""
        if not (0 <= semantic_weight <= 1 and 0 <= skill_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(semantic_weight + skill_weight - 1.0) > 1e-6:
            raise ValueError("Semantic and skill weights must sum to 1.0")
        
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight
        
        logger.info(f"Updated weights: semantic={semantic_weight}, skill={skill_weight}")
    
    def get_top_candidates(self, candidates: List[RankedCandidate], n: int = 10) -> List[RankedCandidate]:
        """Get top N candidates from ranked list."""
        return candidates[:n] if candidates else []