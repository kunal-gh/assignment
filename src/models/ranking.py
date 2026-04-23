"""Ranking and fairness data models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resume import ResumeData


@dataclass
class RankedCandidate:
    """Ranked candidate with scores and explanations."""

    resume: ResumeData
    semantic_score: float
    skill_score: float
    hybrid_score: float
    rank: int
    explanation: Optional[str] = None
    fairness_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate scores after initialization."""
        self.semantic_score = max(0.0, min(1.0, self.semantic_score))
        self.skill_score = max(0.0, min(1.0, self.skill_score))
        self.hybrid_score = max(0.0, min(1.0, self.hybrid_score))
        if self.rank <= 0:
            self.rank = 1

    def get_score_breakdown(self) -> Dict[str, float]:
        """Get detailed score breakdown."""
        return {
            "semantic_score": self.semantic_score,
            "skill_score": self.skill_score,
            "hybrid_score": self.hybrid_score,
            "rank": float(self.rank),
        }

    def has_bias_flags(self) -> bool:
        """Check if candidate has any bias flags."""
        return len(self.fairness_flags) > 0

    def get_candidate_name(self) -> str:
        """Get candidate name safely."""
        if self.resume.contact_info and self.resume.contact_info.name:
            return self.resume.contact_info.name
        return f"Candidate {self.resume.candidate_id[:8]}"


@dataclass
class FairnessReport:
    """Comprehensive fairness analysis report."""

    total_candidates: int
    top_k: int
    demographic_parity: Dict[str, float] = field(default_factory=dict)
    four_fifths_violations: List[str] = field(default_factory=list)
    bias_flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_bias_flags(self) -> bool:
        """Check if report contains any bias flags."""
        return len(self.bias_flags) > 0 or len(self.four_fifths_violations) > 0

    def get_overall_fairness_score(self) -> float:
        """Calculate overall fairness score (0-1, higher is better)."""
        if not self.demographic_parity:
            return 1.0
        parity_scores = [score for score in self.demographic_parity.values() if isinstance(score, (int, float))]
        if not parity_scores:
            return 1.0
        avg_parity = sum(parity_scores) / len(parity_scores)
        violation_penalty = len(self.four_fifths_violations) * 0.1
        return max(0.0, min(1.0, avg_parity - violation_penalty))

    def add_recommendation(self, recommendation: str):
        """Add a fairness recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)

    def add_bias_flag(self, flag: str):
        """Add a bias flag."""
        if flag not in self.bias_flags:
            self.bias_flags.append(flag)


@dataclass
class BatchProcessingResult:
    """Result of batch resume processing."""

    job_id: str
    total_resumes: int
    successfully_parsed: int
    failed_parses: int
    ranked_candidates: List[RankedCandidate] = field(default_factory=list)
    fairness_report: Optional[FairnessReport] = None
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    def get_success_rate(self) -> float:
        """Calculate parsing success rate."""
        if self.total_resumes == 0:
            return 0.0
        return self.successfully_parsed / self.total_resumes

    def get_top_candidates(self, n: int = 10) -> List[RankedCandidate]:
        """Get top N candidates."""
        return sorted(self.ranked_candidates, key=lambda x: x.rank)[:n]
