"""Job description data models."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class JobDescription:
    """Job description data structure."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience_level: str = "mid"  # entry, mid, senior, executive
    location: Optional[str] = None
    salary_range: Optional[str] = None
    company: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate job description after initialization."""
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
        
        # Validate experience level
        valid_levels = ["entry", "mid", "senior", "executive"]
        if self.experience_level not in valid_levels:
            self.experience_level = "mid"
        
        # Ensure required_skills is not empty
        if not self.required_skills and self.description:
            # Extract basic skills from description if none provided
            self.required_skills = self._extract_basic_skills()
    
    def _extract_basic_skills(self) -> List[str]:
        """Extract basic skills from job description text."""
        common_skills = [
            "python", "java", "javascript", "sql", "html", "css",
            "react", "angular", "vue", "node.js", "django", "flask",
            "machine learning", "data science", "aws", "docker", "kubernetes",
            "git", "agile", "scrum", "project management"
        ]
        
        description_lower = self.description.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in description_lower:
                found_skills.append(skill)
        
        return found_skills[:10]  # Limit to top 10 skills
    
    def get_all_skills(self) -> List[str]:
        """Get combined list of required and preferred skills."""
        return list(set(self.required_skills + self.preferred_skills))
    
    def get_combined_text(self) -> str:
        """Combine job description components for embedding."""
        sections = []
        
        if self.title:
            sections.append(f"Job Title: {self.title}")
        
        if self.description:
            sections.append(f"Description: {self.description}")
        
        if self.required_skills:
            sections.append(f"Required Skills: {', '.join(self.required_skills)}")
        
        if self.preferred_skills:
            sections.append(f"Preferred Skills: {', '.join(self.preferred_skills)}")
        
        if self.experience_level:
            sections.append(f"Experience Level: {self.experience_level}")
        
        return " ".join(sections)
    
    def matches_experience_level(self, candidate_years: int) -> bool:
        """Check if candidate experience matches job requirements."""
        level_requirements = {
            "entry": (0, 2),
            "mid": (2, 5),
            "senior": (5, 10),
            "executive": (10, float('inf'))
        }
        
        min_years, max_years = level_requirements.get(self.experience_level, (0, float('inf')))
        return min_years <= candidate_years <= max_years