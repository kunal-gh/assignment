"""Resume data models and structures."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid
import numpy as np


@dataclass
class ContactInfo:
    """Contact information extracted from resume."""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


@dataclass
class Experience:
    """Work experience entry."""
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)
    is_current: bool = False


@dataclass
class Education:
    """Education entry."""
    degree: str
    institution: str
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    major: Optional[str] = None
    relevant_coursework: List[str] = field(default_factory=list)


@dataclass
class ResumeData:
    """Complete resume data structure."""
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str = ""
    raw_text: str = ""
    contact_info: Optional[ContactInfo] = None
    skills: List[str] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    parsed_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate resume data after initialization."""
        if not self.candidate_id:
            self.candidate_id = str(uuid.uuid4())
        
        # Ensure contact_info is not None
        if self.contact_info is None:
            self.contact_info = ContactInfo(name="Unknown")
    
    def get_combined_text(self) -> str:
        """Combine all resume sections into a single text for embedding."""
        sections = []
        
        # Add contact info
        if self.contact_info and self.contact_info.name != "Unknown":
            sections.append(f"Name: {self.contact_info.name}")
        
        # Add skills
        if self.skills:
            sections.append(f"Skills: {', '.join(self.skills)}")
        
        # Add experience
        for exp in self.experience:
            exp_text = f"Experience: {exp.title} at {exp.company}"
            if exp.description:
                exp_text += f" - {exp.description}"
            sections.append(exp_text)
        
        # Add education
        for edu in self.education:
            edu_text = f"Education: {edu.degree}"
            if edu.institution:
                edu_text += f" from {edu.institution}"
            if edu.major:
                edu_text += f" in {edu.major}"
            sections.append(edu_text)
        
        return " ".join(sections)
    
    def get_years_of_experience(self) -> int:
        """Calculate total years of experience."""
        # Simple calculation - count number of experiences
        # In a real implementation, this would parse dates
        return len(self.experience)
    
    def has_required_skills(self, required_skills: List[str]) -> bool:
        """Check if resume contains any of the required skills."""
        resume_skills_lower = [skill.lower() for skill in self.skills]
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        return any(req_skill in resume_skills_lower for req_skill in required_skills_lower)