"""Data models for the AI Resume Screening System."""

from .resume import ResumeData, ContactInfo, Experience, Education
from .job import JobDescription
from .ranking import RankedCandidate, FairnessReport

__all__ = [
    "ResumeData",
    "ContactInfo", 
    "Experience",
    "Education",
    "JobDescription",
    "RankedCandidate",
    "FairnessReport",
]