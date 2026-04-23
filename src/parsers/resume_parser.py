"""Main resume parser class that orchestrates the parsing process."""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List

from ..models.resume import ContactInfo, Education, Experience, ResumeData
from .section_parser import SectionParser
from .skill_extractor import SkillExtractor
from .text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class ResumeParser:
    """Main resume parser that coordinates text extraction and data parsing."""

    def __init__(self):
        """Initialize the resume parser with its components."""
        self.text_extractor = TextExtractor()
        self.section_parser = SectionParser()
        self.skill_extractor = SkillExtractor()

    def parse_resume(self, file_path: str) -> ResumeData:
        """
        Parse a single resume file and extract structured data.

        Args:
            file_path: Path to the resume file (PDF or DOCX)

        Returns:
            ResumeData object with extracted information

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            Exception: For other parsing errors
        """
        try:
            # Validate file exists and format
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Resume file not found: {file_path}")

            if path.suffix.lower() not in [".pdf", ".docx"]:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Check file size (10MB limit)
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError(f"File too large: {path.stat().st_size} bytes (max 10MB)")

            logger.info(f"Starting to parse resume: {file_path}")

            # Extract raw text
            raw_text = self.text_extractor.extract_text(file_path)
            if not raw_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                raw_text = "No text could be extracted from this resume."

            # Parse sections
            sections = self.section_parser.parse_sections(raw_text)

            # Extract contact information
            contact_info = self._extract_contact_info(sections)

            # Extract experience
            experience = self._extract_experience(sections)

            # Extract education
            education = self._extract_education(sections)

            # Extract skills
            skills = self.skill_extractor.extract_skills(raw_text, sections)

            # Create ResumeData object
            resume_data = ResumeData(
                file_name=path.name,
                raw_text=raw_text,
                contact_info=contact_info,
                skills=skills,
                experience=experience,
                education=education,
            )

            logger.info(f"Successfully parsed resume: {file_path}")
            logger.debug(f"Extracted {len(skills)} skills, {len(experience)} experiences, {len(education)} education entries")

            return resume_data

        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Return a minimal ResumeData object with error info
            return ResumeData(
                file_name=Path(file_path).name if Path(file_path).exists() else "unknown",
                raw_text=f"Error parsing resume: {str(e)}",
                contact_info=ContactInfo(name="Parse Error"),
                skills=[],
                experience=[],
                education=[],
            )

    def batch_parse(self, file_paths: List[str]) -> List[ResumeData]:
        """
        Parse multiple resume files in batch.

        Args:
            file_paths: List of paths to resume files

        Returns:
            List of ResumeData objects
        """
        results = []

        logger.info(f"Starting batch parsing of {len(file_paths)} resumes")

        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Parsing resume {i}/{len(file_paths)}: {file_path}")
                resume_data = self.parse_resume(file_path)
                results.append(resume_data)

            except Exception as e:
                logger.error(f"Failed to parse resume {file_path}: {str(e)}")
                # Add error resume to maintain list consistency
                error_resume = ResumeData(
                    file_name=Path(file_path).name,
                    raw_text=f"Batch parsing error: {str(e)}",
                    contact_info=ContactInfo(name="Batch Parse Error"),
                    skills=[],
                    experience=[],
                    education=[],
                )
                results.append(error_resume)

        logger.info(f"Completed batch parsing: {len(results)} resumes processed")
        return results

    def extract_text(self, file_path: str) -> str:
        """
        Extract raw text from resume file.

        Args:
            file_path: Path to the resume file

        Returns:
            Raw text content
        """
        return self.text_extractor.extract_text(file_path)

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract resume sections.

        Args:
            text: Raw resume text

        Returns:
            Dictionary mapping section names to content
        """
        return self.section_parser.parse_sections(text)

    def _extract_contact_info(self, sections: Dict[str, str]) -> ContactInfo:
        """Extract contact information from parsed sections."""
        # Look for contact info in header or contact section
        contact_text = sections.get("contact", "") + " " + sections.get("header", "")

        # Use section parser's contact extraction
        return self.section_parser.extract_contact_info(contact_text)

    def _extract_experience(self, sections: Dict[str, str]) -> List[Experience]:
        """Extract work experience from parsed sections."""
        experience_text = sections.get("experience", "") + " " + sections.get("work", "")

        if not experience_text.strip():
            return []

        return self.section_parser.extract_experience(experience_text)

    def _extract_education(self, sections: Dict[str, str]) -> List[Education]:
        """Extract education information from parsed sections."""
        education_text = sections.get("education", "")

        if not education_text.strip():
            return []

        return self.section_parser.extract_education(education_text)

    def validate_resume_data(self, resume_data: ResumeData) -> Dict[str, Any]:
        """
        Validate parsed resume data and return quality metrics.

        Args:
            resume_data: Parsed resume data

        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {"is_valid": True, "quality_score": 0.0, "issues": [], "warnings": []}

        # Check if basic information is present
        if not resume_data.contact_info or not resume_data.contact_info.name or resume_data.contact_info.name == "Unknown":
            validation_result["issues"].append("No contact name found")
            validation_result["quality_score"] -= 0.2

        if not resume_data.skills:
            validation_result["warnings"].append("No skills extracted")
            validation_result["quality_score"] -= 0.1

        if not resume_data.experience:
            validation_result["warnings"].append("No work experience found")
            validation_result["quality_score"] -= 0.1

        if not resume_data.education:
            validation_result["warnings"].append("No education information found")
            validation_result["quality_score"] -= 0.1

        if len(resume_data.raw_text) < 100:
            validation_result["issues"].append("Very short resume text")
            validation_result["quality_score"] -= 0.3

        # Calculate final quality score (0-1 scale)
        base_score = 1.0
        validation_result["quality_score"] = max(0.0, base_score + validation_result["quality_score"])

        # Mark as invalid if quality is too low
        if validation_result["quality_score"] < 0.3:
            validation_result["is_valid"] = False

        return validation_result
