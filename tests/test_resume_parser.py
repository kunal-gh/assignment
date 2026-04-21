"""Tests for resume parser functionality."""

import os
import tempfile
from pathlib import Path

import pytest

from src.models.resume import ContactInfo, ResumeData
from src.parsers.resume_parser import ResumeParser


class TestResumeParser:
    """Test cases for ResumeParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ResumeParser()
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        assert self.parser is not None
        assert hasattr(self.parser, 'text_extractor')
        assert hasattr(self.parser, 'section_parser')
        assert hasattr(self.parser, 'skill_extractor')
    
    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file returns error resume."""
        result = self.parser.parse_resume("nonexistent_file.pdf")
        
        assert isinstance(result, ResumeData)
        assert "Error parsing resume" in result.raw_text
        assert result.contact_info.name == "Parse Error"
    
    def test_parse_unsupported_format(self):
        """Test parsing unsupported file format."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"Test content")
            tmp_path = tmp.name
        
        try:
            result = self.parser.parse_resume(tmp_path)
            assert isinstance(result, ResumeData)
            assert "Error parsing resume" in result.raw_text
        finally:
            os.unlink(tmp_path)
    
    def test_batch_parse_empty_list(self):
        """Test batch parsing with empty list."""
        results = self.parser.batch_parse([])
        assert results == []
    
    def test_batch_parse_with_invalid_files(self):
        """Test batch parsing with invalid files."""
        invalid_files = ["nonexistent1.pdf", "nonexistent2.pdf"]
        results = self.parser.batch_parse(invalid_files)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ResumeData)
            assert "error" in result.raw_text.lower()
    
    def test_validate_resume_data(self):
        """Test resume data validation."""
        # Create test resume data
        resume_data = ResumeData(
            file_name="test.pdf",
            raw_text="This is a test resume with some content.",
            contact_info=ContactInfo(name="John Doe", email="john@example.com"),
            skills=["Python", "Machine Learning"],
            experience=[],
            education=[]
        )
        
        validation_result = self.parser.validate_resume_data(resume_data)
        
        assert isinstance(validation_result, dict)
        assert 'is_valid' in validation_result
        assert 'quality_score' in validation_result
        assert 'issues' in validation_result
        assert 'warnings' in validation_result
    
    def test_validate_low_quality_resume(self):
        """Test validation of low quality resume data."""
        # Create low quality resume data
        resume_data = ResumeData(
            file_name="test.pdf",
            raw_text="Short",  # Very short text
            contact_info=ContactInfo(name="Unknown"),  # No real name
            skills=[],  # No skills
            experience=[],  # No experience
            education=[]  # No education
        )
        
        validation_result = self.parser.validate_resume_data(resume_data)
        
        assert validation_result['quality_score'] < 0.5
        assert len(validation_result['issues']) > 0
        assert len(validation_result['warnings']) > 0


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing."""
    return """
    John Smith
    Email: john.smith@email.com
    Phone: (555) 123-4567
    
    EXPERIENCE
    Senior Software Engineer at Tech Corp
    2020 - Present
    - Developed web applications using Python and React
    - Led team of 5 developers
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology, 2018
    
    SKILLS
    Python, JavaScript, React, SQL, Machine Learning
    """


def test_extract_text_method(sample_resume_text):
    """Test text extraction method."""
    parser = ResumeParser()
    
    # Since we can't easily test with actual files in unit tests,
    # we'll test that the method exists and handles errors gracefully
    try:
        result = parser.extract_text("nonexistent.pdf")
        # Should not raise an exception, might return error message
        assert isinstance(result, str)
    except Exception as e:
        # If it raises an exception, it should be handled gracefully
        assert "not found" in str(e).lower() or "error" in str(e).lower()


def test_extract_sections_method(sample_resume_text):
    """Test section extraction method."""
    parser = ResumeParser()
    
    sections = parser.extract_sections(sample_resume_text)
    
    assert isinstance(sections, dict)
    # Should have identified some sections
    assert len(sections) > 0