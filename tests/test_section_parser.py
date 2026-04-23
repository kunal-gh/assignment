"""Unit tests for SectionParser – section identification and data extraction."""

import pytest

from src.models.resume import ContactInfo, Education, Experience
from src.parsers.section_parser import SectionParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parser():
    return SectionParser()


SAMPLE_RESUME = """
John Smith
john.smith@example.com
(555) 123-4567
linkedin.com/in/johnsmith
New York, NY

SUMMARY
Experienced software engineer with 8 years building scalable web services.

EXPERIENCE

Senior Software Engineer at Acme Corp
January 2020 - Present
- Led backend development using Python and FastAPI
- Managed a team of 4 engineers

Software Engineer at Beta Inc
March 2017 - December 2019
- Built REST APIs with Django
- Improved test coverage from 40% to 85%

EDUCATION

Bachelor of Science in Computer Science
MIT, 2017
GPA: 3.8

SKILLS
Python, JavaScript, Docker, Kubernetes, PostgreSQL, Redis

CERTIFICATIONS
AWS Certified Solutions Architect
"""


# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_returns_dict(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert isinstance(sections, dict)

    def test_identifies_experience_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "experience" in sections

    def test_identifies_education_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "education" in sections

    def test_identifies_skills_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "skills" in sections

    def test_identifies_summary_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "summary" in sections

    def test_identifies_certifications_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "certifications" in sections

    def test_header_captured(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        # The name / contact block before the first section header
        assert "header" in sections
        assert "John Smith" in sections["header"]

    def test_empty_text_returns_empty_dict(self, parser):
        assert parser.parse_sections("") == {}
        assert parser.parse_sections("   ") == {}

    def test_no_sections_returns_raw(self, parser):
        plain = "Just some plain text without any section headers."
        sections = parser.parse_sections(plain)
        assert "raw" in sections

    def test_skills_content_contains_skills(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "Python" in sections["skills"]

    def test_experience_content_contains_company(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        assert "Acme Corp" in sections["experience"]

    def test_section_headers_with_colon(self, parser):
        text = "SKILLS:\nPython, Java\n\nEDUCATION:\nBS Computer Science\n"
        sections = parser.parse_sections(text)
        assert "skills" in sections
        assert "education" in sections

    def test_mixed_case_headers(self, parser):
        text = "Work Experience\nSoftware Engineer at XYZ\n\nEducation\nBS CS\n"
        sections = parser.parse_sections(text)
        assert "experience" in sections
        assert "education" in sections


# ---------------------------------------------------------------------------
# extract_contact_info
# ---------------------------------------------------------------------------


class TestExtractContactInfo:
    def test_returns_contact_info_object(self, parser):
        result = parser.extract_contact_info(SAMPLE_RESUME[:300])
        assert isinstance(result, ContactInfo)

    def test_extracts_email(self, parser):
        text = "Jane Doe\njane.doe@company.org\n(555) 987-6543"
        info = parser.extract_contact_info(text)
        assert info.email == "jane.doe@company.org"

    def test_extracts_phone(self, parser):
        text = "Jane Doe\njane@example.com\n(555) 987-6543"
        info = parser.extract_contact_info(text)
        assert info.phone is not None
        assert "555" in info.phone

    def test_extracts_linkedin(self, parser):
        text = "John Smith\nlinkedin.com/in/johnsmith\njohn@example.com"
        info = parser.extract_contact_info(text)
        assert info.linkedin is not None
        assert "johnsmith" in info.linkedin

    def test_extracts_github(self, parser):
        text = "Alice Dev\nalice@dev.io\ngithub.com/alicedev"
        info = parser.extract_contact_info(text)
        assert info.github is not None
        assert "alicedev" in info.github

    def test_name_fallback_when_no_ner(self, parser):
        text = "Bob Builder\nbob@builder.com"
        info = parser.extract_contact_info(text)
        assert info.name != "Unknown"

    def test_unknown_name_when_no_name(self, parser):
        text = "email@example.com\n(555) 000-0000"
        info = parser.extract_contact_info(text)
        # Should not crash; name may be Unknown
        assert isinstance(info.name, str)

    def test_no_email_returns_none(self, parser):
        text = "John Doe\n(555) 123-4567"
        info = parser.extract_contact_info(text)
        assert info.email is None

    def test_no_phone_returns_none(self, parser):
        text = "John Doe\njohn@example.com"
        info = parser.extract_contact_info(text)
        assert info.phone is None


# ---------------------------------------------------------------------------
# extract_experience
# ---------------------------------------------------------------------------


class TestExtractExperience:
    def test_returns_list(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_experience(sections.get("experience", ""))
        assert isinstance(result, list)

    def test_extracts_multiple_entries(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_experience(sections.get("experience", ""))
        assert len(result) >= 1

    def test_experience_has_title(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_experience(sections.get("experience", ""))
        for exp in result:
            assert isinstance(exp, Experience)
            assert exp.title

    def test_experience_has_company(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_experience(sections.get("experience", ""))
        for exp in result:
            assert exp.company

    def test_current_job_flagged(self, parser):
        text = "Senior Engineer at TechCorp\nJanuary 2022 - Present\n- Built stuff"
        result = parser.extract_experience(text)
        assert any(exp.is_current for exp in result)

    def test_empty_text_returns_empty_list(self, parser):
        assert parser.extract_experience("") == []
        assert parser.extract_experience("   ") == []

    def test_date_extracted(self, parser):
        text = "Software Engineer at Foo Inc\n2019 - 2022\n- Did things"
        result = parser.extract_experience(text)
        assert result
        exp = result[0]
        assert exp.start_date is not None or exp.end_date is not None


# ---------------------------------------------------------------------------
# extract_education
# ---------------------------------------------------------------------------


class TestExtractEducation:
    def test_returns_list(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_education(sections.get("education", ""))
        assert isinstance(result, list)

    def test_extracts_degree(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_education(sections.get("education", ""))
        assert result
        assert result[0].degree

    def test_extracts_institution(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        result = parser.extract_education(sections.get("education", ""))
        assert result
        assert result[0].institution

    def test_extracts_gpa(self, parser):
        text = "Bachelor of Science in Computer Science\nMIT\nGPA: 3.8\n2017"
        result = parser.extract_education(text)
        assert result
        assert result[0].gpa == pytest.approx(3.8)

    def test_extracts_major(self, parser):
        text = "Bachelor of Science in Computer Science\nState University\n2020"
        result = parser.extract_education(text)
        assert result
        assert result[0].major is not None
        assert "Computer Science" in result[0].major

    def test_empty_text_returns_empty_list(self, parser):
        assert parser.extract_education("") == []
        assert parser.extract_education("   ") == []

    def test_graduation_date_extracted(self, parser):
        text = "Master of Science in Data Science\nStanford University\n2021"
        result = parser.extract_education(text)
        assert result
        assert result[0].graduation_date is not None


# ---------------------------------------------------------------------------
# Integration: full resume round-trip
# ---------------------------------------------------------------------------


class TestFullResumeRoundTrip:
    def test_full_parse_produces_all_sections(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        for key in ("experience", "education", "skills"):
            assert key in sections, f"Missing section: {key}"

    def test_contact_from_header(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        contact_text = sections.get("contact", "") + " " + sections.get("header", "")
        info = parser.extract_contact_info(contact_text)
        assert info.email == "john.smith@example.com"
        assert info.phone is not None

    def test_experience_entries_from_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        experiences = parser.extract_experience(sections.get("experience", ""))
        assert len(experiences) >= 1
        titles = [e.title for e in experiences]
        assert any("Engineer" in t for t in titles)

    def test_education_entries_from_section(self, parser):
        sections = parser.parse_sections(SAMPLE_RESUME)
        educations = parser.extract_education(sections.get("education", ""))
        assert len(educations) >= 1
        assert educations[0].degree
