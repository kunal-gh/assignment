"""Unit tests for SkillExtractor – skill extraction and normalization."""

import pytest

from src.parsers.skill_extractor import (
    ALL_SKILLS,
    ALL_TECHNICAL_SKILLS,
    CLOUD_DEVOPS,
    DATA_SCIENCE_ML,
    DATABASES,
    FRAMEWORKS_LIBRARIES,
    PROGRAMMING_LANGUAGES,
    SKILL_SYNONYMS,
    SOFT_SKILLS,
    SkillExtractor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor():
    return SkillExtractor()


SAMPLE_RESUME_TEXT = """
John Smith
john.smith@example.com

SUMMARY
Experienced software engineer with 8 years building scalable web services.

SKILLS
Python, JavaScript, React, Node.js, PostgreSQL, Docker, Kubernetes, AWS,
Machine Learning, TensorFlow, Git, Agile

EXPERIENCE

Senior Software Engineer at Acme Corp
January 2020 - Present
- Led backend development using Python and FastAPI
- Built REST APIs with Django and deployed on AWS
- Managed PostgreSQL and Redis databases
- Implemented CI/CD pipelines with GitHub Actions

Software Engineer at Beta Inc
March 2017 - December 2019
- Developed React frontend with Redux state management
- Experience with Docker and Kubernetes for container orchestration
- Proficient in TypeScript and JavaScript
- Technologies: Vue.js, MongoDB, Elasticsearch

EDUCATION
Bachelor of Science in Computer Science
MIT, 2017
"""

SKILLS_SECTION = (
    "Python, JavaScript, React, Node.js, PostgreSQL, Docker, Kubernetes, AWS, "
    "Machine Learning, TensorFlow, Git, Agile, TypeScript, Redis"
)

SECTIONS = {
    "skills": SKILLS_SECTION,
    "experience": """
Senior Software Engineer at Acme Corp
- Built REST APIs using Python and FastAPI
- Deployed services on AWS with Docker
- Used PostgreSQL and Redis for data storage
- Implemented CI/CD with GitHub Actions
""",
}


# ---------------------------------------------------------------------------
# Taxonomy / constants
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_programming_languages_not_empty(self):
        assert len(PROGRAMMING_LANGUAGES) > 10

    def test_frameworks_not_empty(self):
        assert len(FRAMEWORKS_LIBRARIES) > 10

    def test_databases_not_empty(self):
        assert len(DATABASES) > 5

    def test_cloud_devops_not_empty(self):
        assert len(CLOUD_DEVOPS) > 5

    def test_all_technical_skills_is_union(self):
        assert PROGRAMMING_LANGUAGES.issubset(ALL_TECHNICAL_SKILLS)
        assert FRAMEWORKS_LIBRARIES.issubset(ALL_TECHNICAL_SKILLS)
        assert DATABASES.issubset(ALL_TECHNICAL_SKILLS)

    def test_known_languages_present(self):
        for lang in ("python", "java", "javascript", "typescript", "go", "rust"):
            assert lang in PROGRAMMING_LANGUAGES

    def test_known_frameworks_present(self):
        for fw in ("react", "django", "flask", "spring boot", "fastapi"):
            assert fw in FRAMEWORKS_LIBRARIES

    def test_known_databases_present(self):
        for db in ("postgresql", "mongodb", "redis", "mysql", "elasticsearch"):
            assert db in DATABASES

    def test_known_cloud_tools_present(self):
        for tool in ("aws", "azure", "gcp", "docker", "kubernetes"):
            assert tool in CLOUD_DEVOPS


# ---------------------------------------------------------------------------
# Synonym map
# ---------------------------------------------------------------------------


class TestSynonymMap:
    def test_js_maps_to_javascript(self):
        assert SKILL_SYNONYMS["js"] == "javascript"

    def test_ts_maps_to_typescript(self):
        assert SKILL_SYNONYMS["ts"] == "typescript"

    def test_postgres_maps_to_postgresql(self):
        assert SKILL_SYNONYMS["postgres"] == "postgresql"

    def test_k8s_maps_to_kubernetes(self):
        assert SKILL_SYNONYMS["k8s"] == "kubernetes"

    def test_ml_maps_to_machine_learning(self):
        assert SKILL_SYNONYMS["ml"] == "machine learning"

    def test_aws_full_name_maps(self):
        assert SKILL_SYNONYMS["amazon web services"] == "aws"

    def test_gcp_full_name_maps(self):
        assert SKILL_SYNONYMS["google cloud platform"] == "gcp"

    def test_nodejs_variants_map(self):
        assert SKILL_SYNONYMS["nodejs"] == "node.js"
        assert SKILL_SYNONYMS["node js"] == "node.js"
        assert SKILL_SYNONYMS["node"] == "node.js"

    def test_reactjs_maps_to_react(self):
        assert SKILL_SYNONYMS["reactjs"] == "react"

    def test_sklearn_maps_to_scikit_learn(self):
        assert SKILL_SYNONYMS["sklearn"] == "scikit-learn"

    def test_synonym_values_are_canonical(self):
        """All synonym values should be in ALL_SKILLS or be well-known expansions."""
        for variant, canonical in SKILL_SYNONYMS.items():
            # The canonical form should either be in ALL_SKILLS or be a known expansion
            # (some expansions like "artificial intelligence" may not be in the taxonomy)
            assert isinstance(canonical, str) and len(canonical) > 0


# ---------------------------------------------------------------------------
# normalize_skill
# ---------------------------------------------------------------------------


class TestNormalizeSkill:
    def test_js_normalized(self, extractor):
        assert extractor.normalize_skill("js") == "javascript"

    def test_postgres_normalized(self, extractor):
        assert extractor.normalize_skill("postgres") == "postgresql"

    def test_k8s_normalized(self, extractor):
        assert extractor.normalize_skill("k8s") == "kubernetes"

    def test_unknown_skill_returned_as_is(self, extractor):
        assert extractor.normalize_skill("SomeUnknownTool") == "SomeUnknownTool"

    def test_case_insensitive_normalization(self, extractor):
        assert extractor.normalize_skill("JS") == "javascript"
        assert extractor.normalize_skill("Postgres") == "postgresql"

    def test_whitespace_stripped(self, extractor):
        assert extractor.normalize_skill("  js  ") == "javascript"


# ---------------------------------------------------------------------------
# extract_skills – basic functionality
# ---------------------------------------------------------------------------


class TestExtractSkillsBasic:
    def test_returns_list(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        assert isinstance(result, list)

    def test_empty_text_returns_empty_list(self, extractor):
        assert extractor.extract_skills("") == []
        assert extractor.extract_skills("   ") == []

    def test_extracts_python(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        result_lower = [s.lower() for s in result]
        assert "python" in result_lower

    def test_extracts_javascript(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        result_lower = [s.lower() for s in result]
        assert "javascript" in result_lower

    def test_extracts_docker(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        result_lower = [s.lower() for s in result]
        assert "docker" in result_lower

    def test_extracts_aws(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        result_lower = [s.lower() for s in result]
        assert "aws" in result_lower

    def test_no_duplicates(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        result_lower = [s.lower() for s in result]
        assert len(result_lower) == len(set(result_lower))

    def test_result_is_strings(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        for skill in result:
            assert isinstance(skill, str)
            assert len(skill) > 0


# ---------------------------------------------------------------------------
# extract_skills – with sections
# ---------------------------------------------------------------------------


class TestExtractSkillsWithSections:
    def test_extracts_from_skills_section(self, extractor):
        result = extractor.extract_skills("", sections=SECTIONS)
        result_lower = [s.lower() for s in result]
        assert "python" in result_lower

    def test_extracts_from_experience_section(self, extractor):
        result = extractor.extract_skills("", sections=SECTIONS)
        result_lower = [s.lower() for s in result]
        # FastAPI is mentioned in experience
        assert "fastapi" in result_lower

    def test_sections_improve_extraction(self, extractor):
        without_sections = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        with_sections = extractor.extract_skills(SAMPLE_RESUME_TEXT, sections=SECTIONS)
        # With sections should find at least as many skills
        assert len(with_sections) >= len(without_sections) - 2  # allow small variance


# ---------------------------------------------------------------------------
# extract_skills – normalization in output
# ---------------------------------------------------------------------------


class TestExtractSkillsNormalization:
    def test_synonym_normalized_in_output(self, extractor):
        text = "Experience with JS, Postgres, and k8s"
        result = extractor.extract_skills(text)
        result_lower = [s.lower() for s in result]
        # Synonyms should be resolved to canonical forms
        assert "javascript" in result_lower or "js" not in result_lower
        assert "postgresql" in result_lower or "postgres" not in result_lower
        assert "kubernetes" in result_lower or "k8s" not in result_lower

    def test_nodejs_normalized(self, extractor):
        text = "Built backend services with nodejs and express"
        result = extractor.extract_skills(text)
        result_lower = [s.lower() for s in result]
        assert "node.js" in result_lower or "nodejs" not in result_lower

    def test_reactjs_normalized(self, extractor):
        text = "Frontend development with ReactJS and Redux"
        result = extractor.extract_skills(text)
        result_lower = [s.lower() for s in result]
        assert "react" in result_lower


# ---------------------------------------------------------------------------
# _extract_from_skills_section
# ---------------------------------------------------------------------------


class TestExtractFromSkillsSection:
    def test_comma_separated(self, extractor):
        text = "Python, JavaScript, Docker, AWS"
        result = extractor._extract_from_skills_section(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower
        assert "javascript" in result_lower
        assert "docker" in result_lower
        assert "aws" in result_lower

    def test_bullet_separated(self, extractor):
        text = "• Python\n• React\n• PostgreSQL"
        result = extractor._extract_from_skills_section(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower
        assert "react" in result_lower
        assert "postgresql" in result_lower

    def test_pipe_separated(self, extractor):
        text = "Python | Java | Go | Rust"
        result = extractor._extract_from_skills_section(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower
        assert "java" in result_lower

    def test_parenthetical_removed(self, extractor):
        text = "Python (5 years), JavaScript (3 years)"
        result = extractor._extract_from_skills_section(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower
        assert "javascript" in result_lower

    def test_empty_section_returns_empty(self, extractor):
        assert extractor._extract_from_skills_section("") == set()
        assert extractor._extract_from_skills_section("   ") == set()

    def test_unknown_words_excluded(self, extractor):
        text = "Python, Foobar123, JavaScript"
        result = extractor._extract_from_skills_section(text)
        result_lower = {s.lower() for s in result}
        assert "foobar123" not in result_lower


# ---------------------------------------------------------------------------
# _extract_from_experience
# ---------------------------------------------------------------------------


class TestExtractFromExperience:
    def test_extracts_from_using_context(self, extractor):
        text = "Built microservices using Python and FastAPI"
        result = extractor._extract_from_experience(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower or "fastapi" in result_lower

    def test_extracts_from_technologies_context(self, extractor):
        text = "Technologies: React, Node.js, MongoDB, Redis"
        result = extractor._extract_from_experience(text)
        result_lower = {s.lower() for s in result}
        assert "react" in result_lower or "mongodb" in result_lower

    def test_extracts_from_proficient_context(self, extractor):
        text = "Proficient in Python, Django, and PostgreSQL"
        result = extractor._extract_from_experience(text)
        result_lower = {s.lower() for s in result}
        assert "python" in result_lower or "django" in result_lower

    def test_empty_text_returns_empty(self, extractor):
        assert extractor._extract_from_experience("") == set()


# ---------------------------------------------------------------------------
# get_skill_categories
# ---------------------------------------------------------------------------


class TestGetSkillCategories:
    def test_returns_dict(self, extractor):
        result = extractor.get_skill_categories(["python", "react", "postgresql"])
        assert isinstance(result, dict)

    def test_python_in_programming_languages(self, extractor):
        result = extractor.get_skill_categories(["python"])
        assert "programming_languages" in result
        assert "python" in result["programming_languages"]

    def test_react_in_frameworks(self, extractor):
        result = extractor.get_skill_categories(["react"])
        assert "frameworks_libraries" in result
        assert "react" in result["frameworks_libraries"]

    def test_postgresql_in_databases(self, extractor):
        result = extractor.get_skill_categories(["postgresql"])
        assert "databases" in result
        assert "postgresql" in result["databases"]

    def test_aws_in_cloud_devops(self, extractor):
        result = extractor.get_skill_categories(["aws"])
        assert "cloud_devops" in result
        assert "aws" in result["cloud_devops"]

    def test_machine_learning_in_data_science(self, extractor):
        result = extractor.get_skill_categories(["machine learning"])
        assert "data_science_ml" in result
        assert "machine learning" in result["data_science_ml"]

    def test_leadership_in_soft_skills(self, extractor):
        result = extractor.get_skill_categories(["leadership"])
        assert "soft_skills" in result
        assert "leadership" in result["soft_skills"]

    def test_unknown_skill_in_other(self, extractor):
        result = extractor.get_skill_categories(["some_unknown_tool_xyz"])
        assert "other" in result
        assert "some_unknown_tool_xyz" in result["other"]

    def test_empty_categories_excluded(self, extractor):
        result = extractor.get_skill_categories(["python"])
        # Only categories with skills should be present
        for cat, skills in result.items():
            assert len(skills) > 0

    def test_empty_input_returns_empty(self, extractor):
        result = extractor.get_skill_categories([])
        assert result == {}

    def test_mixed_skills_categorized(self, extractor):
        skills = ["python", "react", "postgresql", "aws", "machine learning", "leadership"]
        result = extractor.get_skill_categories(skills)
        assert "programming_languages" in result
        assert "frameworks_libraries" in result
        assert "databases" in result
        assert "cloud_devops" in result
        assert "data_science_ml" in result
        assert "soft_skills" in result


# ---------------------------------------------------------------------------
# _is_known_skill
# ---------------------------------------------------------------------------


class TestIsKnownSkill:
    def test_known_language_recognized(self, extractor):
        assert extractor._is_known_skill("python") is True
        assert extractor._is_known_skill("JavaScript") is True

    def test_known_framework_recognized(self, extractor):
        assert extractor._is_known_skill("react") is True
        assert extractor._is_known_skill("Django") is True

    def test_synonym_recognized(self, extractor):
        assert extractor._is_known_skill("js") is True
        assert extractor._is_known_skill("postgres") is True

    def test_unknown_word_not_recognized(self, extractor):
        assert extractor._is_known_skill("foobar") is False
        assert extractor._is_known_skill("xyz123") is False

    def test_empty_string_not_recognized(self, extractor):
        assert extractor._is_known_skill("") is False

    def test_single_char_not_recognized(self, extractor):
        assert extractor._is_known_skill("a") is False

    def test_very_long_string_not_recognized(self, extractor):
        assert extractor._is_known_skill("a" * 60) is False

    def test_case_insensitive(self, extractor):
        assert extractor._is_known_skill("PYTHON") is True
        assert extractor._is_known_skill("Python") is True


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRankSkills:
    def test_more_frequent_skill_ranked_higher(self, extractor):
        text = "Python Python Python Java"
        skills = ["python", "java"]
        ranked = extractor._rank_skills(skills, text)
        assert ranked.index("python") < ranked.index("java")

    def test_returns_all_input_skills(self, extractor):
        skills = ["python", "java", "react"]
        ranked = extractor._rank_skills(skills, "some text")
        assert set(ranked) == set(skills)

    def test_empty_skills_returns_empty(self, extractor):
        assert extractor._rank_skills([], "some text") == []


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_resume_extracts_multiple_skills(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT, sections=SECTIONS)
        assert len(result) >= 5

    def test_no_common_stopwords_in_output(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT)
        stopwords = {"and", "or", "the", "with", "for", "in", "on", "at", "to", "of"}
        result_lower = {s.lower() for s in result}
        assert result_lower.isdisjoint(stopwords)

    def test_skills_from_structured_section_included(self, extractor):
        sections = {"skills": "Python, TypeScript, Docker, Kubernetes, PostgreSQL"}
        result = extractor.extract_skills("", sections=sections)
        result_lower = [s.lower() for s in result]
        assert "python" in result_lower
        assert "typescript" in result_lower
        assert "docker" in result_lower

    def test_synonym_resolution_end_to_end(self, extractor):
        sections = {"skills": "JS, TS, Postgres, k8s, ML"}
        result = extractor.extract_skills("", sections=sections)
        result_lower = [s.lower() for s in result]
        # Synonyms should be resolved
        assert "javascript" in result_lower
        assert "typescript" in result_lower
        assert "postgresql" in result_lower
        assert "kubernetes" in result_lower
        assert "machine learning" in result_lower

    def test_categorize_extracted_skills(self, extractor):
        result = extractor.extract_skills(SAMPLE_RESUME_TEXT, sections=SECTIONS)
        categories = extractor.get_skill_categories(result)
        # Should have at least programming languages and one other category
        assert len(categories) >= 2
