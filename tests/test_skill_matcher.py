"""Unit tests for SkillMatcher."""

import pytest
from src.ranking.skill_matcher import SkillMatcher


class TestSkillMatcher:
    """Test cases for SkillMatcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = SkillMatcher()
    
    def test_matcher_initialization(self):
        """Test matcher initializes correctly."""
        assert self.matcher is not None
        assert isinstance(self.matcher.skill_synonyms, dict)
        assert isinstance(self.matcher.skill_categories, dict)
    
    def test_calculate_skill_match_exact_match(self):
        """Test perfect skill match returns 1.0."""
        resume_skills = ["python", "javascript", "react"]
        required_skills = ["python", "javascript", "react"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_calculate_skill_match_no_match(self):
        """Test no skill match returns 0.0."""
        resume_skills = ["python", "django"]
        required_skills = ["java", "spring"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        assert score == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_skill_match_partial_match(self):
        """Test partial skill match returns appropriate score."""
        resume_skills = ["python", "javascript", "react"]
        required_skills = ["python", "java", "angular"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        assert 0.0 < score < 1.0
    
    def test_calculate_skill_match_with_preferred_skills(self):
        """Test skill matching with both required and preferred skills."""
        resume_skills = ["python", "javascript", "react", "docker"]
        required_skills = ["python", "javascript"]
        preferred_skills = ["react", "aws"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills, preferred_skills)
        assert 0.4 < score <= 1.0
    
    def test_calculate_skill_match_empty_resume_skills(self):
        """Test empty resume skills returns 0.0."""
        resume_skills = []
        required_skills = ["python", "java"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        assert score == 0.0
    
    def test_calculate_skill_match_empty_required_skills(self):
        """Test empty required skills with preferred skills."""
        resume_skills = ["python", "javascript"]
        required_skills = []
        preferred_skills = ["python", "react"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills, preferred_skills)
        assert score > 0.0
    
    def test_calculate_skill_match_with_synonyms(self):
        """Test skill matching handles synonyms correctly."""
        resume_skills = ["js", "nodejs", "postgres"]
        required_skills = ["javascript", "node.js", "postgresql"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_calculate_skill_match_weighted_scoring(self):
        """Test weighted scoring for required vs preferred skills."""
        resume_skills = ["python", "react"]
        required_skills = ["python"]
        preferred_skills = ["react"]
        
        # Score with default weights (0.7 required, 0.3 preferred)
        score_default = self.matcher.calculate_skill_match(resume_skills, required_skills, preferred_skills)
        
        # Score with custom weights (0.5 required, 0.5 preferred)
        score_equal = self.matcher.calculate_skill_match(
            resume_skills, required_skills, preferred_skills, 
            required_weight=0.5, preferred_weight=0.5
        )
        
        # Both should be valid scores
        assert 0.0 < score_default <= 1.0
        assert 0.0 < score_equal <= 1.0
        
        # Test with asymmetric coverage: candidate has required skill but NOT preferred.
        # required_coverage = 1.0 (python matched)
        # preferred_coverage = 0.0 (react not in resume)
        # So: req_heavy = 0.9 * 1.0 + 0.1 * 0.0 = 0.90
        #     pref_heavy = 0.1 * 1.0 + 0.9 * 0.0 = 0.10
        # These must be different.
        resume_skills2 = ["python"]          # has required, lacks preferred
        required_skills2 = ["python"]
        preferred_skills2 = ["java"]         # java not in resume
        
        score_req_heavy = self.matcher.calculate_skill_match(
            resume_skills2, required_skills2, preferred_skills2,
            required_weight=0.9, preferred_weight=0.1
        )
        score_pref_heavy = self.matcher.calculate_skill_match(
            resume_skills2, required_skills2, preferred_skills2,
            required_weight=0.1, preferred_weight=0.9
        )
        
        # With asymmetric coverage and different weights, scores must differ
        assert score_req_heavy != score_pref_heavy
        # Req-heavy score should be higher (candidate matches the required skill)
        assert score_req_heavy > score_pref_heavy
    
    def test_calculate_skill_match_coverage_bonus(self):
        """Test coverage bonus for high required skill matches."""
        resume_skills = ["python", "javascript", "react", "django", "postgresql"]
        required_skills = ["python", "javascript", "react", "django", "postgresql"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        # Should get coverage bonus for 100% required skill match
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_analyze_skill_match_structure(self):
        """Test analyze_skill_match returns correct structure."""
        resume_skills = ["python", "javascript", "react"]
        required_skills = ["python", "java"]
        preferred_skills = ["react", "aws"]
        
        analysis = self.matcher.analyze_skill_match(resume_skills, required_skills, preferred_skills)
        
        expected_keys = [
            'matched_required', 'matched_preferred', 'missing_required', 
            'missing_preferred', 'extra_skills', 'required_coverage', 
            'preferred_coverage', 'total_resume_skills', 'skill_categories', 
            'overall_score'
        ]
        
        for key in expected_keys:
            assert key in analysis
    
    def test_analyze_skill_match_content(self):
        """Test analyze_skill_match returns correct content."""
        resume_skills = ["python", "javascript", "react"]
        required_skills = ["python", "java"]
        preferred_skills = ["react", "aws"]
        
        analysis = self.matcher.analyze_skill_match(resume_skills, required_skills, preferred_skills)
        
        assert "python" in analysis['matched_required']
        assert "java" in analysis['missing_required']
        assert "react" in analysis['matched_preferred']
        assert "aws" in analysis['missing_preferred']
        assert "javascript" in analysis['extra_skills']
        assert analysis['required_coverage'] == 0.5  # 1 out of 2 required skills
        assert analysis['preferred_coverage'] == 0.5  # 1 out of 2 preferred skills
        assert analysis['total_resume_skills'] == 3
    
    def test_find_skill_gaps(self):
        """Test skill gap identification."""
        resume_skills = ["python", "javascript"]
        job_skills = ["python", "java", "react", "aws"]
        
        gaps = self.matcher.find_skill_gaps(resume_skills, job_skills)
        
        assert isinstance(gaps, dict)
        # Should identify java, react, aws as missing
        all_missing = []
        for category_skills in gaps.values():
            all_missing.extend(category_skills)
        
        assert "java" in all_missing
        assert "react" in all_missing
        assert "aws" in all_missing
        assert "python" not in all_missing
        assert "javascript" not in all_missing
    
    def test_suggest_skill_improvements(self):
        """Test skill improvement suggestions."""
        resume_skills = ["python", "html"]
        required_skills = ["python", "javascript", "react", "node.js", "aws"]
        
        suggestions = self.matcher.suggest_skill_improvements(resume_skills, required_skills)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert len(suggestions) <= 10  # Should limit suggestions
        
        # Should prioritize missing required skills
        missing_required = ["javascript", "react", "node.js", "aws"]
        for skill in suggestions[:4]:  # First few should be missing required
            assert skill in missing_required
    
    def test_normalize_skills(self):
        """Test skill normalization with synonyms."""
        skills = ["js", "nodejs", "postgres", "python", "react"]
        normalized = self.matcher._normalize_skills(skills)
        
        assert "javascript" in normalized
        assert "node.js" in normalized
        assert "postgresql" in normalized
        assert "python" in normalized
        assert "react" in normalized
        
        # Original synonyms should not be in normalized list
        assert "js" not in normalized
        assert "nodejs" not in normalized
        assert "postgres" not in normalized
    
    def test_normalize_skills_deduplication(self):
        """Test skill normalization removes duplicates."""
        skills = ["python", "python", "js", "javascript"]
        normalized = self.matcher._normalize_skills(skills)
        
        # Should have only unique skills
        assert len(normalized) == len(set(normalized))
        assert "python" in normalized
        assert "javascript" in normalized
        # Should not have both js and javascript
        assert normalized.count("javascript") == 1
    
    def test_categorize_skills(self):
        """Test skill categorization."""
        skills = ["python", "react", "postgresql", "aws", "tensorflow", "git", "leadership"]
        categories = self.matcher._categorize_skills(skills)
        
        assert isinstance(categories, dict)
        assert "programming_languages" in categories
        assert "frameworks_libraries" in categories
        assert "databases" in categories
        assert "cloud_devops" in categories
        
        assert "python" in categories["programming_languages"]
        assert "react" in categories["frameworks_libraries"]
        assert "postgresql" in categories["databases"]
        assert "aws" in categories["cloud_devops"]
    
    def test_get_skill_category(self):
        """Test individual skill categorization."""
        assert self.matcher._get_skill_category("python") == "programming_languages"
        assert self.matcher._get_skill_category("react") == "frameworks_libraries"
        assert self.matcher._get_skill_category("postgresql") == "databases"
        assert self.matcher._get_skill_category("aws") == "cloud_devops"
        assert self.matcher._get_skill_category("tensorflow") == "data_science"
        assert self.matcher._get_skill_category("git") == "tools"
        assert self.matcher._get_skill_category("leadership") == "soft_skills"
        assert self.matcher._get_skill_category("unknown_skill") == "other"
    
    def test_get_related_skills(self):
        """Test related skills suggestions."""
        python_related = self.matcher._get_related_skills("python")
        assert isinstance(python_related, list)
        assert "django" in python_related or "flask" in python_related
        
        js_related = self.matcher._get_related_skills("javascript")
        assert "react" in js_related or "node.js" in js_related
        
        # Unknown skill should return empty list
        unknown_related = self.matcher._get_related_skills("unknown_skill")
        assert unknown_related == []
    
    def test_jaccard_similarity_calculation(self):
        """Test Jaccard similarity calculation is correct."""
        # Test case: resume has [A, B, C], job requires [B, C, D]
        # Intersection: [B, C] (2 items)
        # Union: [A, B, C, D] (4 items)
        # Jaccard = 2/4 = 0.5
        
        resume_skills = ["python", "javascript", "html"]
        required_skills = ["javascript", "html", "css"]
        
        score = self.matcher.calculate_skill_match(resume_skills, required_skills)
        
        # Expected Jaccard similarity: 2/4 = 0.5
        expected_jaccard = 2.0 / 4.0  # 0.5
        assert score == pytest.approx(expected_jaccard, abs=0.01)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty lists
        assert self.matcher.calculate_skill_match([], []) == 0.0
        assert self.matcher.calculate_skill_match(["python"], []) == 0.0
        assert self.matcher.calculate_skill_match([], ["python"]) == 0.0
        
        # None values
        assert self.matcher.calculate_skill_match(["python"], ["java"], None) >= 0.0
        
        # Single skill matches
        assert self.matcher.calculate_skill_match(["python"], ["python"]) == pytest.approx(1.0, abs=0.01)
        
        # Case sensitivity - should be handled through normalization
        score1 = self.matcher.calculate_skill_match(["Python"], ["python"])
        score2 = self.matcher.calculate_skill_match(["python"], ["python"])
        assert score1 == score2  # Should be case insensitive through normalization