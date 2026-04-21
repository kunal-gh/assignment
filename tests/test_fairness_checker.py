"""Unit tests for FairnessChecker."""

import pytest

from src.models.ranking import FairnessReport, RankedCandidate
from src.models.resume import ContactInfo, ResumeData
from src.ranking.fairness_checker import FairnessChecker


def _make_candidate(name: str, score: float, rank: int, years_exp: int = 3,
                    skills: list = None) -> RankedCandidate:
    """Helper — build a RankedCandidate with minimal data."""
    contact = ContactInfo(name=name, email=f"{name.lower().replace(' ', '.')}@test.com")
    resume = ResumeData(
        contact_info=contact,
        skills=skills or [],
        raw_text=f"Resume of {name}",
    )
    return RankedCandidate(
        resume=resume,
        semantic_score=score,
        skill_score=score,
        hybrid_score=score,
        rank=rank,
    )


class TestFairnessCheckerInitialisation:

    def test_initialises_correctly(self):
        checker = FairnessChecker()
        assert checker is not None
        assert checker.four_fifths_threshold == 0.8
        assert "gender" in checker.protected_attributes

    def test_protected_attributes_list(self):
        checker = FairnessChecker()
        expected = ["gender", "ethnicity", "age_group", "education_level"]
        for attr in expected:
            assert attr in checker.protected_attributes


class TestFairnessReportGeneration:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_empty_candidates_returns_report(self):
        report = self.checker.generate_fairness_report([], top_k=10)
        assert isinstance(report, FairnessReport)
        assert report.total_candidates == 0

    def test_single_candidate_report(self):
        candidates = [_make_candidate("Alice Smith", 0.85, 1)]
        report = self.checker.generate_fairness_report(candidates, top_k=1)
        assert report.total_candidates == 1

    def test_report_contains_all_fields(self):
        candidates = [_make_candidate(f"Person {i}", float(1 - i * 0.1), i + 1)
                      for i in range(5)]
        report = self.checker.generate_fairness_report(candidates, top_k=3)
        assert hasattr(report, "bias_flags")
        assert hasattr(report, "recommendations")
        assert hasattr(report, "total_candidates")
        assert hasattr(report, "top_k")

    def test_top_k_capped_at_total_candidates(self):
        candidates = [_make_candidate(f"Person {i}", 0.8, i + 1) for i in range(3)]
        report = self.checker.generate_fairness_report(candidates, top_k=100)
        # Should not fail even with top_k > total
        assert report.total_candidates == 3

    def test_overall_fairness_score_in_range(self):
        candidates = [_make_candidate(f"Candidate {i}", float(i + 1) / 10.0, i + 1)
                      for i in range(10)]
        report = self.checker.generate_fairness_report(candidates, top_k=5)
        score = report.get_overall_fairness_score()
        assert 0.0 <= score <= 1.0

    def test_bias_flags_is_list(self):
        candidates = [_make_candidate(f"P {i}", 0.5, i + 1) for i in range(5)]
        report = self.checker.generate_fairness_report(candidates, top_k=3)
        assert isinstance(report.bias_flags, list)

    def test_recommendations_is_list(self):
        candidates = [_make_candidate(f"P {i}", 0.5, i + 1) for i in range(5)]
        report = self.checker.generate_fairness_report(candidates, top_k=3)
        assert isinstance(report.recommendations, list)


class TestFourFifthsRule:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_no_violation_at_equal_parity(self):
        parity = {"gender_male": 1.0, "gender_female": 1.0}
        violations = self.checker.check_four_fifths_rule(parity)
        assert violations == []

    def test_violation_below_threshold(self):
        parity = {"gender_female": 0.5}  # 0.5 < 0.8 → violation
        violations = self.checker.check_four_fifths_rule(parity)
        assert len(violations) == 1
        assert "gender_female" in violations[0]

    def test_no_violation_at_exactly_threshold(self):
        parity = {"gender_female": 0.80}  # exactly 0.8 → no violation
        violations = self.checker.check_four_fifths_rule(parity)
        assert violations == []

    def test_multiple_violations_detected(self):
        parity = {
            "gender_female": 0.5,
            "ethnicity_hispanic": 0.6,
            "age_group_over_50": 0.75,
        }
        violations = self.checker.check_four_fifths_rule(parity)
        assert len(violations) == 3

    def test_empty_parity_results_no_violations(self):
        violations = self.checker.check_four_fifths_rule({})
        assert violations == []


class TestDemographicParity:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_demographic_parity_returns_dict(self):
        candidates = [_make_candidate(f"Person {i}", 0.7, i + 1) for i in range(6)]
        demographics = self.checker._extract_demographics(candidates)
        result = self.checker.check_demographic_parity(candidates, top_k=3, demographics=demographics)
        assert isinstance(result, dict)

    def test_parity_values_are_floats(self):
        candidates = [_make_candidate(f"Person {i}", 0.7, i + 1) for i in range(6)]
        demographics = self.checker._extract_demographics(candidates)
        result = self.checker.check_demographic_parity(candidates, top_k=3, demographics=demographics)
        for key, val in result.items():
            assert isinstance(val, float), f"Expected float for {key}, got {type(val)}"


class TestFairnessMetrics:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_calculate_fairness_metrics_returns_dict(self):
        candidates = [_make_candidate(f"P {i}", float(i) / 10, i + 1) for i in range(5)]
        metrics = self.checker.calculate_fairness_metrics(candidates)
        assert isinstance(metrics, dict)

    def test_empty_candidates_returns_empty_dict(self):
        metrics = self.checker.calculate_fairness_metrics([])
        assert metrics == {}


class TestAgeGroupSimulation:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_under_30_group(self):
        group = self.checker._simulate_age_group(1)
        assert group == "under_30"

    def test_thirty_to_forty_group(self):
        group = self.checker._simulate_age_group(5)
        assert group == "30_40"

    def test_forty_to_fifty_group(self):
        group = self.checker._simulate_age_group(10)
        assert group == "40_50"

    def test_over_fifty_group(self):
        group = self.checker._simulate_age_group(20)
        assert group == "over_50"


class TestSuggestAdjustments:

    def setup_method(self):
        self.checker = FairnessChecker()

    def test_no_bias_returns_no_adjustment_suggestion(self):
        candidates = [_make_candidate(f"P {i}", 0.7, i + 1) for i in range(5)]
        report = self.checker.generate_fairness_report(candidates, top_k=3)
        report.bias_flags = []  # Force no flags
        suggestions = self.checker.suggest_adjustments(candidates, report)
        assert isinstance(suggestions, list)
        assert any("No fairness adjustments" in s for s in suggestions)
