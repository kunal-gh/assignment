"""Fairness checking and bias detection for resume rankings."""

import logging
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..models.ranking import FairnessReport, RankedCandidate

logger = logging.getLogger(__name__)


class FairnessChecker:
    """Checks for bias and fairness issues in resume rankings."""
    
    def __init__(self):
        """Initialize fairness checker."""
        self.protected_attributes = [
            'gender', 'ethnicity', 'age_group', 'education_level'
        ]
        self.four_fifths_threshold = 0.8
        
    def generate_fairness_report(self, candidates: List[RankedCandidate], 
                               top_k: int = 10) -> FairnessReport:
        """
        Generate comprehensive fairness analysis report.
        
        Args:
            candidates: List of ranked candidates
            top_k: Number of top candidates to analyze
            
        Returns:
            FairnessReport with bias analysis
        """
        if not candidates:
            return FairnessReport(
                total_candidates=0,
                top_k=0,
                bias_flags=["No candidates to analyze"]
            )
        
        top_k = min(top_k, len(candidates))
        top_candidates = candidates[:top_k]
        
        report = FairnessReport(
            total_candidates=len(candidates),
            top_k=top_k
        )
        
        # Extract demographic information (simulated for demo)
        demographics = self._extract_demographics(candidates)
        
        if not demographics:
            report.add_bias_flag("No demographic data available for fairness analysis")
            report.add_recommendation("Consider collecting demographic data for bias monitoring")
            return report
        
        # Check demographic parity
        parity_results = self.check_demographic_parity(candidates, top_k, demographics)
        report.demographic_parity = parity_results
        
        # Apply four-fifths rule
        violations = self.check_four_fifths_rule(parity_results)
        report.four_fifths_violations = violations
        
        # Generate bias flags
        for violation in violations:
            report.add_bias_flag(f"Four-fifths rule violation: {violation}")
        
        # Check for score disparities
        score_disparities = self._check_score_disparities(candidates, demographics)
        for disparity in score_disparities:
            report.add_bias_flag(disparity)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(report, demographics)
        for rec in recommendations:
            report.add_recommendation(rec)
        
        logger.info(f"Fairness analysis completed. Found {len(report.bias_flags)} potential issues.")
        
        return report
    
    def check_demographic_parity(self, candidates: List[RankedCandidate], 
                                top_k: int, demographics: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
        """
        Check for demographic parity in top-k candidates.
        
        Args:
            candidates: All candidates
            top_k: Number of top candidates
            demographics: Demographic information
            
        Returns:
            Dictionary with parity ratios for each group
        """
        parity_results = {}
        top_candidates = candidates[:top_k]
        
        for attribute, groups in demographics.items():
            for group_name, candidate_ids in groups.items():
                if not candidate_ids:
                    continue
                
                # Count group members in top-k
                top_group_count = sum(
                    1 for candidate in top_candidates
                    if candidate.resume.candidate_id in candidate_ids
                )
                
                # Calculate selection rates
                group_selection_rate = top_group_count / len(candidate_ids)
                overall_selection_rate = top_k / len(candidates)
                
                # Calculate parity ratio
                if overall_selection_rate > 0:
                    parity_ratio = group_selection_rate / overall_selection_rate
                else:
                    parity_ratio = 0.0
                
                parity_results[f"{attribute}_{group_name}"] = parity_ratio
        
        return parity_results
    
    def check_four_fifths_rule(self, parity_results: Dict[str, float]) -> List[str]:
        """
        Apply four-fifths rule to detect bias.
        
        Args:
            parity_results: Parity ratios from demographic analysis
            
        Returns:
            List of four-fifths rule violations
        """
        violations = []
        
        for group, parity_ratio in parity_results.items():
            if parity_ratio < self.four_fifths_threshold:
                violations.append(
                    f"{group}: {parity_ratio:.2f} (below {self.four_fifths_threshold} threshold)"
                )
        
        return violations
    
    def suggest_adjustments(self, candidates: List[RankedCandidate], 
                          fairness_report: FairnessReport) -> List[str]:
        """
        Suggest ranking adjustments to improve fairness.
        
        Args:
            candidates: Current rankings
            fairness_report: Fairness analysis results
            
        Returns:
            List of adjustment suggestions
        """
        suggestions = []
        
        if not fairness_report.has_bias_flags():
            suggestions.append("No fairness adjustments needed - rankings appear unbiased")
            return suggestions
        
        # Analyze score distributions
        demographics = self._extract_demographics(candidates)
        
        if demographics:
            # Suggest promoting underrepresented candidates with competitive scores
            underrepresented_groups = self._identify_underrepresented_groups(
                candidates, demographics, fairness_report
            )
            
            for group in underrepresented_groups:
                suggestions.append(
                    f"Consider reviewing high-scoring candidates from {group} "
                    "for potential promotion in rankings"
                )
        
        # General suggestions
        suggestions.extend([
            "Review job requirements to ensure they don't inadvertently exclude qualified candidates",
            "Consider blind resume review to reduce unconscious bias",
            "Implement diverse interview panels for shortlisted candidates",
            "Track hiring outcomes to monitor long-term fairness trends"
        ])
        
        return suggestions
    
    def _extract_demographics(self, candidates: List[RankedCandidate]) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract demographic information from candidates.
        
        Note: In a real implementation, this would extract actual demographic data.
        For this demo, we simulate demographic data based on names and other heuristics.
        """
        demographics = {
            'gender': {'male': [], 'female': [], 'other': []},
            'ethnicity': {'asian': [], 'black': [], 'hispanic': [], 'white': [], 'other': []},
            'age_group': {'under_30': [], '30_40': [], '40_50': [], 'over_50': []},
            'education_level': {'bachelors': [], 'masters': [], 'phd': [], 'other': []}
        }
        
        # Simulate demographic data (in real implementation, this would come from actual data)
        for candidate in candidates:
            candidate_id = candidate.resume.candidate_id
            name = candidate.resume.contact_info.name if candidate.resume.contact_info else "Unknown"
            
            # Simulate gender based on name patterns (very rough heuristic)
            gender = self._simulate_gender(name)
            demographics['gender'][gender].append(candidate_id)
            
            # Simulate ethnicity (random distribution for demo)
            ethnicity = self._simulate_ethnicity(name)
            demographics['ethnicity'][ethnicity].append(candidate_id)
            
            # Simulate age group based on experience
            age_group = self._simulate_age_group(candidate.resume.get_years_of_experience())
            demographics['age_group'][age_group].append(candidate_id)
            
            # Simulate education level based on education data
            education_level = self._simulate_education_level(candidate.resume.education)
            demographics['education_level'][education_level].append(candidate_id)
        
        return demographics
    
    def _simulate_gender(self, name: str) -> str:
        """Simulate gender based on name (for demo purposes only)."""
        if not name or name == "Unknown":
            return 'other'
        
        # Very basic heuristic - in real implementation, use proper demographic data
        name_lower = name.lower()
        
        # Common female name endings/patterns
        if any(name_lower.endswith(suffix) for suffix in ['a', 'ia', 'ina', 'ara', 'lyn']):
            return 'female'
        
        # Common male name patterns
        if any(name_lower.endswith(suffix) for suffix in ['er', 'on', 'an', 'el']):
            return 'male'
        
        # Default to other if uncertain
        return 'other'
    
    def _simulate_ethnicity(self, name: str) -> str:
        """Simulate ethnicity (for demo purposes only)."""
        # In real implementation, this should not be inferred from names
        # This is just for demonstration of fairness checking
        return 'other'  # Default to avoid making assumptions
    
    def _simulate_age_group(self, years_experience: int) -> str:
        """Simulate age group based on years of experience."""
        if years_experience < 3:
            return 'under_30'
        elif years_experience < 8:
            return '30_40'
        elif years_experience < 15:
            return '40_50'
        else:
            return 'over_50'
    
    def _simulate_education_level(self, education_list) -> str:
        """Simulate education level based on education data."""
        if not education_list:
            return 'other'
        
        # Look for highest degree
        degrees = [edu.degree.lower() if edu.degree else '' for edu in education_list]
        
        if any('phd' in degree or 'ph.d' in degree or 'doctorate' in degree for degree in degrees):
            return 'phd'
        elif any('master' in degree or 'mba' in degree or 'm.s' in degree or 'm.a' in degree for degree in degrees):
            return 'masters'
        elif any('bachelor' in degree or 'b.s' in degree or 'b.a' in degree for degree in degrees):
            return 'bachelors'
        else:
            return 'other'
    
    def _check_score_disparities(self, candidates: List[RankedCandidate], 
                               demographics: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Check for significant score disparities between demographic groups."""
        disparities = []
        
        for attribute, groups in demographics.items():
            group_scores = {}
            
            # Calculate average scores for each group
            for group_name, candidate_ids in groups.items():
                if not candidate_ids:
                    continue
                
                scores = [
                    candidate.hybrid_score for candidate in candidates
                    if candidate.resume.candidate_id in candidate_ids
                ]
                
                if scores:
                    group_scores[group_name] = statistics.mean(scores)
            
            # Check for significant disparities
            if len(group_scores) > 1:
                max_score = max(group_scores.values())
                min_score = min(group_scores.values())
                
                # Flag if difference is more than 20%
                if max_score > 0 and (max_score - min_score) / max_score > 0.2:
                    disparities.append(
                        f"Significant score disparity in {attribute}: "
                        f"range {min_score:.3f} to {max_score:.3f}"
                    )
        
        return disparities
    
    def _generate_recommendations(self, report: FairnessReport, 
                                demographics: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []
        
        if report.has_bias_flags():
            recommendations.extend([
                "Review job requirements to ensure they are truly necessary and don't create barriers",
                "Consider implementing blind resume screening to reduce unconscious bias",
                "Ensure diverse representation in the hiring team and decision-making process",
                "Provide bias training for recruiters and hiring managers"
            ])
        
        if report.four_fifths_violations:
            recommendations.append(
                "Monitor hiring outcomes over time to ensure consistent fairness"
            )
        
        # Add specific recommendations based on violations
        for violation in report.four_fifths_violations:
            if 'gender' in violation:
                recommendations.append(
                    "Review job descriptions and requirements for gender-neutral language"
                )
            elif 'ethnicity' in violation:
                recommendations.append(
                    "Expand recruitment channels to reach more diverse candidate pools"
                )
            elif 'education' in violation:
                recommendations.append(
                    "Consider whether education requirements are essential or could be replaced with experience"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_underrepresented_groups(self, candidates: List[RankedCandidate],
                                        demographics: Dict[str, Dict[str, List[str]]],
                                        fairness_report: FairnessReport) -> List[str]:
        """Identify underrepresented groups in top rankings."""
        underrepresented = []
        
        for violation in fairness_report.four_fifths_violations:
            # Extract group name from violation string
            group_name = violation.split(':')[0]
            underrepresented.append(group_name)
        
        return underrepresented
    
    def calculate_fairness_metrics(self, candidates: List[RankedCandidate]) -> Dict[str, float]:
        """Calculate various fairness metrics."""
        if not candidates:
            return {}
        
        demographics = self._extract_demographics(candidates)
        
        metrics = {}
        
        # Calculate demographic parity for top 10
        top_10_parity = self.check_demographic_parity(candidates, 10, demographics)
        metrics.update(top_10_parity)
        
        # Calculate overall fairness score
        if top_10_parity:
            parity_values = [v for v in top_10_parity.values() if v > 0]
            if parity_values:
                # Fairness score based on how close parity ratios are to 1.0
                fairness_deviations = [abs(1.0 - ratio) for ratio in parity_values]
                avg_deviation = statistics.mean(fairness_deviations)
                fairness_score = max(0.0, 1.0 - avg_deviation)
                metrics['overall_fairness_score'] = fairness_score
        
        return metrics