"""Skill matching and analysis functionality."""

import logging
from typing import List, Dict, Set, Any
import re

logger = logging.getLogger(__name__)


class SkillMatcher:
    """Handles skill matching and analysis between resumes and job requirements."""
    
    def __init__(self):
        """Initialize skill matcher with normalization rules."""
        self.skill_synonyms = self._load_skill_synonyms()
        self.skill_categories = self._load_skill_categories()
        
    def calculate_skill_match(self, resume_skills: List[str], 
                            required_skills: List[str], 
                            preferred_skills: List[str] = None,
                            required_weight: float = 0.7,
                            preferred_weight: float = 0.3) -> float:
        """
        Calculate skill matching score using weighted Jaccard similarity.
        
        Args:
            resume_skills: Skills from resume
            required_skills: Required skills from job
            preferred_skills: Preferred skills from job
            required_weight: Weight for required skills (default 0.7)
            preferred_weight: Weight for preferred skills (default 0.3)
            
        Returns:
            Skill matching score (0-1)
        """
        if not resume_skills:
            return 0.0
        
        if not required_skills and not preferred_skills:
            return 0.0
        
        preferred_skills = preferred_skills or []
        
        # Normalize skills using synonyms and standard forms
        resume_skills_norm = self._normalize_skills(resume_skills)
        required_skills_norm = self._normalize_skills(required_skills)
        preferred_skills_norm = self._normalize_skills(preferred_skills)
        
        # Convert to sets for efficient set operations
        resume_set = set(resume_skills_norm)
        required_set = set(required_skills_norm)
        preferred_set = set(preferred_skills_norm)
        
        # Calculate Jaccard similarity for required skills
        required_score = 0.0
        if required_set:
            required_intersection = resume_set.intersection(required_set)
            required_union = resume_set.union(required_set)
            required_score = len(required_intersection) / len(required_union) if required_union else 0.0
            
            # Apply coverage bonus for high matches in required skills
            required_coverage = len(required_intersection) / len(required_set)
            if required_coverage >= 0.8:  # 80%+ coverage gets bonus
                coverage_bonus = (required_coverage - 0.8) * 0.5  # Up to 10% bonus
                required_score = min(1.0, required_score + coverage_bonus)
        
        # Calculate Jaccard similarity for preferred skills
        preferred_score = 0.0
        if preferred_set:
            preferred_intersection = resume_set.intersection(preferred_set)
            preferred_union = resume_set.union(preferred_set)
            preferred_score = len(preferred_intersection) / len(preferred_union) if preferred_union else 0.0
        
        # Coverage-based scores for weighted combination
        # Using precision-style coverage: what fraction of job skills does the candidate have?
        required_coverage_score = 0.0
        if required_set:
            required_matched = resume_set.intersection(required_set)
            required_coverage_score = len(required_matched) / len(required_set)
            # Apply coverage bonus for high matches in required skills (80%+ gets bonus)
            if required_coverage_score >= 0.8:
                coverage_bonus = (required_coverage_score - 0.8) * 0.5
                required_coverage_score = min(1.0, required_coverage_score + coverage_bonus)

        preferred_coverage_score = 0.0
        if preferred_set:
            preferred_matched = resume_set.intersection(preferred_set)
            preferred_coverage_score = len(preferred_matched) / len(preferred_set)

        # Weighted combination of scores
        if required_set and preferred_set:
            # Both required and preferred skills exist — use coverage-based scoring
            # so that different weights produce meaningfully different results
            final_score = (required_weight * required_coverage_score) + (preferred_weight * preferred_coverage_score)
        elif required_set:
            # Only required skills exist — use Jaccard for consistency
            final_score = required_score
        else:
            # Only preferred skills exist — use Jaccard for consistency
            final_score = preferred_score
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, final_score))
    
    def analyze_skill_match(self, resume_skills: List[str], 
                          required_skills: List[str], 
                          preferred_skills: List[str] = None) -> Dict[str, Any]:
        """
        Provide detailed analysis of skill matching.
        
        Args:
            resume_skills: Skills from resume
            required_skills: Required skills from job
            preferred_skills: Preferred skills from job
            
        Returns:
            Dictionary with detailed skill analysis
        """
        preferred_skills = preferred_skills or []
        
        # Normalize skills
        resume_skills_norm = self._normalize_skills(resume_skills)
        required_skills_norm = self._normalize_skills(required_skills)
        preferred_skills_norm = self._normalize_skills(preferred_skills)
        
        # Convert to sets
        resume_set = set(resume_skills_norm)
        required_set = set(required_skills_norm)
        preferred_set = set(preferred_skills_norm)
        
        # Find matches and gaps
        matched_required = list(resume_set.intersection(required_set))
        matched_preferred = list(resume_set.intersection(preferred_set))
        missing_required = list(required_set - resume_set)
        missing_preferred = list(preferred_set - resume_set)
        extra_skills = list(resume_set - required_set - preferred_set)
        
        # Calculate coverage percentages
        required_coverage = len(matched_required) / len(required_set) if required_set else 0
        preferred_coverage = len(matched_preferred) / len(preferred_set) if preferred_set else 0
        
        # Categorize skills
        skill_categories = self._categorize_skills(resume_skills_norm)
        
        return {
            'matched_required': matched_required,
            'matched_preferred': matched_preferred,
            'missing_required': missing_required,
            'missing_preferred': missing_preferred,
            'extra_skills': extra_skills,
            'required_coverage': required_coverage,
            'preferred_coverage': preferred_coverage,
            'total_resume_skills': len(resume_skills_norm),
            'skill_categories': skill_categories,
            'overall_score': self.calculate_skill_match(resume_skills, required_skills, preferred_skills)
        }
    
    def find_skill_gaps(self, resume_skills: List[str], job_skills: List[str]) -> Dict[str, List[str]]:
        """
        Identify skill gaps between resume and job requirements.
        
        Args:
            resume_skills: Skills from resume
            job_skills: Skills required for job
            
        Returns:
            Dictionary with skill gaps by category
        """
        resume_skills_norm = set(self._normalize_skills(resume_skills))
        job_skills_norm = set(self._normalize_skills(job_skills))
        
        # Find missing skills
        missing_skills = job_skills_norm - resume_skills_norm
        
        # Categorize missing skills
        gaps_by_category = {}
        for skill in missing_skills:
            category = self._get_skill_category(skill)
            if category not in gaps_by_category:
                gaps_by_category[category] = []
            gaps_by_category[category].append(skill)
        
        return gaps_by_category
    
    def suggest_skill_improvements(self, resume_skills: List[str], 
                                 required_skills: List[str]) -> List[str]:
        """
        Suggest skills to add based on job requirements.
        
        Args:
            resume_skills: Current skills from resume
            required_skills: Required skills from job
            
        Returns:
            List of suggested skills to learn/add
        """
        analysis = self.analyze_skill_match(resume_skills, required_skills)
        suggestions = []
        
        # Prioritize missing required skills
        missing_required = analysis['missing_required']
        
        # Sort by importance/frequency
        skill_importance = self._get_skill_importance()
        
        sorted_missing = sorted(
            missing_required,
            key=lambda skill: skill_importance.get(skill, 0),
            reverse=True
        )
        
        # Add top suggestions
        suggestions.extend(sorted_missing[:5])  # Top 5 missing required skills
        
        # Add related skills that might be valuable
        for skill in sorted_missing[:3]:  # For top 3 missing skills
            related = self._get_related_skills(skill)
            for related_skill in related:
                if (related_skill not in resume_skills and 
                    related_skill not in suggestions and 
                    len(suggestions) < 10):
                    suggestions.append(related_skill)
        
        return suggestions
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names using synonyms and standard forms."""
        normalized = []
        
        for skill in skills:
            if not skill:
                continue
                
            skill_lower = skill.lower().strip()
            
            # Check for synonyms
            if skill_lower in self.skill_synonyms:
                canonical_skill = self.skill_synonyms[skill_lower]
                if canonical_skill not in normalized:
                    normalized.append(canonical_skill)
            else:
                # Use lowercase version for consistency
                if skill_lower not in normalized:
                    normalized.append(skill_lower)
        
        return normalized
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different types."""
        categories = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'databases': [],
            'cloud_devops': [],
            'data_science': [],
            'tools': [],
            'soft_skills': [],
            'other': []
        }
        
        for skill in skills:
            category = self._get_skill_category(skill)
            categories[category].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _get_skill_category(self, skill: str) -> str:
        """Get category for a specific skill."""
        skill_lower = skill.lower()
        
        # Check each category for exact matches first
        for category, skills_list in self.skill_categories.items():
            if skill_lower in [s.lower() for s in skills_list]:
                return category
        
        # Fallback to keyword matching
        for category, skills_list in self.skill_categories.items():
            if any(keyword.lower() in skill_lower for keyword in skills_list):
                return category
        
        return 'other'
    
    def _get_skill_importance(self) -> Dict[str, int]:
        """Get importance scores for skills (higher = more important)."""
        return {
            # High importance
            'python': 10, 'java': 10, 'javascript': 10, 'sql': 10,
            'react': 9, 'node.js': 9, 'aws': 9, 'docker': 9,
            'machine learning': 9, 'git': 9,
            
            # Medium importance
            'angular': 8, 'vue': 8, 'django': 8, 'flask': 8,
            'postgresql': 8, 'mongodb': 8, 'kubernetes': 8,
            'tensorflow': 8, 'pandas': 8,
            
            # Lower importance
            'jquery': 6, 'bootstrap': 6, 'sass': 6,
            'redis': 7, 'elasticsearch': 7,
            
            # Soft skills
            'communication': 8, 'leadership': 8, 'teamwork': 8,
            'problem solving': 8, 'project management': 8
        }
    
    def _get_related_skills(self, skill: str) -> List[str]:
        """Get skills related to the given skill."""
        skill_lower = skill.lower()
        
        related_skills_map = {
            'python': ['django', 'flask', 'pandas', 'numpy', 'scikit-learn'],
            'javascript': ['react', 'angular', 'vue', 'node.js', 'typescript'],
            'java': ['spring', 'hibernate', 'maven', 'gradle'],
            'react': ['redux', 'jsx', 'webpack', 'babel'],
            'aws': ['ec2', 's3', 'lambda', 'cloudformation', 'terraform'],
            'docker': ['kubernetes', 'docker-compose', 'containerization'],
            'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'pandas'],
            'sql': ['postgresql', 'mysql', 'database design', 'data modeling'],
            'git': ['github', 'gitlab', 'version control', 'ci/cd']
        }
        
        return related_skills_map.get(skill_lower, [])
    
    def _load_skill_synonyms(self) -> Dict[str, str]:
        """Load skill synonyms for normalization."""
        return {
            # Programming language synonyms
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'c++': 'cpp',
            'c#': 'csharp',
            'c sharp': 'csharp',
            'node': 'node.js',
            'nodejs': 'node.js',
            
            # Framework synonyms
            'reactjs': 'react',
            'angularjs': 'angular',
            'vuejs': 'vue',
            'express.js': 'express',
            'django rest framework': 'django',
            'spring boot': 'spring',
            'ruby on rails': 'rails',
            
            # Database synonyms
            'postgres': 'postgresql',
            'mongo': 'mongodb',
            'elastic': 'elasticsearch',
            'mssql': 'sql server',
            
            # Cloud synonyms
            'amazon web services': 'aws',
            'google cloud platform': 'gcp',
            'google cloud': 'gcp',
            'microsoft azure': 'azure',
            
            # Tool synonyms
            'github': 'git',
            'gitlab': 'git',
            'bitbucket': 'git',
            'k8s': 'kubernetes',
            'kube': 'kubernetes',
            
            # ML synonyms
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            
            # Methodology synonyms
            'agile methodology': 'agile',
            'scrum methodology': 'scrum',
            'continuous integration': 'ci/cd',
            'continuous deployment': 'ci/cd'
        }
    
    def _load_skill_categories(self) -> Dict[str, List[str]]:
        """Load skill category mappings."""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'csharp', 
                'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r'
            ],
            'frameworks_libraries': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'rails', 'laravel', 'asp.net', 'jquery', 'bootstrap', 'tailwind'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'sqlite', 'oracle', 'sql server', 'cassandra', 'dynamodb'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'chef', 'puppet', 'nginx', 'apache'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'jupyter',
                'spark', 'hadoop', 'kafka'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'postman', 'swagger', 'figma',
                'photoshop', 'slack', 'teams', 'zoom'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'analytical', 'creative', 'adaptable'
            ]
        }