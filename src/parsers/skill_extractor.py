"""Skill extraction and normalization from resume text."""

import re
import logging
from typing import List, Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillExtractor:
    """Extracts and normalizes skills from resume text."""
    
    def __init__(self):
        """Initialize skill extractor with skill databases."""
        self.technical_skills = self._load_technical_skills()
        self.soft_skills = self._load_soft_skills()
        self.skill_synonyms = self._load_skill_synonyms()
        
        # Compile regex patterns for better performance
        self._compile_skill_patterns()
    
    def extract_skills(self, text: str, sections: Dict[str, str] = None) -> List[str]:
        """
        Extract skills from resume text.
        
        Args:
            text: Raw resume text
            sections: Parsed resume sections (optional)
            
        Returns:
            List of normalized skill names
        """
        found_skills = set()
        
        # Extract from skills section first (higher confidence)
        if sections and 'skills' in sections:
            skills_text = sections['skills']
            found_skills.update(self._extract_from_skills_section(skills_text))
        
        # Extract from full text
        found_skills.update(self._extract_from_full_text(text))
        
        # Extract from experience descriptions
        if sections and 'experience' in sections:
            exp_text = sections['experience']
            found_skills.update(self._extract_from_experience(exp_text))
        
        # Normalize and deduplicate
        normalized_skills = self._normalize_skills(list(found_skills))
        
        # Sort by relevance/frequency
        return self._rank_skills(normalized_skills, text)
    
    def _extract_from_skills_section(self, skills_text: str) -> Set[str]:
        """Extract skills from dedicated skills section."""
        found_skills = set()
        
        # Common separators in skills sections
        separators = r'[,;|•\\n\\t]+'
        skill_items = re.split(separators, skills_text)
        
        for item in skill_items:
            item = item.strip()
            if not item:
                continue
            
            # Clean up common prefixes/suffixes
            item = re.sub(r'^[-•\\s]+', '', item)
            item = re.sub(r'\\s*\\([^)]*\\)\\s*$', '', item)  # Remove parenthetical info
            
            # Check if it's a known skill
            if self._is_valid_skill(item):
                found_skills.add(item)
            
            # Also check individual words for compound skills
            words = item.split()
            for word in words:
                if len(word) > 2 and self._is_valid_skill(word):
                    found_skills.add(word)
        
        return found_skills
    
    def _extract_from_full_text(self, text: str) -> Set[str]:
        """Extract skills from full resume text using pattern matching."""
        found_skills = set()
        
        # Use pre-compiled patterns for technical skills
        for skill_pattern in self.technical_skill_patterns:
            matches = skill_pattern.findall(text.lower())
            found_skills.update(matches)
        
        # Look for programming languages with common patterns
        prog_patterns = [
            r'\\b(python|java|javascript|c\\+\\+|c#|php|ruby|go|rust|swift|kotlin)\\b',
            r'\\b(html|css|sql|r|matlab|scala|perl|shell|bash)\\b',
        ]
        
        for pattern in prog_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            found_skills.update(matches)
        
        return found_skills
    
    def _extract_from_experience(self, experience_text: str) -> Set[str]:
        """Extract skills mentioned in experience descriptions."""
        found_skills = set()
        
        # Look for skills in context of usage
        context_patterns = [
            r'(?:using|with|in|developed|built|implemented|worked with)\\s+([^\\n.]{1,30})',
            r'(?:experience with|proficient in|skilled in)\\s+([^\\n.]{1,30})',
            r'(?:technologies|tools|frameworks)\\s*:?\\s*([^\\n.]{1,50})'
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, experience_text, re.IGNORECASE)
            for match in matches:
                # Extract individual skills from the match
                potential_skills = re.split(r'[,;\\s]+', match.strip())
                for skill in potential_skills:
                    skill = skill.strip()
                    if self._is_valid_skill(skill):
                        found_skills.add(skill)
        
        return found_skills
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Check if a string represents a valid skill."""
        skill_lower = skill.lower().strip()
        
        # Skip very short or very long strings
        if len(skill_lower) < 2 or len(skill_lower) > 30:
            return False
        
        # Skip common non-skill words
        non_skills = {
            'and', 'or', 'the', 'with', 'for', 'in', 'on', 'at', 'to', 'of',
            'experience', 'years', 'work', 'working', 'development', 'using'
        }
        
        if skill_lower in non_skills:
            return False
        
        # Check against known skill databases
        return (skill_lower in self.technical_skills or 
                skill_lower in self.soft_skills or
                skill_lower in self.skill_synonyms)
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names using synonyms and standard forms."""
        normalized = []
        
        for skill in skills:
            skill_lower = skill.lower().strip()
            
            # Check for synonyms
            if skill_lower in self.skill_synonyms:
                canonical_skill = self.skill_synonyms[skill_lower]
                if canonical_skill not in normalized:
                    normalized.append(canonical_skill)
            else:
                # Use original case if not in synonyms
                if skill not in normalized:
                    normalized.append(skill)
        
        return normalized
    
    def _rank_skills(self, skills: List[str], text: str) -> List[str]:
        """Rank skills by relevance and frequency in text."""
        skill_scores = {}
        text_lower = text.lower()
        
        for skill in skills:
            score = 0
            skill_lower = skill.lower()
            
            # Count occurrences
            occurrences = len(re.findall(r'\\b' + re.escape(skill_lower) + r'\\b', text_lower))
            score += occurrences
            
            # Boost score for technical skills
            if skill_lower in self.technical_skills:
                score += 2
            
            # Boost score for skills mentioned in skills section
            if 'skill' in text_lower and skill_lower in text_lower:
                score += 1
            
            skill_scores[skill] = score
        
        # Sort by score (descending) and return
        return sorted(skills, key=lambda x: skill_scores.get(x, 0), reverse=True)
    
    def _load_technical_skills(self) -> Set[str]:
        """Load technical skills database."""
        return {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell',
            'bash', 'powershell', 'sql', 'html', 'css', 'sass', 'less',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'svelte', 'node.js', 'express', 'django',
            'flask', 'fastapi', 'spring', 'laravel', 'rails', 'asp.net', 'jquery',
            'bootstrap', 'tailwind', 'material-ui', 'redux', 'vuex', 'mobx',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
            'oracle', 'sql server', 'cassandra', 'dynamodb', 'neo4j',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
            'github actions', 'terraform', 'ansible', 'chef', 'puppet', 'vagrant',
            'nginx', 'apache', 'linux', 'ubuntu', 'centos', 'debian',
            
            # Data Science & ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'jupyter', 'spark', 'hadoop', 'kafka', 'airflow', 'mlflow',
            
            # Tools & Technologies
            'git', 'svn', 'jira', 'confluence', 'slack', 'teams', 'zoom',
            'postman', 'swagger', 'graphql', 'rest api', 'soap', 'json', 'xml',
            'yaml', 'markdown', 'latex', 'figma', 'sketch', 'photoshop',
            
            # Testing
            'pytest', 'junit', 'jest', 'mocha', 'selenium', 'cypress', 'postman',
            'unit testing', 'integration testing', 'tdd', 'bdd',
            
            # Methodologies
            'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd', 'microservices',
            'serverless', 'event-driven', 'domain-driven design', 'clean architecture'
        }
    
    def _load_soft_skills(self) -> Set[str]:
        """Load soft skills database."""
        return {
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'creative', 'adaptable', 'organized', 'detail-oriented', 'time management',
            'project management', 'critical thinking', 'collaboration', 'mentoring',
            'presentation', 'negotiation', 'customer service', 'sales', 'marketing',
            'strategic planning', 'decision making', 'conflict resolution', 'coaching'
        }
    
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
    
    def _compile_skill_patterns(self):
        """Compile regex patterns for technical skills."""
        self.technical_skill_patterns = []
        
        # Create patterns for each technical skill
        for skill in self.technical_skills:
            # Escape special regex characters
            escaped_skill = re.escape(skill)
            pattern = re.compile(r'\\b' + escaped_skill + r'\\b', re.IGNORECASE)
            self.technical_skill_patterns.append(pattern)
    
    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different types."""
        categories = {
            'programming_languages': [],
            'frameworks': [],
            'databases': [],
            'cloud_devops': [],
            'data_science': [],
            'tools': [],
            'soft_skills': [],
            'other': []
        }
        
        # Define category keywords
        category_keywords = {
            'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'rails'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle'],
            'cloud_devops': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'data_science': ['machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
            'tools': ['git', 'jira', 'postman', 'figma', 'photoshop']
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            for category, keywords in category_keywords.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categories[category].append(skill)
                    categorized = True
                    break
            
            if not categorized:
                if skill_lower in self.soft_skills:
                    categories['soft_skills'].append(skill)
                else:
                    categories['other'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}