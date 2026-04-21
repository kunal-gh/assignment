"""Skill extraction and normalization from resume text.

Uses a combination of:
- Curated skill taxonomy (technical + soft skills)
- Normalization rules and synonym mapping
- spaCy NER for skill identification in free text
- Pattern matching for structured skills sections
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill Taxonomy
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGES: Set[str] = {
    "python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby",
    "go", "golang", "rust", "swift", "kotlin", "scala", "r", "matlab", "perl",
    "shell", "bash", "powershell", "sql", "html", "css", "sass", "less",
    "cobol", "fortran", "haskell", "erlang", "elixir", "clojure", "f#",
    "dart", "lua", "groovy", "assembly", "vba", "objective-c",
}

FRAMEWORKS_LIBRARIES: Set[str] = {
    # Frontend
    "react", "angular", "vue", "svelte", "ember", "backbone", "jquery",
    "bootstrap", "tailwind", "material-ui", "chakra-ui", "ant design",
    "redux", "vuex", "mobx", "next.js", "nuxt.js", "gatsby",
    # Backend
    "node.js", "express", "django", "flask", "fastapi", "spring", "spring boot",
    "laravel", "rails", "asp.net", "gin", "fiber", "echo", "actix",
    "nestjs", "koa", "hapi", "tornado", "aiohttp", "starlette",
    # Mobile
    "react native", "flutter", "xamarin", "ionic",
    # Data / ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "bokeh",
    "hugging face", "transformers", "langchain", "openai",
    # Testing
    "pytest", "junit", "jest", "mocha", "chai", "jasmine", "cypress",
    "selenium", "playwright", "testng", "rspec",
}

DATABASES: Set[str] = {
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
    "oracle", "sql server", "cassandra", "dynamodb", "neo4j", "couchdb",
    "mariadb", "cockroachdb", "influxdb", "clickhouse", "snowflake",
    "bigquery", "redshift", "hbase", "memcached", "firebase",
}

CLOUD_DEVOPS: Set[str] = {
    # Cloud providers
    "aws", "azure", "gcp", "heroku", "digitalocean", "linode", "cloudflare",
    # Container / orchestration
    "docker", "kubernetes", "helm", "istio", "openshift",
    # CI/CD
    "jenkins", "gitlab ci", "github actions", "circleci", "travis ci",
    "teamcity", "bamboo", "argocd", "spinnaker",
    # IaC
    "terraform", "ansible", "chef", "puppet", "vagrant", "pulumi",
    # Web servers
    "nginx", "apache", "caddy",
    # OS
    "linux", "ubuntu", "centos", "debian", "rhel", "windows server",
}

DATA_SCIENCE_ML: Set[str] = {
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "data science",
    "data engineering", "data analysis", "statistical modeling",
    "feature engineering", "model deployment", "mlops",
    "spark", "hadoop", "kafka", "airflow", "mlflow", "kubeflow",
    "jupyter", "databricks", "dbt", "looker", "tableau", "power bi",
}

TOOLS_TECHNOLOGIES: Set[str] = {
    "git", "svn", "mercurial", "jira", "confluence", "trello", "asana",
    "postman", "swagger", "graphql", "rest api", "soap", "grpc",
    "json", "xml", "yaml", "protobuf", "avro",
    "figma", "sketch", "adobe xd", "photoshop", "illustrator",
    "latex", "markdown", "vim", "emacs",
    "rabbitmq", "celery", "sidekiq", "kafka",
    "oauth", "jwt", "saml", "ldap",
}

METHODOLOGIES: Set[str] = {
    "agile", "scrum", "kanban", "waterfall", "lean", "xp",
    "devops", "ci/cd", "tdd", "bdd", "ddd",
    "microservices", "serverless", "event-driven architecture",
    "domain-driven design", "clean architecture", "solid",
    "rest", "soap", "graphql",
}

SOFT_SKILLS: Set[str] = {
    "leadership", "communication", "teamwork", "problem solving",
    "analytical thinking", "critical thinking", "creativity", "adaptability",
    "time management", "project management", "mentoring", "coaching",
    "presentation", "negotiation", "conflict resolution", "collaboration",
    "customer service", "strategic planning", "decision making",
    "attention to detail", "self-motivated", "organized",
}

# Combined technical skills set (used for fast lookup)
ALL_TECHNICAL_SKILLS: Set[str] = (
    PROGRAMMING_LANGUAGES
    | FRAMEWORKS_LIBRARIES
    | DATABASES
    | CLOUD_DEVOPS
    | DATA_SCIENCE_ML
    | TOOLS_TECHNOLOGIES
    | METHODOLOGIES
)

ALL_SKILLS: Set[str] = ALL_TECHNICAL_SKILLS | SOFT_SKILLS


# ---------------------------------------------------------------------------
# Normalization / Synonym Map
# Maps lowercase variant -> canonical form
# ---------------------------------------------------------------------------

SKILL_SYNONYMS: Dict[str, str] = {
    # Programming languages
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "c plus plus": "c++",
    "cplusplus": "c++",
    "cpp": "c++",
    "c sharp": "c#",
    "csharp": "c#",
    "golang": "go",
    "node": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "objective c": "objective-c",
    "objc": "objective-c",
    # Frameworks
    "reactjs": "react",
    "react.js": "react",
    "angularjs": "angular",
    "angular.js": "angular",
    "vuejs": "vue",
    "vue.js": "vue",
    "express.js": "express",
    "expressjs": "express",
    "nextjs": "next.js",
    "next js": "next.js",
    "nuxtjs": "nuxt.js",
    "nuxt js": "nuxt.js",
    "django rest framework": "django",
    "drf": "django",
    "spring boot": "spring boot",
    "springboot": "spring boot",
    "ruby on rails": "rails",
    "ror": "rails",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "sk-learn": "scikit-learn",
    "hf": "hugging face",
    "huggingface": "hugging face",
    # Databases
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "elastic": "elasticsearch",
    "es": "elasticsearch",
    "mssql": "sql server",
    "ms sql": "sql server",
    "microsoft sql server": "sql server",
    "dynamo": "dynamodb",
    "dynamo db": "dynamodb",
    # Cloud
    "amazon web services": "aws",
    "amazon aws": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "google gcp": "gcp",
    "microsoft azure": "azure",
    "azure cloud": "azure",
    # DevOps / tools
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "gh actions": "github actions",
    "gitlab-ci": "gitlab ci",
    "circle ci": "circleci",
    "travis": "travis ci",
    "tf": "terraform",
    # ML / AI
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "llm": "large language models",
    "genai": "generative ai",
    "gen ai": "generative ai",
    # Methodologies
    "agile methodology": "agile",
    "scrum methodology": "scrum",
    "continuous integration": "ci/cd",
    "continuous deployment": "ci/cd",
    "continuous delivery": "ci/cd",
    "test driven development": "tdd",
    "behavior driven development": "bdd",
    "domain driven design": "ddd",
    # Tools
    "github": "git",
    "gitlab": "git",
    "bitbucket": "git",
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "rest apis": "rest api",
    "power bi": "power bi",
    "powerbi": "power bi",
}

# Category mapping for get_skill_categories()
_CATEGORY_SETS: Dict[str, Set[str]] = {
    "programming_languages": PROGRAMMING_LANGUAGES,
    "frameworks_libraries": FRAMEWORKS_LIBRARIES,
    "databases": DATABASES,
    "cloud_devops": CLOUD_DEVOPS,
    "data_science_ml": DATA_SCIENCE_ML,
    "tools_technologies": TOOLS_TECHNOLOGIES,
    "methodologies": METHODOLOGIES,
    "soft_skills": SOFT_SKILLS,
}


# ---------------------------------------------------------------------------
# SkillExtractor
# ---------------------------------------------------------------------------

class SkillExtractor:
    """Extracts and normalizes skills from resume text.

    Extraction pipeline:
    1. Structured skills section parsing (comma/bullet separated lists)
    2. spaCy NER on full text (PRODUCT, ORG, SKILL-like entities)
    3. Regex pattern matching against the skill taxonomy
    4. Context-aware extraction from experience descriptions
    5. Normalization via synonym map
    6. Deduplication and relevance ranking
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self._spacy_model = spacy_model
        self._load_spacy()
        self._skill_patterns = self._compile_skill_patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_skills(
        self,
        text: str,
        sections: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Extract and normalize skills from resume text.

        Args:
            text: Raw resume text.
            sections: Pre-parsed resume sections (optional but improves accuracy).

        Returns:
            Deduplicated, normalized list of skill names sorted by relevance.
        """
        # Allow empty text when sections are provided
        if (not text or not text.strip()) and not sections:
            return []

        text = text or ""
        found: Set[str] = set()

        # 1. Structured skills section (highest confidence)
        if sections and "skills" in sections:
            found.update(self._extract_from_skills_section(sections["skills"]))

        # 2. spaCy NER on full text
        found.update(self._extract_via_ner(text))

        # 3. Regex taxonomy matching on full text
        found.update(self._extract_via_patterns(text))

        # 4. Context patterns in experience section
        if sections and "experience" in sections:
            found.update(self._extract_from_experience(sections["experience"]))
        else:
            found.update(self._extract_from_experience(text))

        # 5. Normalize synonyms → canonical forms
        normalized = self._normalize_skills(found)

        # 6. Rank by frequency / category weight
        return self._rank_skills(normalized, text)

    def normalize_skill(self, skill: str) -> str:
        """Return the canonical form of a single skill string."""
        key = skill.lower().strip()
        return SKILL_SYNONYMS.get(key, skill.strip())

    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize a list of skills into taxonomy groups.

        Args:
            skills: List of (already normalized) skill names.

        Returns:
            Dict mapping category name to list of skills in that category.
            Skills that don't fit any category appear under 'other'.
        """
        result: Dict[str, List[str]] = {cat: [] for cat in _CATEGORY_SETS}
        result["other"] = []

        for skill in skills:
            skill_lower = skill.lower()
            placed = False
            for category, skill_set in _CATEGORY_SETS.items():
                if skill_lower in skill_set:
                    result[category].append(skill)
                    placed = True
                    break
            if not placed:
                result["other"].append(skill)

        # Remove empty categories
        return {k: v for k, v in result.items() if v}

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_from_skills_section(self, skills_text: str) -> Set[str]:
        """Parse a dedicated skills section (comma/bullet/pipe separated)."""
        found: Set[str] = set()
        if not skills_text:
            return found

        # Split on common separators
        items = re.split(r"[,;|\n\t•·\-–]+", skills_text)

        for raw in items:
            item = raw.strip().strip("•·-–*").strip()
            if not item:
                continue
            # Remove parenthetical annotations like "(3 years)"
            item = re.sub(r"\s*\([^)]*\)\s*", "", item).strip()
            if not item:
                continue

            # Check whole phrase first (e.g. "machine learning")
            if self._is_known_skill(item):
                found.add(item)
                continue

            # Fall back to checking individual tokens for compound items
            for word in item.split():
                word = word.strip(".,;:")
                if len(word) >= 2 and self._is_known_skill(word):
                    found.add(word)

        return found

    def _extract_via_ner(self, text: str) -> Set[str]:
        """Use spaCy NER to identify skill-like entities."""
        found: Set[str] = set()
        if self._nlp is None or not text:
            return found

        try:
            # Process in chunks to stay within spaCy's token limit
            for chunk in self._chunk_text(text, max_chars=50_000):
                doc = self._nlp(chunk)
                for ent in doc.ents:
                    # PRODUCT and ORG entities often correspond to technologies
                    if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART"):
                        candidate = ent.text.strip()
                        if self._is_known_skill(candidate):
                            found.add(candidate)
        except Exception as exc:
            logger.debug("spaCy NER failed during skill extraction: %s", exc)

        return found

    def _extract_via_patterns(self, text: str) -> Set[str]:
        """Match skills against pre-compiled regex patterns."""
        found: Set[str] = set()
        if not text:
            return found

        text_lower = text.lower()
        for skill, pattern in self._skill_patterns:
            if pattern.search(text_lower):
                found.add(skill)

        return found

    def _extract_from_experience(self, experience_text: str) -> Set[str]:
        """Extract skills mentioned in context within experience descriptions."""
        found: Set[str] = set()
        if not experience_text:
            return found

        # Context trigger phrases
        context_patterns = [
            r"(?:using|with|via|through|leveraging)\s+([^.\n]{1,60})",
            r"(?:developed|built|implemented|designed|architected)\s+(?:with\s+)?([^.\n]{1,60})",
            r"(?:experience\s+(?:with|in)|proficient\s+in|skilled\s+in|expertise\s+in)\s+([^.\n]{1,60})",
            r"(?:technologies|tools|frameworks|stack)\s*:?\s*([^.\n]{1,80})",
            r"(?:languages?|platforms?)\s*:?\s*([^.\n]{1,60})",
        ]

        for pat in context_patterns:
            for match in re.finditer(pat, experience_text, re.IGNORECASE):
                phrase = match.group(1).strip()
                # Split phrase into candidate tokens
                for candidate in re.split(r"[,;/\s]+", phrase):
                    candidate = candidate.strip(".,;:()")
                    if self._is_known_skill(candidate):
                        found.add(candidate)
                # Also check the whole phrase
                if self._is_known_skill(phrase):
                    found.add(phrase)

        return found

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_skills(self, skills: Set[str]) -> List[str]:
        """Map skill variants to canonical forms and deduplicate."""
        canonical: Dict[str, str] = {}  # canonical_lower -> display form

        for skill in skills:
            key = skill.lower().strip()
            # Resolve synonym
            resolved = SKILL_SYNONYMS.get(key, skill.strip())
            resolved_lower = resolved.lower()

            if resolved_lower not in canonical:
                canonical[resolved_lower] = resolved

        return list(canonical.values())

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_skills(self, skills: List[str], text: str) -> List[str]:
        """Sort skills by relevance: frequency in text + category weight."""
        if not skills:
            return []
        
        text_lower = text.lower() if text else ""
        scores: Dict[str, float] = {}

        for skill in skills:
            skill_lower = skill.lower()
            # Frequency score
            freq = 0
            if text_lower:
                try:
                    freq = len(re.findall(r"\b" + re.escape(skill_lower) + r"\b", text_lower))
                except re.error:
                    freq = text_lower.count(skill_lower)

            # Category weight bonus
            weight = 1.0
            if skill_lower in PROGRAMMING_LANGUAGES:
                weight = 3.0
            elif skill_lower in FRAMEWORKS_LIBRARIES:
                weight = 2.5
            elif skill_lower in DATABASES:
                weight = 2.0
            elif skill_lower in CLOUD_DEVOPS:
                weight = 2.0
            elif skill_lower in DATA_SCIENCE_ML:
                weight = 2.5
            elif skill_lower in TOOLS_TECHNOLOGIES:
                weight = 1.5
            elif skill_lower in METHODOLOGIES:
                weight = 1.5

            scores[skill] = freq * weight + weight

        return sorted(skills, key=lambda s: scores.get(s, 0.0), reverse=True)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _is_known_skill(self, text: str) -> bool:
        """Return True if *text* (case-insensitive) is in the skill taxonomy or synonym map."""
        if not text:
            return False
        key = text.lower().strip()
        # Skip very short or very long strings
        if len(key) < 2 or len(key) > 50:
            return False
        return key in ALL_SKILLS or key in SKILL_SYNONYMS

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    def _compile_skill_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Pre-compile word-boundary patterns for every skill in the taxonomy."""
        patterns: List[Tuple[str, re.Pattern]] = []
        for skill in sorted(ALL_SKILLS, key=len, reverse=True):
            try:
                escaped = re.escape(skill)
                pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
                patterns.append((skill, pat))
            except re.error:
                logger.debug("Could not compile pattern for skill: %s", skill)
        return patterns

    # ------------------------------------------------------------------
    # spaCy loading
    # ------------------------------------------------------------------

    def _load_spacy(self) -> None:
        """Load spaCy model, falling back gracefully if unavailable."""
        try:
            import spacy
            self._nlp = spacy.load(self._spacy_model)
            logger.debug("Loaded spaCy model: %s", self._spacy_model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. NER-based skill extraction disabled.",
                self._spacy_model,
            )
            self._nlp = None
        except ImportError:
            logger.warning("spaCy not installed. NER-based skill extraction disabled.")
            self._nlp = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 50_000) -> List[str]:
        """Split text into chunks of at most *max_chars* characters."""
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
