"""Section identification and extraction from resume text using spaCy NLP and regex."""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section header patterns
# ---------------------------------------------------------------------------

# Maps canonical section key -> list of regex patterns that match headers
SECTION_HEADER_PATTERNS: Dict[str, List[str]] = {
    "contact": [
        r"contact\s*(?:information|info|details)?",
        r"personal\s*(?:information|info|details|data)?",
        r"profile",
    ],
    "summary": [
        r"(?:professional\s+)?summary",
        r"(?:career\s+)?objective",
        r"about\s*(?:me)?",
        r"overview",
        r"professional\s+profile",
        r"executive\s+summary",
    ],
    "experience": [
        r"(?:work\s+|professional\s+|employment\s+)?experience",
        r"work\s*history",
        r"employment\s*(?:history|record)?",
        r"career\s*(?:history|experience)?",
        r"professional\s*(?:background|history)",
        r"positions?\s*(?:held)?",
    ],
    "education": [
        r"education(?:al\s+background)?",
        r"academic\s*(?:background|history|qualifications?)?",
        r"qualifications?",
        r"degrees?",
        r"schooling",
    ],
    "skills": [
        r"(?:technical\s+|core\s+|key\s+|professional\s+)?skills?",
        r"competenc(?:ies|y)",
        r"expertise",
        r"technologies",
        r"technical\s*(?:proficiencies|expertise|stack)",
        r"tools?\s*(?:and\s*technologies)?",
        r"programming\s*(?:languages?|skills?)?",
    ],
    "certifications": [
        r"certifications?",
        r"certificates?",
        r"licenses?\s*(?:and\s*certifications?)?",
        r"credentials?",
        r"accreditations?",
    ],
    "projects": [
        r"projects?",
        r"personal\s*projects?",
        r"side\s*projects?",
        r"notable\s*projects?",
        r"portfolio",
    ],
    "awards": [
        r"awards?\s*(?:and\s*(?:honors?|achievements?))?",
        r"honors?",
        r"achievements?",
        r"accomplishments?",
        r"recognition",
    ],
    "publications": [
        r"publications?",
        r"research",
        r"papers?",
        r"articles?",
    ],
    "languages": [
        r"languages?",
        r"language\s*skills?",
        r"spoken\s*languages?",
    ],
    "interests": [
        r"interests?",
        r"hobbies",
        r"activities",
        r"volunteer(?:ing)?",
        r"extracurricular",
    ],
    "references": [
        r"references?",
        r"referees?",
    ],
}

# Pre-compile patterns for performance
_COMPILED_SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    key: [re.compile(r"^\s*" + pat + r"\s*:?\s*$", re.IGNORECASE | re.MULTILINE) for pat in patterns]
    for key, patterns in SECTION_HEADER_PATTERNS.items()
}

# Regex for a generic "looks like a section header" line
_GENERIC_HEADER_RE = re.compile(
    r"^(?:[A-Z][A-Z\s&/\-]{2,40}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\s*:?\s*$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Contact extraction patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:\+?1[\s\-.]?)?" r"(?:\(?\d{3}\)?[\s\-.]?)" r"\d{3}[\s\-.]?\d{4}" r"(?:\s*(?:ext|x|ext\.)\s*\d{1,5})?",
    re.IGNORECASE,
)
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Experience / Education date patterns
# ---------------------------------------------------------------------------

_DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+)?"
    r"\d{4}"
    r"(?:\s*[-–—]\s*"
    r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+)?"
    r"(?:\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Tt]oday))?",
    re.IGNORECASE,
)

_DEGREE_KEYWORDS = re.compile(
    r"\b(?:bachelor|master|phd|ph\.d|doctorate|associate|b\.?s\.?|m\.?s\.?|"
    r"b\.?a\.?|m\.?a\.?|m\.?b\.?a\.?|b\.?e\.?|m\.?e\.?|b\.?tech|m\.?tech|"
    r"b\.?sc|m\.?sc|diploma|certificate|degree)\b",
    re.IGNORECASE,
)

_GPA_RE = re.compile(r"\bgpa\s*:?\s*(\d+\.\d+)", re.IGNORECASE)


class SectionParser:
    """
    Identifies and extracts resume sections using spaCy NLP and regex patterns.

    Sections detected: contact, summary, experience, education, skills,
    certifications, projects, awards, publications, languages, interests, references.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialise the section parser.

        Args:
            spacy_model: Name of the spaCy model to load.
        """
        self._nlp = None
        self._spacy_model = spacy_model
        self._load_spacy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Split resume text into labelled sections.

        Args:
            text: Raw resume text.

        Returns:
            Dict mapping section key (e.g. 'experience') to section content.
            A 'header' key holds any text before the first recognised section.
        """
        if not text or not text.strip():
            return {}

        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Find all section boundary positions
        boundaries = self._find_section_boundaries(text)

        if not boundaries:
            # No sections found – return everything as raw text
            return {"raw": text.strip()}

        sections: Dict[str, str] = {}

        # Text before the first section header → 'header'
        first_start = boundaries[0][1]
        header_text = text[:first_start].strip()
        if header_text:
            sections["header"] = header_text

        # Slice content between consecutive boundaries
        for idx, (section_key, start, header_end) in enumerate(boundaries):
            end = boundaries[idx + 1][1] if idx + 1 < len(boundaries) else len(text)
            content = text[header_end:end].strip()
            # Merge duplicate sections (e.g. two 'skills' blocks)
            if section_key in sections:
                sections[section_key] = sections[section_key] + "\n" + content
            else:
                sections[section_key] = content

        return sections

    def extract_contact_info(self, text: str):
        """
        Extract contact information from text.

        Args:
            text: Text containing contact information (header or contact section).

        Returns:
            ContactInfo dataclass instance.
        """
        from ..models.resume import ContactInfo  # local import to avoid circular

        name = self._extract_name(text)
        email = self._extract_first(text, _EMAIL_RE)
        phone = self._extract_first(text, _PHONE_RE)
        linkedin = self._extract_first(text, _LINKEDIN_RE)
        github = self._extract_first(text, _GITHUB_RE)
        location = self._extract_location(text)

        return ContactInfo(
            name=name or "Unknown",
            email=email,
            phone=phone,
            location=location,
            linkedin=linkedin,
            github=github,
        )

    def extract_experience(self, text: str):
        """
        Parse work experience entries from experience section text.

        Args:
            text: Content of the experience section.

        Returns:
            List of Experience dataclass instances.
        """
        from ..models.resume import Experience  # local import  # noqa: F401

        if not text or not text.strip():
            return []

        entries = self._split_experience_entries(text)
        experiences = []

        for entry in entries:
            exp = self._parse_experience_entry(entry)
            if exp:
                experiences.append(exp)

        return experiences

    def extract_education(self, text: str):
        """
        Parse education entries from education section text.

        Args:
            text: Content of the education section.

        Returns:
            List of Education dataclass instances.
        """
        from ..models.resume import Education  # local import  # noqa: F401

        if not text or not text.strip():
            return []

        entries = self._split_education_entries(text)
        educations = []

        for entry in entries:
            edu = self._parse_education_entry(entry)
            if edu:
                educations.append(edu)

        return educations

    # ------------------------------------------------------------------
    # Section boundary detection
    # ------------------------------------------------------------------

    def _find_section_boundaries(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Return list of (section_key, line_start_pos, content_start_pos) tuples
        sorted by position in the text.
        """
        found: List[Tuple[str, int, int]] = []
        seen_positions: set = set()

        lines = text.split("\n")
        pos = 0

        for line in lines:
            line_end = pos + len(line) + 1  # +1 for the newline
            stripped = line.strip()

            if stripped:
                section_key = self._classify_header_line(stripped)
                if section_key and pos not in seen_positions:
                    seen_positions.add(pos)
                    content_start = line_end  # content starts after this line
                    found.append((section_key, pos, content_start))

            pos = line_end

        return found

    def _classify_header_line(self, line: str) -> Optional[str]:
        """
        Return the section key if *line* looks like a section header, else None.
        """
        # Must be reasonably short to be a header
        if len(line) > 80:
            return None

        # Strip trailing colon / punctuation for matching
        clean = re.sub(r"[:\-–—•*#]+$", "", line).strip()

        for section_key, patterns in _COMPILED_SECTION_PATTERNS.items():
            for pat in patterns:
                if pat.match(clean):
                    return section_key

        # Fallback: ALL-CAPS short line that looks like a header
        if clean.isupper() and 3 <= len(clean) <= 50:
            # Try to map to a known section
            lower = clean.lower()
            for section_key, patterns in SECTION_HEADER_PATTERNS.items():
                for pat in patterns:
                    if re.fullmatch(pat, lower, re.IGNORECASE):
                        return section_key
            # Unknown all-caps header – use lowercased version as key
            return clean.lower().replace(" ", "_")

        return None

    # ------------------------------------------------------------------
    # Name extraction (uses spaCy PERSON NER)
    # ------------------------------------------------------------------

    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from the top of the resume."""
        # Only look at the first ~500 characters (header area)
        snippet = text[:500]

        # Try spaCy NER first
        if self._nlp is not None:
            try:
                doc = self._nlp(snippet)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        name = ent.text.strip()
                        if self._looks_like_name(name):
                            return name
            except Exception as exc:
                logger.debug("spaCy NER failed for name extraction: %s", exc)

        # Fallback: first non-empty line that looks like a name
        for line in snippet.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip lines that contain obvious non-name content
            if any(marker in line.lower() for marker in ("@", "http", "phone", "email", "address", "linkedin")):
                continue
            if _EMAIL_RE.search(line) or _PHONE_RE.search(line):
                continue
            # A name is typically 2-4 words, each capitalised
            words = line.split()
            if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
                return line
        return None

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        words = text.split()
        if not (2 <= len(words) <= 5):
            return False
        return all(w[0].isupper() for w in words if w.isalpha())

    # ------------------------------------------------------------------
    # Location extraction
    # ------------------------------------------------------------------

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location/address from contact text using spaCy GPE entities."""
        snippet = text[:600]

        if self._nlp is not None:
            try:
                doc = self._nlp(snippet)
                gpe_entities = [ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
                if gpe_entities:
                    return ", ".join(gpe_entities[:2])
            except Exception as exc:
                logger.debug("spaCy NER failed for location extraction: %s", exc)

        # Fallback: look for "City, State" or "City, Country" pattern
        loc_re = re.compile(r"\b([A-Z][a-zA-Z\s]+),\s*([A-Z]{2}|[A-Z][a-zA-Z]+)\b")
        match = loc_re.search(snippet)
        if match:
            return match.group(0)
        return None

    # ------------------------------------------------------------------
    # Experience parsing helpers
    # ------------------------------------------------------------------

    def _split_experience_entries(self, text: str) -> List[str]:
        """
        Split experience section text into individual job entries.
        Heuristic: a new entry starts when we see a blank line separator.
        If no blank lines exist, treat the whole block as one entry.
        """
        # Split on blank lines first (most common resume format)
        blocks = re.split(r"\n{2,}", text.strip())
        if len(blocks) >= 2:
            return [b.strip() for b in blocks if b.strip()]

        # Single block – return as-is so date extraction works on the full text
        return [text.strip()]

    def _parse_experience_entry(self, entry: str):
        """Parse a single experience entry block into an Experience object."""
        from ..models.resume import Experience

        if not entry.strip():
            return None

        lines = [line.strip() for line in entry.splitlines() if line.strip()]
        if not lines:
            return None

        title = ""
        company = ""
        start_date = None
        end_date = None
        is_current = False
        description_lines = []

        # Extract date range from the full entry text
        date_match = _DATE_RANGE_RE.search(entry)
        if date_match:
            date_str = date_match.group(0)
            parts = re.split(r"\s*[-–—]\s*", date_str)
            start_date = parts[0].strip() if parts else None
            if len(parts) > 1:
                end_raw = parts[1].strip()
                if re.match(r"present|current|now|today", end_raw, re.IGNORECASE):
                    is_current = True
                    end_date = "Present"
                else:
                    end_date = end_raw

        # Identify which lines are "date lines" so we can skip them
        def _is_date_line(line: str) -> bool:
            cleaned = _DATE_RANGE_RE.sub("", line).strip()
            return cleaned == "" and bool(_DATE_RANGE_RE.search(line))

        non_date_lines = [line for line in lines if not _is_date_line(line)]

        # First non-date line is usually "Title at Company" or just "Title"
        if not non_date_lines:
            return None

        first_line = non_date_lines[0]
        # Remove any embedded date from the first line
        first_line_clean = _DATE_RANGE_RE.sub("", first_line).strip()

        at_split = re.split(r"\s+at\s+|\s*[|,]\s*", first_line_clean, maxsplit=1)
        if len(at_split) == 2:
            title = at_split[0].strip()
            company = at_split[1].strip()
            description_lines = non_date_lines[1:]
        else:
            title = first_line_clean
            if len(non_date_lines) > 1:
                second_line = _DATE_RANGE_RE.sub("", non_date_lines[1]).strip()
                # If second line doesn't look like a bullet point, treat as company
                if second_line and not second_line.startswith(("-", "•", "*", "·")):
                    company = second_line
                    description_lines = non_date_lines[2:]
                else:
                    description_lines = non_date_lines[1:]
            else:
                description_lines = []

        description = " ".join(description_lines).strip() if description_lines else None

        if not title:
            return None

        return Experience(
            title=title,
            company=company or "Unknown",
            start_date=start_date,
            end_date=end_date,
            description=description,
            is_current=is_current,
        )

    # ------------------------------------------------------------------
    # Education parsing helpers
    # ------------------------------------------------------------------

    def _split_education_entries(self, text: str) -> List[str]:
        """Split education section into individual entries."""
        blocks = re.split(r"\n{2,}", text.strip())
        if len(blocks) >= 1:
            return [b.strip() for b in blocks if b.strip()]
        return [text.strip()]

    def _parse_education_entry(self, entry: str):
        """Parse a single education entry block into an Education object."""
        from ..models.resume import Education

        if not entry.strip():
            return None

        lines = [line.strip() for line in entry.splitlines() if line.strip()]
        if not lines:
            return None

        degree = ""
        institution = ""
        graduation_date = None
        gpa = None
        major = None

        # Extract GPA
        gpa_match = _GPA_RE.search(entry)
        if gpa_match:
            try:
                gpa = float(gpa_match.group(1))
            except ValueError:
                pass

        # Extract graduation date
        date_match = _DATE_RANGE_RE.search(entry)
        if date_match:
            graduation_date = date_match.group(0).strip()

        # Find degree line
        for line in lines:
            if _DEGREE_KEYWORDS.search(line):
                degree_line = _DATE_RANGE_RE.sub("", line).strip()
                # Try to split "Degree in Major"
                in_split = re.split(r"\s+in\s+", degree_line, maxsplit=1, flags=re.IGNORECASE)
                if len(in_split) == 2:
                    degree = in_split[0].strip()
                    major = in_split[1].strip()
                else:
                    degree = degree_line
                break

        # Find institution (line that doesn't contain degree keywords and isn't a date)
        for line in lines:
            clean = _DATE_RANGE_RE.sub("", line).strip()
            if clean and not _DEGREE_KEYWORDS.search(clean) and not _GPA_RE.search(clean) and clean != degree:
                institution = clean
                break

        if not degree and lines:
            degree = _DATE_RANGE_RE.sub("", lines[0]).strip()

        if not degree:
            return None

        return Education(
            degree=degree,
            institution=institution or "Unknown",
            graduation_date=graduation_date,
            gpa=gpa,
            major=major,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first(text: str, pattern: re.Pattern) -> Optional[str]:
        """Return the first match of *pattern* in *text*, or None."""
        match = pattern.search(text)
        return match.group(0).strip() if match else None

    def _load_spacy(self) -> None:
        """Load the spaCy model, falling back gracefully if unavailable."""
        try:
            import spacy

            self._nlp = spacy.load(self._spacy_model)
            logger.debug("Loaded spaCy model: %s", self._spacy_model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Falling back to regex-only parsing.",
                self._spacy_model,
            )
            self._nlp = None
        except ImportError:
            logger.warning("spaCy not installed. Falling back to regex-only parsing.")
            self._nlp = None
