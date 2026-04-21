"""Section parsing and information extraction from resume text."""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..models.resume import ContactInfo, Experience, Education

logger = logging.getLogger(__name__)


class SectionParser:
    """Parses resume text into structured sections and extracts information."""
    
    def __init__(self):
        """Initialize section parser with regex patterns."""
        # Common section headers
        self.section_patterns = {
            'contact': r'(?i)(contact|personal|info)',
            'summary': r'(?i)(summary|profile|objective|about)',
            'experience': r'(?i)(experience|employment|work|career|professional)',
            'education': r'(?i)(education|academic|qualification|degree)',
            'skills': r'(?i)(skills|technical|competenc|expertise|proficienc)',
            'projects': r'(?i)(projects|portfolio)',
            'certifications': r'(?i)(certification|certificate|license)',
            'awards': r'(?i)(awards|achievement|honor|recognition)'
        }
        
        # Email pattern
        self.email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        
        # Phone pattern (various formats)
        self.phone_pattern = r'(?:\\+?1[-\\s]?)?\\(?[0-9]{3}\\)?[-\\s]?[0-9]{3}[-\\s]?[0-9]{4}'
        
        # LinkedIn pattern
        self.linkedin_pattern = r'(?i)(?:linkedin\\.com/in/|linkedin\\.com/pub/)([A-Za-z0-9-]+)'
        
        # GitHub pattern
        self.github_pattern = r'(?i)(?:github\\.com/)([A-Za-z0-9-]+)'
        
        # Date patterns
        self.date_patterns = [
            r'\\b(\\d{1,2})/(\\d{1,2})/(\\d{4})\\b',  # MM/DD/YYYY
            r'\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})\\b',  # YYYY-MM-DD
            r'\\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+(\\d{4})\\b',  # Month YYYY
            r'\\b(\\d{4})\\b'  # Just year
        ]
    
    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parse resume text into sections.
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        lines = text.split('\\n')
        current_section = 'header'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            section_found = None
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, line) and len(line) < 50:  # Section headers are usually short
                    section_found = section_name
                    break
            
            if section_found:
                # Save previous section
                if current_content:
                    sections[current_section] = '\\n'.join(current_content)
                
                # Start new section
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\\n'.join(current_content)
        
        return sections
    
    def extract_contact_info(self, text: str) -> ContactInfo:
        """
        Extract contact information from text.
        
        Args:
            text: Text containing contact information
            
        Returns:
            ContactInfo object
        """
        # Extract email
        email_match = re.search(self.email_pattern, text)
        email = email_match.group(0) if email_match else None
        
        # Extract phone
        phone_match = re.search(self.phone_pattern, text)
        phone = phone_match.group(0) if phone_match else None
        
        # Extract LinkedIn
        linkedin_match = re.search(self.linkedin_pattern, text)
        linkedin = f"linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None
        
        # Extract GitHub
        github_match = re.search(self.github_pattern, text)
        github = f"github.com/{github_match.group(1)}" if github_match else None
        
        # Extract name (heuristic: first line that looks like a name)
        name = self._extract_name(text)
        
        # Extract location (look for city, state patterns)
        location = self._extract_location(text)
        
        return ContactInfo(
            name=name,
            email=email,
            phone=phone,
            location=location,
            linkedin=linkedin,
            github=github
        )
    
    def extract_experience(self, text: str) -> List[Experience]:
        """
        Extract work experience from text.
        
        Args:
            text: Text containing work experience
            
        Returns:
            List of Experience objects
        """
        experiences = []
        
        # Split by common separators
        experience_blocks = re.split(r'\\n\\s*\\n|\\n(?=[A-Z][^\\n]*(?:at|@|\\|))', text)
        
        for block in experience_blocks:
            block = block.strip()
            if len(block) < 20:  # Skip very short blocks
                continue
            
            experience = self._parse_experience_block(block)
            if experience:
                experiences.append(experience)
        
        return experiences
    
    def extract_education(self, text: str) -> List[Education]:
        """
        Extract education information from text.
        
        Args:
            text: Text containing education information
            
        Returns:
            List of Education objects
        """
        educations = []
        
        # Split by common separators
        education_blocks = re.split(r'\\n\\s*\\n|\\n(?=[A-Z][^\\n]*(?:University|College|Institute|School))', text)
        
        for block in education_blocks:
            block = block.strip()
            if len(block) < 10:  # Skip very short blocks
                continue
            
            education = self._parse_education_block(block)
            if education:
                educations.append(education)
        
        return educations
    
    def _extract_name(self, text: str) -> str:
        """Extract name from contact text."""
        lines = [line.strip() for line in text.split('\\n') if line.strip()]
        
        for line in lines[:5]:  # Check first 5 lines
            # Skip lines that look like email, phone, or addresses
            if re.search(self.email_pattern, line) or re.search(self.phone_pattern, line):
                continue
            
            # Look for lines that could be names (2-4 words, proper case)
            words = line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word.isalpha()):
                return line
        
        return "Unknown"
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from contact text."""
        # Look for city, state patterns
        location_pattern = r'\\b([A-Z][a-z]+),\\s*([A-Z]{2})\\b'
        match = re.search(location_pattern, text)
        
        if match:
            return f"{match.group(1)}, {match.group(2)}"
        
        # Look for just city names
        city_pattern = r'\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?)\\b'
        cities = re.findall(city_pattern, text)
        
        # Filter out common non-city words
        non_cities = {'Email', 'Phone', 'Address', 'Contact', 'LinkedIn', 'GitHub'}
        cities = [city for city in cities if city not in non_cities]
        
        return cities[0] if cities else None
    
    def _parse_experience_block(self, block: str) -> Optional[Experience]:
        """Parse a single experience block."""
        lines = [line.strip() for line in block.split('\\n') if line.strip()]
        
        if not lines:
            return None
        
        # First line usually contains title and company
        first_line = lines[0]
        
        # Try to extract title and company
        title, company = self._parse_title_company(first_line)
        
        if not title and not company:
            return None
        
        # Look for dates
        start_date, end_date, is_current = self._extract_dates(block)
        
        # Combine remaining lines as description
        description_lines = lines[1:] if len(lines) > 1 else []
        description = '\\n'.join(description_lines) if description_lines else None
        
        return Experience(
            title=title or "Unknown Position",
            company=company or "Unknown Company",
            start_date=start_date,
            end_date=end_date,
            description=description,
            is_current=is_current
        )
    
    def _parse_education_block(self, block: str) -> Optional[Education]:
        """Parse a single education block."""
        lines = [line.strip() for line in block.split('\\n') if line.strip()]
        
        if not lines:
            return None
        
        # Look for degree and institution
        degree, institution = self._parse_degree_institution(block)
        
        if not degree and not institution:
            return None
        
        # Look for graduation date
        graduation_date = self._extract_graduation_date(block)
        
        # Look for GPA
        gpa = self._extract_gpa(block)
        
        # Look for major
        major = self._extract_major(block)
        
        return Education(
            degree=degree or "Unknown Degree",
            institution=institution or "Unknown Institution",
            graduation_date=graduation_date,
            gpa=gpa,
            major=major
        )
    
    def _parse_title_company(self, line: str) -> tuple:
        """Parse job title and company from a line."""
        # Common patterns: "Title at Company", "Title | Company", "Title - Company"
        patterns = [
            r'^(.+?)\\s+at\\s+(.+)$',
            r'^(.+?)\\s*\\|\\s*(.+)$',
            r'^(.+?)\\s*-\\s*(.+)$',
            r'^(.+?)\\s*@\\s*(.+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        # If no pattern matches, assume the whole line is the title
        return line.strip(), None
    
    def _parse_degree_institution(self, text: str) -> tuple:
        """Parse degree and institution from education text."""
        # Look for degree patterns
        degree_patterns = [
            r'(Bachelor|Master|PhD|Ph\\.D|MBA|BS|BA|MS|MA|B\\.S|B\\.A|M\\.S|M\\.A)\\s*(?:of|in|degree)?\\s*([^\\n]*)',
            r'([^\\n]*(?:Bachelor|Master|PhD|Ph\\.D|MBA|BS|BA|MS|MA|B\\.S|B\\.A|M\\.S|M\\.A)[^\\n]*)'
        ]
        
        degree = None
        for pattern in degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                degree = match.group(0).strip()
                break
        
        # Look for institution patterns
        institution_patterns = [
            r'(University|College|Institute|School)\\s+of\\s+([^\\n]+)',
            r'([^\\n]*(?:University|College|Institute|School)[^\\n]*)'
        ]
        
        institution = None
        for pattern in institution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                institution = match.group(0).strip()
                break
        
        return degree, institution
    
    def _extract_dates(self, text: str) -> tuple:
        """Extract start date, end date, and current status."""
        # Look for date ranges
        date_range_patterns = [
            r'(\\d{4})\\s*-\\s*(\\d{4}|present|current)',
            r'(\\w+\\s+\\d{4})\\s*-\\s*(\\w+\\s+\\d{4}|present|current)',
            r'(\\d{1,2}/\\d{4})\\s*-\\s*(\\d{1,2}/\\d{4}|present|current)'
        ]
        
        for pattern in date_range_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_date = match.group(1)
                end_date = match.group(2)
                is_current = end_date.lower() in ['present', 'current']
                return start_date, None if is_current else end_date, is_current
        
        return None, None, False
    
    def _extract_graduation_date(self, text: str) -> Optional[str]:
        """Extract graduation date from education text."""
        # Look for years that could be graduation dates
        year_matches = re.findall(r'\\b(19|20)\\d{2}\\b', text)
        
        if year_matches:
            # Return the most recent year
            years = [int(year) for year in year_matches]
            return str(max(years))
        
        return None
    
    def _extract_gpa(self, text: str) -> Optional[float]:
        """Extract GPA from education text."""
        gpa_pattern = r'GPA:?\\s*(\\d+\\.\\d+)'
        match = re.search(gpa_pattern, text, re.IGNORECASE)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_major(self, text: str) -> Optional[str]:
        """Extract major from education text."""
        major_patterns = [
            r'(?:major|concentration|specialization)\\s*:?\\s*([^\\n]+)',
            r'(?:in|of)\\s+([^\\n]*(?:Science|Engineering|Arts|Business|Studies)[^\\n]*)'
        ]
        
        for pattern in major_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None