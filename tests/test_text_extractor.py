"""Tests for text extraction functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.parsers.text_extractor import TextExtractor


class TestTextExtractor:
    """Test cases for TextExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor()

    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        assert self.extractor is not None
        # Should have at least one PDF extractor available
        assert len(self.extractor._pdf_extractors) > 0
        # Should have DOCX extractor available
        assert len(self.extractor._docx_extractors) > 0

    def test_extract_nonexistent_file(self):
        """Test extracting from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_text("nonexistent_file.pdf")

    def test_extract_unsupported_format(self):
        """Test extracting from unsupported file format returns error message."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Test content")
            tmp_path = tmp.name

        try:
            result = self.extractor.extract_text(tmp_path)
            # Should return error message, not raise exception
            assert "Error extracting text" in result
        finally:
            os.unlink(tmp_path)

    @patch("fitz.open")
    def test_pymupdf_extraction_with_ocr_warning(self, mock_fitz_open):
        """Test PyMuPDF extraction with OCR warning for scanned PDFs."""
        # Mock a scanned PDF (lots of images, little text)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "A"  # Very little text
        mock_page.get_images.return_value = [1, 2, 3, 4, 5]  # Many images
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page  # Fix iteration
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor._extract_with_pymupdf(tmp_path)

            # Should contain OCR warning
            assert "OCR WARNING" in result
            assert "scanned PDF" in result
            assert "A" in result  # Original text should still be there

        finally:
            os.unlink(tmp_path)

    @patch("fitz.open")
    def test_pymupdf_extraction_normal_pdf(self, mock_fitz_open):
        """Test PyMuPDF extraction with normal PDF (no OCR warning)."""
        # Mock a normal PDF (good text, few images)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "This is a normal PDF with plenty of text content that should not trigger OCR warnings."
        )
        mock_page.get_images.return_value = []  # No images
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page  # Fix iteration
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor._extract_with_pymupdf(tmp_path)

            # Should NOT contain OCR warning
            assert "OCR WARNING" not in result
            assert "This is a normal PDF" in result

        finally:
            os.unlink(tmp_path)

    @patch("pdfplumber.open")
    def test_pdfplumber_extraction_with_tables(self, mock_pdfplumber_open):
        """Test pdfplumber extraction with table fallback."""
        # Mock pdfplumber with tables
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Short"  # Little text
        mock_page.extract_tables.return_value = [[["Name", "Position"], ["John Doe", "Engineer"], ["Jane Smith", "Manager"]]]
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber_open.return_value = mock_pdf

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor._extract_with_pdfplumber(tmp_path)

            # Should contain table data
            assert "Name | Position" in result
            assert "John Doe | Engineer" in result
            assert "Jane Smith | Manager" in result

        finally:
            os.unlink(tmp_path)

    @patch("pdfplumber.open")
    def test_pdfplumber_low_text_ocr_warning(self, mock_pdfplumber_open):
        """Test pdfplumber OCR warning for low text extraction."""
        # Mock pdfplumber with very little text
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A"  # Very little text
        mock_page.extract_tables.return_value = []  # No tables
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber_open.return_value = mock_pdf

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor._extract_with_pdfplumber(tmp_path)

            # Should contain OCR warning
            assert "OCR WARNING" in result
            assert "scanned document" in result

        finally:
            os.unlink(tmp_path)

    @patch("fitz.open")
    def test_is_scanned_pdf_detection(self, mock_fitz_open):
        """Test scanned PDF detection logic."""
        # Mock a scanned PDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "A"  # Very little text
        mock_page.get_images.return_value = [1, 2, 3]  # Some images
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page  # Fix iteration
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor.is_scanned_pdf(tmp_path)
            assert result is True

        finally:
            os.unlink(tmp_path)

    @patch("fitz.open")
    def test_is_not_scanned_pdf_detection(self, mock_fitz_open):
        """Test normal PDF detection (not scanned)."""
        # Mock a normal PDF with sufficient text
        mock_doc = MagicMock()
        mock_page = MagicMock()
        # Use longer text to ensure it doesn't trigger scanned detection
        mock_page.get_text.return_value = "This is a normal PDF with plenty of text content. " * 10  # Lots of text
        mock_page.get_images.return_value = []  # No images
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page  # Fix iteration
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor.is_scanned_pdf(tmp_path)
            assert result is False

        finally:
            os.unlink(tmp_path)

    def test_is_scanned_pdf_non_pdf_file(self):
        """Test scanned PDF detection on non-PDF file."""
        # Create temporary DOCX file
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = self.extractor.is_scanned_pdf(tmp_path)
            assert result is False

        finally:
            os.unlink(tmp_path)

    @patch("fitz.open")
    def test_get_document_info_with_scanned_detection(self, mock_fitz_open):
        """Test document info includes scanned PDF detection."""
        # Mock PDF metadata
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 5
        mock_doc.metadata = {"title": "Test Document", "author": "Test Author", "creator": "Test Creator"}
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_fitz_open.return_value = mock_doc

        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock the is_scanned_pdf method
            with patch.object(self.extractor, "is_scanned_pdf", return_value=True):
                info = self.extractor.get_document_info(tmp_path)

            assert info["file_extension"] == ".pdf"
            assert info["page_count"] == 5
            assert info["title"] == "Test Document"
            assert info["author"] == "Test Author"
            assert info["creator"] == "Test Creator"
            assert info["is_scanned"] is True

        finally:
            os.unlink(tmp_path)

    def test_pdf_extraction_fallback_mechanism(self):
        """Test that PDF extraction tries PyMuPDF first, then pdfplumber."""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a test extractor with controlled behavior
            test_extractor = TextExtractor()

            # Replace the extractors list with our mocked functions
            def mock_pymupdf_fail(file_path):
                raise Exception("PyMuPDF failed")

            def mock_pdfplumber_success(file_path):
                return "Extracted with pdfplumber"

            # Replace the extractor methods
            test_extractor._pdf_extractors = [mock_pymupdf_fail, mock_pdfplumber_success]

            result = test_extractor._extract_pdf_text(tmp_path)
            assert result == "Extracted with pdfplumber"

        finally:
            os.unlink(tmp_path)

    def test_pdf_extraction_all_extractors_fail(self):
        """Test PDF extraction when all extractors fail."""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock all extractors to fail
            with patch.object(self.extractor, "_extract_with_pymupdf", side_effect=Exception("PyMuPDF failed")):
                with patch.object(self.extractor, "_extract_with_pdfplumber", side_effect=Exception("pdfplumber failed")):
                    result = self.extractor._extract_pdf_text(tmp_path)
                    assert "Failed to extract text from PDF file" in result

        finally:
            os.unlink(tmp_path)


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return "John Smith\\nSoftware Engineer\\nEmail: john@example.com\\nSkills: Python, JavaScript"


def test_text_extraction_integration():
    """Integration test for text extraction."""
    extractor = TextExtractor()

    # Test that extractor has the required methods
    assert hasattr(extractor, "extract_text")
    assert hasattr(extractor, "is_scanned_pdf")
    assert hasattr(extractor, "get_document_info")

    # Test that PDF extractors are available
    assert len(extractor._pdf_extractors) > 0
    assert len(extractor._docx_extractors) > 0

    # DOCX-specific tests for enhanced functionality
    @patch("src.parsers.text_extractor.docx")
    def test_docx_extraction_with_headers_footers(self, mock_docx):
        """Test DOCX extraction with headers and footers."""
        # Mock document structure
        mock_doc = Mock()

        # Mock header/footer
        mock_header_paragraph = Mock()
        mock_header_paragraph.text = "Document Header"
        mock_header = Mock()
        mock_header.paragraphs = [mock_header_paragraph]

        mock_footer_paragraph = Mock()
        mock_footer_paragraph.text = "Document Footer"
        mock_footer = Mock()
        mock_footer.paragraphs = [mock_footer_paragraph]

        mock_section = Mock()
        mock_section.header = mock_header
        mock_section.footer = mock_footer
        mock_doc.sections = [mock_section]

        # Mock body content
        mock_paragraph = Mock()
        mock_paragraph.text = "Main content paragraph"
        mock_paragraph.style.name = "Normal"
        mock_run = Mock()
        mock_run.text = "Main content paragraph"
        mock_run.bold = False
        mock_run.italic = False
        mock_paragraph.runs = [mock_run]
        mock_paragraph._element = Mock()
        mock_doc.paragraphs = [mock_paragraph]

        # Mock document element structure
        mock_p_element = Mock()
        mock_p_element.tag = "p"
        mock_doc.element.body = [mock_p_element]

        mock_docx.Document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            result = self.extractor._extract_with_python_docx(tmp_file_path)

            assert "[HEADER] Document Header" in result
            assert "[FOOTER] Document Footer" in result
            assert "Main content paragraph" in result
        finally:
            os.unlink(tmp_file_path)

    @patch("src.parsers.text_extractor.docx")
    def test_docx_extraction_with_formatted_text(self, mock_docx):
        """Test DOCX extraction preserving bold/italic formatting."""
        mock_doc = Mock()
        mock_doc.sections = []  # No headers/footers

        # Mock paragraph with formatted runs
        mock_paragraph = Mock()
        mock_paragraph.text = "Bold text and italic text"
        mock_paragraph.style.name = "Normal"

        # Mock bold run
        mock_bold_run = Mock()
        mock_bold_run.text = "Bold text"
        mock_bold_run.bold = True
        mock_bold_run.italic = False

        # Mock italic run
        mock_italic_run = Mock()
        mock_italic_run.text = " and "
        mock_italic_run.bold = False
        mock_italic_run.italic = False

        mock_italic_run2 = Mock()
        mock_italic_run2.text = "italic text"
        mock_italic_run2.bold = False
        mock_italic_run2.italic = True

        mock_paragraph.runs = [mock_bold_run, mock_italic_run, mock_italic_run2]
        mock_paragraph._element = Mock()
        mock_doc.paragraphs = [mock_paragraph]

        # Mock document element structure
        mock_p_element = Mock()
        mock_p_element.tag = "p"
        mock_doc.element.body = [mock_p_element]

        mock_docx.Document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            result = self.extractor._extract_with_python_docx(tmp_file_path)

            assert "**Bold text**" in result
            assert "*italic text*" in result
        finally:
            os.unlink(tmp_file_path)

    @patch("src.parsers.text_extractor.docx")
    def test_docx_extraction_with_section_headers(self, mock_docx):
        """Test DOCX extraction with section header detection."""
        mock_doc = Mock()
        mock_doc.sections = []  # No headers/footers

        # Mock header paragraph
        mock_header_paragraph = Mock()
        mock_header_paragraph.text = "SECTION HEADER"
        mock_header_paragraph.style.name = "Heading 1"
        mock_header_run = Mock()
        mock_header_run.text = "SECTION HEADER"
        mock_header_run.bold = True
        mock_header_run.italic = False
        mock_header_paragraph.runs = [mock_header_run]
        mock_header_paragraph._element = Mock()

        # Mock normal paragraph
        mock_normal_paragraph = Mock()
        mock_normal_paragraph.text = "Regular content under header"
        mock_normal_paragraph.style.name = "Normal"
        mock_normal_run = Mock()
        mock_normal_run.text = "Regular content under header"
        mock_normal_run.bold = False
        mock_normal_run.italic = False
        mock_normal_paragraph.runs = [mock_normal_run]
        mock_normal_paragraph._element = Mock()

        mock_doc.paragraphs = [mock_header_paragraph, mock_normal_paragraph]

        # Mock document element structure
        mock_p_element1 = Mock()
        mock_p_element1.tag = "p"
        mock_p_element2 = Mock()
        mock_p_element2.tag = "p"
        mock_doc.element.body = [mock_p_element1, mock_p_element2]

        mock_docx.Document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            result = self.extractor._extract_with_python_docx(tmp_file_path)

            assert "## **SECTION HEADER**" in result
            assert "Regular content under header" in result
        finally:
            os.unlink(tmp_file_path)

    @patch("src.parsers.text_extractor.docx")
    def test_docx_extraction_with_structured_tables(self, mock_docx):
        """Test DOCX extraction with structured table formatting."""
        mock_doc = Mock()
        mock_doc.sections = []  # No headers/footers
        mock_doc.paragraphs = []  # No paragraphs

        # Mock table with header row
        mock_table = Mock()

        # Mock header row with bold formatting
        mock_header_row = Mock()
        mock_header_cell1 = Mock()
        mock_header_cell1.text = "Name"
        mock_header_cell2 = Mock()
        mock_header_cell2.text = "Age"
        mock_header_cell3 = Mock()
        mock_header_cell3.text = "City"

        # Mock bold formatting in header cells
        mock_header_paragraph = Mock()
        mock_bold_run = Mock()
        mock_bold_run.bold = True
        mock_header_paragraph.runs = [mock_bold_run]
        mock_header_cell1.paragraphs = [mock_header_paragraph]
        mock_header_cell2.paragraphs = [mock_header_paragraph]
        mock_header_cell3.paragraphs = [mock_header_paragraph]

        mock_header_row.cells = [mock_header_cell1, mock_header_cell2, mock_header_cell3]

        # Mock data row
        mock_data_row = Mock()
        mock_data_cell1 = Mock()
        mock_data_cell1.text = "John"
        mock_data_cell2 = Mock()
        mock_data_cell2.text = "25"
        mock_data_cell3 = Mock()
        mock_data_cell3.text = "NYC"

        # Mock normal formatting in data cells
        mock_normal_paragraph = Mock()
        mock_normal_run = Mock()
        mock_normal_run.bold = False
        mock_normal_paragraph.runs = [mock_normal_run]
        mock_data_cell1.paragraphs = [mock_normal_paragraph]
        mock_data_cell2.paragraphs = [mock_normal_paragraph]
        mock_data_cell3.paragraphs = [mock_normal_paragraph]

        mock_data_row.cells = [mock_data_cell1, mock_data_cell2, mock_data_cell3]

        mock_table.rows = [mock_header_row, mock_data_row]
        mock_table._element = Mock()
        mock_doc.tables = [mock_table]

        # Mock document element structure
        mock_tbl_element = Mock()
        mock_tbl_element.tag = "tbl"
        mock_doc.element.body = [mock_tbl_element]

        mock_docx.Document.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            result = self.extractor._extract_with_python_docx(tmp_file_path)

            assert "[TABLE]" in result
            assert "| Name | Age | City |" in result
            assert "| --- | --- | --- |" in result
            assert "| John | 25 | NYC |" in result
            assert "[/TABLE]" in result
        finally:
            os.unlink(tmp_file_path)

    def test_docx_extraction_fallback_mechanism(self):
        """Test that DOCX extraction handles missing libraries."""
        with patch.object(self.extractor, "_docx_extractors", []):
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name

            try:
                result = self.extractor._extract_docx_text(tmp_file_path)
                assert "No DOCX extraction libraries available" in result
            finally:
                os.unlink(tmp_file_path)
