"""Text extraction from PDF and DOCX files."""

import logging
from pathlib import Path
from typing import Optional
import traceback

logger = logging.getLogger(__name__)


class TextExtractor:
    """Handles text extraction from various document formats."""
    
    def __init__(self):
        """Initialize text extractor."""
        self._pdf_extractors = []
        self._docx_extractors = []
        
        # Try to import PDF libraries
        try:
            import fitz  # PyMuPDF
            self._pdf_extractors.append(self._extract_with_pymupdf)
            logger.debug("PyMuPDF available for PDF extraction")
        except ImportError:
            logger.warning("PyMuPDF not available")
        
        try:
            import pdfplumber
            self._pdf_extractors.append(self._extract_with_pdfplumber)
            logger.debug("pdfplumber available for PDF extraction")
        except ImportError:
            logger.warning("pdfplumber not available")
        
        # Try to import DOCX library
        try:
            import docx
            self._docx_extractors.append(self._extract_with_python_docx)
            logger.debug("python-docx available for DOCX extraction")
        except ImportError:
            logger.warning("python-docx not available")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension == '.docx':
                return self._extract_docx_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return f"Error extracting text: {str(e)}"
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using available libraries."""
        if not self._pdf_extractors:
            raise ValueError("No PDF extraction libraries available. Install PyMuPDF or pdfplumber.")
        
        # Try each PDF extractor until one works
        for extractor in self._pdf_extractors:
            try:
                text = extractor(file_path)
                if text and text.strip():
                    return text
            except Exception as e:
                logger.debug(f"PDF extractor failed: {str(e)}")
                continue
        
        # If all extractors fail, return error message
        return "Failed to extract text from PDF file"
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX using available libraries."""
        if not self._docx_extractors:
            raise ValueError("No DOCX extraction libraries available. Install python-docx.")
        
        # Try each DOCX extractor until one works
        for extractor in self._docx_extractors:
            try:
                text = extractor(file_path)
                if text and text.strip():
                    return text
            except Exception as e:
                logger.debug(f"DOCX extractor failed: {str(e)}")
                continue
        
        # If all extractors fail, return error message
        return "Failed to extract text from DOCX file"
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fitz)."""
        import fitz
        
        text_content = []
        
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_content.append(text)
        
        return "\\n".join(text_content)
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                
                if text and text.strip():
                    text_content.append(text)
        
        return "\\n".join(text_content)
    
    def _extract_with_python_docx(self, file_path: str) -> str:
        """Extract text using python-docx."""
        import docx
        
        doc = docx.Document(file_path)
        text_content = []
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        return "\\n".join(text_content)
    
    def get_document_info(self, file_path: str) -> dict:
        """
        Get metadata information about the document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document metadata
        """
        path = Path(file_path)
        
        info = {
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'file_extension': path.suffix.lower(),
            'extractors_available': {
                'pdf': len(self._pdf_extractors),
                'docx': len(self._docx_extractors)
            }
        }
        
        # Try to get additional metadata for PDFs
        if path.suffix.lower() == '.pdf' and self._pdf_extractors:
            try:
                import fitz
                with fitz.open(file_path) as doc:
                    info.update({
                        'page_count': len(doc),
                        'title': doc.metadata.get('title', ''),
                        'author': doc.metadata.get('author', ''),
                        'creator': doc.metadata.get('creator', '')
                    })
            except Exception as e:
                logger.debug(f"Could not extract PDF metadata: {str(e)}")
        
        return info