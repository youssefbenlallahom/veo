
# utils/pdf_validator.py
import pypdf
from typing import Tuple


class PDFValidator:
    """Validates PDF files for resume analysis."""
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """Validate that the PDF is readable."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                if num_pages == 0:
                    return False, "PDF has no pages"
                
                # Try to read first page text
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                
                return True, f"PDF is valid with {num_pages} pages"
        except Exception as e:
            return False, f"PDF validation failed: {str(e)}"