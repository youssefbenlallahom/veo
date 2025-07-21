from crewai.tools import BaseTool
from typing import Type, Union, Any
from pydantic import BaseModel, Field, field_validator
import PyPDF2
import pdfplumber
import os
import re


class CustomPDFToolInput(BaseModel):
    """Input schema for CustomPDFTool."""
    query: str = Field(..., description="Search query to find specific information in the PDF, or use 'extract_all' to get the complete document content.")
    
    @field_validator('query', mode='before')
    @classmethod
    def validate_query(cls, v):
        """Convert various input types to string."""
        if isinstance(v, dict):
            # If it's a dict, try to extract the actual query value
            if 'description' in v:
                return v['description']
            elif 'query' in v:
                return v['query']
            elif 'value' in v:
                return v['value']
            else:
                # If dict has other structure, convert to string
                return str(v)
        elif isinstance(v, str):
            return v
        else:
            return str(v)


def sanitize_text(text):
    """Sanitize text to remove problematic characters."""
    if not text:
        return ""
    
    # First try with UTF-8 encoding/decoding with error handling
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove other problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', text)
    
    # As a last resort, keep only ASCII characters
    if '\ufffd' in text or any(ord(c) > 127 for c in text):
        text = ''.join(c for c in text if ord(c) < 128)
    
    return text.strip()


class CustomPDFTool(BaseTool):
    name: str = "PDF Document Reader"
    description: str = (
        "Extracts and searches text content from PDF documents. "
        "USAGE: Call with query parameter as a string. "
        "Examples: query='extract_all' for complete text, "
        "query='contact information' for contact details, "
        "query='work experience' for employment history, "
        "query='education' for educational background, "
        "query='skills' for technical skills."
    )
    args_schema: Type[BaseModel] = CustomPDFToolInput

    def __init__(self, pdf_path: str, **kwargs):
        super().__init__(**kwargs)
        # Store path using underscore to avoid Pydantic conflicts
        self._pdf_file_path = pdf_path
        
        # Validate PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not os.path.isfile(pdf_path):
            raise ValueError(f"Path is not a file: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

    def _run(self, query: str) -> str:
        """Extract text from PDF based on the query."""
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                query = str(query)
            
            print(f"[DEBUG] PDF tool received query: '{query}'")
            
            text_content = ""
            
            # Try pdfplumber first (better for text extraction and formatting)
            try:
                with pdfplumber.open(self._pdf_file_path) as pdf:
                    print(f"[DEBUG] Successfully opened PDF with {len(pdf.pages)} pages")
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            # Apply thorough text sanitization
                            cleaned_text = sanitize_text(page_text)
                            if cleaned_text:
                                text_content += f"\n--- Page {page_num} ---\n{cleaned_text}\n"
                            else:
                                print(f"[DEBUG] Page {page_num} text was empty after sanitization")
                        
                        # Also try to extract tables if they exist
                        try:
                            tables = page.extract_tables()
                            if tables:
                                for table_num, table in enumerate(tables, 1):
                                    text_content += f"\n--- Page {page_num} Table {table_num} ---\n"
                                    for row in table:
                                        if row:
                                            # Apply thorough sanitization to each cell
                                            clean_row = [sanitize_text(str(cell)) if cell else "" for cell in row]
                                            text_content += " | ".join(clean_row) + "\n"
                        except Exception as table_error:
                            print(f"[DEBUG] Error extracting tables on page {page_num}: {table_error}")
                        
            except Exception as pdfplumber_error:
                print(f"[DEBUG] PDFPlumber failed: {pdfplumber_error}, trying PyPDF2...")
                
                # Fallback to PyPDF2
                with open(self._pdf_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    print(f"[DEBUG] PyPDF2 opened PDF with {len(pdf_reader.pages)} pages")
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                # Apply thorough text sanitization
                                cleaned_text = sanitize_text(page_text)
                                if cleaned_text:
                                    text_content += f"\n--- Page {page_num} ---\n{cleaned_text}\n"
                                else:
                                    print(f"[DEBUG] PyPDF2 page {page_num} text was empty after sanitization")
                        except Exception as page_error:
                            print(f"[DEBUG] Error extracting page {page_num}: {page_error}")
                            continue
            
            print(f"[DEBUG] Total extracted text length: {len(text_content)} characters")
            
            if not text_content.strip():
                return "ERROR: Could not extract any readable text from the PDF. The document may be image-based, corrupted, or password-protected."
            
            # Process the query
            query_lower = query.lower().strip()
            
            if query_lower in ["extract_all", "all", "everything", "complete", "full", "entire", "whole"]:
                result = f"=== COMPLETE PDF DOCUMENT CONTENT ===\n\n{text_content}\n\n=== END OF DOCUMENT ==="
                print(f"[DEBUG] Returning complete content ({len(result)} characters)")
                return result
            else:
                # Ensure text_content is properly sanitized before splitting
                text_content = sanitize_text(text_content)
                
                # Search for specific content with improved matching
                lines = text_content.split('\n')
                relevant_lines = []
                query_terms = [term.strip() for term in query_lower.split() if term.strip()]
                
                # More sophisticated matching
                for line in lines:
                    line_lower = line.lower().strip()
                    if line_lower:  # Skip empty lines
                        # Check if any query term is in the line or if line contains relevant keywords
                        if any(term in line_lower for term in query_terms):
                            relevant_lines.append(line.strip())
                        # Add context lines (lines before and after matches)
                        elif relevant_lines and len(relevant_lines) < 100:  # Add some context
                            relevant_lines.append(line.strip())
                
                if relevant_lines:
                    # Limit to reasonable number of lines
                    if len(relevant_lines) > 100:
                        relevant_lines = relevant_lines[:100]
                        truncated_msg = "\n\n[Content truncated - showing first 100 relevant lines]"
                    else:
                        truncated_msg = ""
                    
                    result = f"=== RELEVANT CONTENT FOR '{query}' ===\n\n" + '\n'.join(relevant_lines) + truncated_msg + "\n\n=== END OF RELEVANT CONTENT ==="
                    print(f"[DEBUG] Returning relevant content ({len(result)} characters)")
                    return result
                else:
                    # If no specific matches, return a substantial sample
                    sample_length = min(5000, len(text_content))
                    result = f"=== NO EXACT MATCHES FOR '{query}' ===\n\nDocument sample:\n\n{text_content[:sample_length]}{'...\n\n[Document continues]' if len(text_content) > sample_length else ''}\n\n=== END OF SAMPLE ==="
                    print(f"[DEBUG] No matches found, returning sample ({len(result)} characters)")
                    return result
                    
        except Exception as e:
            error_msg = f"CRITICAL ERROR reading PDF: {str(e)}. The PDF may be corrupted, password-protected, or in an unsupported format."
            print(f"[DEBUG] {error_msg}")
            return error_msg