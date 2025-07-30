from crewai.tools import BaseTool
from typing import Type, Union, Any
from pydantic import BaseModel, Field, field_validator
import fitz  # PyMuPDF
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


def detect_visual_skill_indicators(page):
    """Detect visual skill proficiency indicators like progress bars, dots, stars."""
    visual_skills = []
    
    try:
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # 1. Detect filled/unfilled shapes (progress bars, dots, stars)
        drawings = page.get_drawings()
        
        # Group shapes by vertical position (same skill line)
        shape_groups = {}
        tolerance = 5  # pixels
        
        for drawing in drawings:
            if 'items' in drawing:
                for item in drawing['items']:
                    if item[0] in ['re', 'c']:  # rectangles and circles
                        bbox = item[1]  # bounding box
                        y_pos = round(bbox[1] / tolerance) * tolerance
                        
                        if y_pos not in shape_groups:
                            shape_groups[y_pos] = []
                        
                        shape_groups[y_pos].append({
                            'type': 'rectangle' if item[0] == 're' else 'circle',
                            'bbox': bbox,
                            'filled': 'f' in drawing.get('fill', {}),
                            'color': drawing.get('fill', {}).get('color', 0)
                        })
        
        # 2. Analyze each group to determine skill proficiency
        for y_pos, shapes in shape_groups.items():
            if len(shapes) >= 3:  # At least 3 indicators suggest a skill rating
                # Sort shapes by x position (left to right)
                shapes.sort(key=lambda x: x['bbox'][0])
                
                # Count filled vs unfilled
                filled_count = sum(1 for shape in shapes if shape['filled'] or shape['color'] > 0)
                total_count = len(shapes)
                
                if total_count > 0:
                    proficiency = round((filled_count / total_count) * 100)
                    
                    # Try to find nearby text (skill name)
                    text_blocks = page.get_text("dict")
                    skill_name = None
                    
                    for block in text_blocks["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                line_bbox = line["bbox"]
                                # Check if text is on the same horizontal line
                                if abs(line_bbox[1] - y_pos) < 20:  # within 20 pixels
                                    line_text = ""
                                    for span in line["spans"]:
                                        line_text += span["text"] + " "
                                    
                                    if line_text.strip():
                                        skill_name = line_text.strip()
                                        break
                    
                    if skill_name:
                        visual_skills.append({
                            'skill': skill_name,
                            'proficiency': proficiency,
                            'indicators': f"{filled_count}/{total_count}",
                            'type': 'visual_gauge'
                        })
        
        # 3. Detect star ratings (★★★☆☆ patterns)
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    
                    # Look for star patterns
                    star_pattern = re.search(r'([★✦✧⭐🌟]+)([☆✧⭒🌟]*)', line_text)
                    if star_pattern:
                        filled_stars = len(star_pattern.group(1))
                        empty_stars = len(star_pattern.group(2))
                        total_stars = filled_stars + empty_stars
                        
                        if total_stars > 0:
                            proficiency = round((filled_stars / total_stars) * 100)
                            
                            # Extract skill name from the same line
                            skill_text = re.sub(r'[★✦✧⭐🌟☆✧⭒]+', '', line_text).strip()
                            if skill_text:
                                visual_skills.append({
                                    'skill': skill_text,
                                    'proficiency': proficiency,
                                    'indicators': f"{filled_stars}/{total_stars} stars",
                                    'type': 'star_rating'
                                })
        
        # 4. Detect percentage indicators
        text = page.get_text()
        percentage_pattern = re.finditer(r'(\w+(?:\s+\w+)*)\s*[:\-]?\s*(\d+)%', text)
        for match in percentage_pattern:
            skill_name = match.group(1).strip()
            percentage = int(match.group(2))
            
            # Filter for skill-related terms
            skill_keywords = ['javascript', 'python', 'java', 'css', 'html', 'sql', 'react', 'angular', 'vue', 'node', 'php', 'c++', 'c#', 'ruby', 'go', 'swift', 'kotlin', 'photoshop', 'excel', 'powerpoint', 'word', 'autocad', 'tableau', 'powerbi']
            if any(keyword in skill_name.lower() for keyword in skill_keywords) or len(skill_name.split()) <= 3:
                visual_skills.append({
                    'skill': skill_name,
                    'proficiency': percentage,
                    'indicators': f"{percentage}%",
                    'type': 'percentage'
                })
        
        # 5. Detect common progress bar symbols (■■■□□, ●●●○○, etc.)
        progress_patterns = [
            (r'([■▪▇█]+)([□▫▢▣]*)', 'squares'),
            (r'([●●●]+)([○○○]*)', 'circles'),
            (r'([♦♦♦]+)([◊◊◊]*)', 'diamonds')
        ]
        
        for pattern, pattern_type in progress_patterns:
            for match in re.finditer(pattern, text):
                filled = len(match.group(1))
                empty = len(match.group(2))
                total = filled + empty
                
                if total > 0:
                    proficiency = round((filled / total) * 100)
                    
                    # Look for skill name in the same line
                    match_start = match.start()
                    line_start = text.rfind('\n', 0, match_start) + 1
                    line_end = text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(text)
                    
                    line_content = text[line_start:line_end]
                    skill_name = re.sub(pattern, '', line_content).strip()
                    
                    if skill_name and len(skill_name) > 2:
                        visual_skills.append({
                            'skill': skill_name,
                            'proficiency': proficiency,
                            'indicators': f"{filled}/{total} {pattern_type}",
                            'type': f'{pattern_type}_gauge'
                        })
        
    except Exception as e:
        print(f"[DEBUG] Visual skill detection failed: {e}")
    
    return visual_skills


def detect_text_based_proficiency(page):
    """Detect text-based skill proficiency indicators and descriptive levels."""
    text_skills = []
    
    try:
        text = page.get_text()
        lines = text.split('\n')
        
        # Define proficiency level mappings
        proficiency_levels = {
            # Expert/Advanced levels (85-100%)
            'expert': 95, 'expertise': 95, 'mastery': 95, 'master': 95,
            'advanced': 85, 'excellent': 90, 'outstanding': 95, 'exceptional': 95,
            'proficient': 80, 'fluent': 85, 'native': 100, 'mother': 100,
            
            # Intermediate levels (60-84%)
            'intermediate': 70, 'good': 75, 'solid': 70, 'strong': 75,
            'competent': 65, 'skilled': 75, 'experienced': 80,
            'working knowledge': 65, 'working': 65, 'conversational': 70,
            
            # Basic/Beginner levels (30-59%)
            'basic': 45, 'beginner': 35, 'novice': 30, 'elementary': 40,
            'limited': 35, 'fair': 50, 'some': 40, 'little': 30,
            'learning': 25, 'studying': 25, 'familiar': 50,
            
            # Years of experience mapping
            '10+ years': 95, '8+ years': 90, '5+ years': 85, '3+ years': 75,
            '2+ years': 65, '1+ year': 55, '6 months': 40, '< 1 year': 35
        }
        
        # 1. Detect explicit proficiency patterns
        proficiency_patterns = [
            # Pattern: "Skill: Level" or "Skill - Level"
            r'([A-Za-z\s\+\#\.]+?)[\s]*[:|-][\s]*([a-zA-Z\s\+]+)',
            # Pattern: "Level in Skill" or "Level Skill"
            r'([a-zA-Z\s]+?)[\s]+(?:in|with|of)[\s]+([A-Za-z\s\+\#\.]+)',
            # Pattern: "X years of Skill" or "X+ years Skill"
            r'(\d+\+?\s*(?:years?|yrs?))\s+(?:of|in|with)?\s*([A-Za-z\s\+\#\.]+)',
        ]
        
        for pattern in proficiency_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'years' in match.group(1).lower():
                    # Years pattern
                    years_text = match.group(1).strip()
                    skill_name = match.group(2).strip()
                    proficiency = proficiency_levels.get(years_text.lower(), None)
                else:
                    # Regular proficiency pattern
                    skill_name = match.group(1).strip()
                    level_text = match.group(2).strip().lower()
                    proficiency = proficiency_levels.get(level_text, None)
                
                if proficiency and len(skill_name) > 2:
                    # Clean skill name
                    skill_name = re.sub(r'[^\w\s\+\#\.]', '', skill_name).strip()
                    if skill_name and not any(stop_word in skill_name.lower() for stop_word in ['the', 'and', 'or', 'with', 'years', 'experience']):
                        text_skills.append({
                            'skill': skill_name,
                            'proficiency': proficiency,
                            'indicators': level_text if 'years' not in match.group(1).lower() else years_text,
                            'type': 'text_proficiency'
                        })
        
        # 2. Detect skills in sections with contextual proficiency
        skill_sections = {}
        current_section = None
        section_patterns = [
            r'##?\s*(skills?|technical\s*skills?|programming|languages?|tools?)',
            r'##?\s*(data\s*skills?|network\s*skills?|soft\s*skills?)',
        ]
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Check if this is a skill section header
            section_found = False
            for section_pattern in section_patterns:
                if re.search(section_pattern, line_clean, re.IGNORECASE):
                    current_section = line_clean
                    skill_sections[current_section] = []
                    section_found = True
                    break
            
            # If we're in a skills section, collect skills
            if not section_found and current_section and line_clean:
                # Skip if it's another header
                if line_clean.startswith('#') or line_clean.startswith('='):
                    current_section = None
                    continue
                
                # Extract skills from bullet points or plain lists
                skill_line = re.sub(r'^[\s•\-\*\d\.\)]+', '', line_clean)
                
                if skill_line and len(skill_line) > 1:
                    # Check for embedded proficiency in the line
                    embedded_proficiency = None
                    for level, score in proficiency_levels.items():
                        if level in skill_line.lower():
                            embedded_proficiency = score
                            skill_line = re.sub(re.escape(level), '', skill_line, flags=re.IGNORECASE).strip()
                            break
                    
                    # Split multiple skills if separated by commas or other delimiters
                    potential_skills = re.split(r'[,;/\|]', skill_line)
                    
                    for skill in potential_skills:
                        skill = skill.strip()
                        if len(skill) > 1 and not skill.isdigit():
                            # Clean the skill name
                            skill_clean = re.sub(r'[^\w\s\+\#\.\-]', '', skill).strip()
                            
                            if skill_clean:
                                # Assign contextual proficiency based on section
                                if not embedded_proficiency:
                                    if 'language' in current_section.lower():
                                        embedded_proficiency = 70  # Default for languages
                                    elif 'data' in current_section.lower():
                                        embedded_proficiency = 75  # Data skills tend to be more advanced
                                    else:
                                        embedded_proficiency = 65  # General technical skills
                                
                                text_skills.append({
                                    'skill': skill_clean,
                                    'proficiency': embedded_proficiency,
                                    'indicators': f"Listed in {current_section}",
                                    'type': 'contextual_proficiency'
                                })
        
        # 3. Detect numeric proficiency (1-10 scale, percentages)
        numeric_patterns = [
            r'([A-Za-z\s\+\#\.]+?)[\s]*[:|-][\s]*(\d+)/10',  # Skill: 8/10
            r'([A-Za-z\s\+\#\.]+?)[\s]*[:|-][\s]*(\d+)%',     # Skill: 85%
            r'([A-Za-z\s\+\#\.]+?)[\s]*[:|-][\s]*(\d+)/5',    # Skill: 4/5
        ]
        
        for pattern in numeric_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill_name = match.group(1).strip()
                score = int(match.group(2))
                
                if '/10' in pattern:
                    proficiency = score * 10
                elif '%' in pattern:
                    proficiency = score
                elif '/5' in pattern:
                    proficiency = score * 20
                
                if len(skill_name) > 2 and 0 <= proficiency <= 100:
                    skill_name = re.sub(r'[^\w\s\+\#\.]', '', skill_name).strip()
                    text_skills.append({
                        'skill': skill_name,
                        'proficiency': proficiency,
                        'indicators': match.group(2) + ('/' + pattern.split('/')[-1].split(')')[0] if '/' in pattern else '%'),
                        'type': 'numeric_proficiency'
                    })
        
        # 4. Remove duplicates and clean up
        seen_skills = set()
        unique_skills = []
        
        for skill_info in text_skills:
            skill_key = skill_info['skill'].lower().replace(' ', '')
            if skill_key not in seen_skills and len(skill_info['skill']) > 2:
                seen_skills.add(skill_key)
                unique_skills.append(skill_info)
        
    except Exception as e:
        print(f"[DEBUG] Text-based proficiency detection failed: {e}")
    
    return unique_skills


def extract_structured_content(page):
    """Extract structured content with formatting information and visual skill indicators using PyMuPDF."""
    try:
        # First get regular text content
        blocks = page.get_text("dict")
        structured_content = []
        
        for block in blocks["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            font_size = span["size"]
                            font_flags = span["flags"]
                            
                            # Detect headers (larger font or bold)
                            is_header = font_size > 12 or (font_flags & 2**4)  # Bold flag
                            is_bold = font_flags & 2**4
                            
                            if is_header:
                                text = f"## {text}"
                            elif is_bold:
                                text = f"**{text}**"
                            
                            line_text += text + " "
                    
                    if line_text.strip():
                        structured_content.append(line_text.strip())
        
        # Add visual skill indicators detection
        visual_skills = detect_visual_skill_indicators(page)
        if visual_skills:
            structured_content.append("\n--- DETECTED VISUAL SKILL PROFICIENCY ---")
            for skill_info in visual_skills:
                skill_line = f"**{skill_info['skill']}**: {skill_info['proficiency']}% proficiency ({skill_info['indicators']}, {skill_info['type']})"
                structured_content.append(skill_line)
            structured_content.append("--- END VISUAL SKILLS ---\n")
        
        # Add text-based proficiency detection
        text_skills = detect_text_based_proficiency(page)
        if text_skills:
            structured_content.append("\n--- DETECTED TEXT-BASED SKILL PROFICIENCY ---")
            for skill_info in text_skills:
                skill_line = f"**{skill_info['skill']}**: {skill_info['proficiency']}% proficiency ({skill_info['indicators']}, {skill_info['type']})"
                structured_content.append(skill_line)
            structured_content.append("--- END TEXT-BASED SKILLS ---\n")
        
        return "\n".join(structured_content)
    except Exception as e:
        print(f"[DEBUG] Structured extraction failed: {e}")
        # Fallback to simple text extraction
        return page.get_text()


class CustomPDFTool(BaseTool):
    name: str = "PDF Document Reader"
    description: str = (
        "Advanced PDF text extraction tool with intelligent content analysis using PyMuPDF. "
        "Extracts structured content preserving formatting and layout information. "
        "Detects both visual and text-based skill proficiency indicators. "
        "Visual: progress bars, star ratings, percentages, filled shapes. "
        "Text-based: descriptive levels (expert, advanced, good, basic), years of experience, numeric scores. "
        "USAGE: Call with query parameter as a string. "
        "Examples: query='extract_all' for complete structured text with all skill proficiency analysis, "
        "query='contact information' for contact details, "
        "query='work experience' for employment history, "
        "query='education' for educational background, "
        "query='skills' for comprehensive technical skills analysis with proficiency levels."
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
        """Extract text from PDF using PyMuPDF with advanced features."""
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                query = str(query)
            
            print(f"[DEBUG] PyMuPDF tool received query: '{query}'")
            
            text_content = ""
            metadata = {}
            
            # Open PDF with PyMuPDF
            doc = fitz.open(self._pdf_file_path)
            print(f"[DEBUG] Successfully opened PDF with {doc.page_count} pages")
            
            # Extract metadata
            metadata = doc.metadata
            if metadata:
                print(f"[DEBUG] PDF metadata: {metadata}")
            
            # Extract text with formatting preservation
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Method 1: Structured extraction with formatting
                try:
                    structured_text = extract_structured_content(page)
                    if structured_text:
                        text_content += f"\n=== Page {page_num + 1} ===\n{structured_text}\n"
                except Exception as struct_error:
                    print(f"[DEBUG] Structured extraction failed on page {page_num + 1}: {struct_error}")
                    
                    # Fallback to simple text extraction
                    page_text = page.get_text()
                    if page_text:
                        cleaned_text = sanitize_text(page_text)
                        if cleaned_text:
                            text_content += f"\n=== Page {page_num + 1} ===\n{cleaned_text}\n"
                
                # Extract tables if present
                try:
                    tables = page.find_tables()
                    for table_num, table in enumerate(tables):
                        table_data = table.extract()
                        if table_data:
                            text_content += f"\n--- Page {page_num + 1} Table {table_num + 1} ---\n"
                            for row in table_data:
                                if row and any(cell for cell in row if cell):
                                    clean_row = [sanitize_text(str(cell)) if cell else "" for cell in row]
                                    text_content += " | ".join(clean_row) + "\n"
                except Exception as table_error:
                    print(f"[DEBUG] Table extraction failed on page {page_num + 1}: {table_error}")
            
            doc.close()
            
            print(f"[DEBUG] Total extracted text length: {len(text_content)} characters")
            
            if not text_content.strip():
                return "ERROR: Could not extract any readable text from the PDF. The document may be image-based, corrupted, or password-protected."
            
            # Add metadata to content if available
            if metadata and any(metadata.values()):
                metadata_text = "\n=== DOCUMENT METADATA ===\n"
                for key, value in metadata.items():
                    if value:
                        metadata_text += f"{key}: {value}\n"
                text_content = metadata_text + text_content
            
            # Process the query
            query_lower = query.lower().strip()
            
            if query_lower in ["extract_all", "all", "everything", "complete", "full", "entire", "whole"]:
                result = f"=== COMPLETE PDF DOCUMENT CONTENT ===\n\n{text_content}\n\n=== END OF DOCUMENT ==="
                print(f"[DEBUG] Returning complete content ({len(result)} characters)")
                return result
            else:
                # Enhanced search with section awareness
                text_content = sanitize_text(text_content)
                lines = text_content.split('\n')
                relevant_lines = []
                query_terms = [term.strip() for term in query_lower.split() if term.strip()]
                
                # Context-aware matching
                context_window = 3  # Lines before/after a match
                matches_found = set()
                
                for i, line in enumerate(lines):
                    line_lower = line.lower().strip()
                    if line_lower:
                        # Check for matches
                        if any(term in line_lower for term in query_terms):
                            # Add context window
                            start_idx = max(0, i - context_window)
                            end_idx = min(len(lines), i + context_window + 1)
                            
                            for j in range(start_idx, end_idx):
                                if j not in matches_found:
                                    matches_found.add(j)
                                    relevant_lines.append((j, lines[j].strip()))
                
                if relevant_lines:
                    # Sort by original line order and limit results
                    relevant_lines.sort(key=lambda x: x[0])
                    content_lines = [line[1] for line in relevant_lines[:100]]
                    
                    truncated_msg = "\n\n[Content truncated - showing first 100 relevant lines]" if len(relevant_lines) > 100 else ""
                    result = f"=== RELEVANT CONTENT FOR '{query}' ===\n\n" + '\n'.join(content_lines) + truncated_msg + "\n\n=== END OF RELEVANT CONTENT ==="
                    print(f"[DEBUG] Returning relevant content ({len(result)} characters)")
                    return result
                else:
                    # Return structured sample if no matches
                    sample_length = min(8000, len(text_content))  # Larger sample due to better extraction
                    result = f"=== NO EXACT MATCHES FOR '{query}' ===\n\nDocument sample:\n\n{text_content[:sample_length]}{'...\n\n[Document continues]' if len(text_content) > sample_length else ''}\n\n=== END OF SAMPLE ==="
                    print(f"[DEBUG] No matches found, returning sample ({len(result)} characters)")
                    return result
                    
        except Exception as e:
            error_msg = f"CRITICAL ERROR reading PDF with PyMuPDF: {str(e)}. The PDF may be corrupted, password-protected, or in an unsupported format."
            print(f"[DEBUG] {error_msg}")
            return error_msg