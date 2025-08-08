"""
Azure AI-powered skill extraction module for resume analysis.
Simple skill extraction that returns technical skills grouped by categories.
"""

import os
import json
import re
import sys
from typing import Dict, List
from dotenv import load_dotenv
from crewai.llm import LLM

# Fix import issue when running directly
try:
    from .tools.custom_tool import CustomPDFTool
except ImportError:
    # Add current directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from tools.custom_tool import CustomPDFTool

# Load environment variables
load_dotenv()


def extract_skills_from_pdf(pdf_path: str) -> Dict[str, List[str]]:
    """
    Extract technical skills from a PDF resume using Azure AI.
    
    Args:
        pdf_path: Path to the PDF resume file
        
    Returns:
        Dictionary with categories as keys and lists of skills as values
        Format: {"Business Intelligence": ["Power BI", "Tableau"], "Programming Languages": ["Python", "SQL"]}
    """
    try:
        # Step 1: Extract text from PDF using custom tool
        pdf_tool = CustomPDFTool(pdf_path=pdf_path)
        resume_text = pdf_tool._run("extract_all")
        
        if not resume_text or "ERROR" in resume_text:
            return {"Error": [f"Failed to extract text from PDF: {pdf_path}"]}
        
        # Step 2: Initialize LLM client
        model = os.getenv("model")
        api_key = os.getenv("AZURE_AI_API_KEY")
        base_url = os.getenv("AZURE_AI_ENDPOINT")
        api_version = os.getenv("AZURE_AI_API_VERSION")
        
        if not model or not api_key or not base_url:
            return {"Error": ["Azure AI credentials not found. Set model, AZURE_AI_API_KEY, AZURE_AI_ENDPOINT, and AZURE_AI_API_VERSION in .env file"]}
        
        # Configure LLM with Azure AI
        llm = LLM(
            model=model,  # This should match your deployment name
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            temperature=0.0,  # Reduced for more consistent outputs
            stream=False,
        )
        
        # Step 3: Create skill extraction prompt
        prompt = f"""You are an expert HR assistant that extracts technical skills from resumes and groups them into standardized categories.

SKILL CATEGORIES TO USE:
- Business Intelligence: Power BI, Tableau, QlikView, Looker, etc.
- Programming Languages: Python, Java, JavaScript, SQL, C++, etc.
- Database & Data: MySQL, PostgreSQL, MongoDB, Data Analysis, etc.
- Web Development: HTML, CSS, React, Angular, Node.js, etc.
- Cloud & DevOps: AWS, Azure, Docker, Kubernetes, Git, etc.
- Machine Learning & AI: TensorFlow, PyTorch, scikit-learn, etc.
- Mobile Development: iOS, Android, React Native, Flutter, etc.
- Design & UX: Figma, Adobe XD, Photoshop, UI/UX Design, etc.
- Analytics & Reporting: Excel Advanced, Google Analytics, etc.
- Security: Cybersecurity, Penetration Testing, etc.

INSTRUCTIONS:
1. Extract ALL technical skills mentioned in the resume
2. Group them into the appropriate categories above
3. Return ONLY a JSON object with categories as keys and skill arrays as values
4. Do NOT include soft skills, certifications, or languages
5. Use exact category names from the list above

RESUME TEXT:
{resume_text}

Return ONLY valid JSON in this format:
{{
  "Business Intelligence": ["Power BI", "Tableau"],
  "Programming Languages": ["Python", "SQL", "JavaScript"],
  "Database & Data": ["MySQL", "PostgreSQL"]
}}"""

        # Step 4: Call LLM for skill extraction
        full_prompt = f"""You are an expert resume skill extraction assistant. Extract only technical skills and group them by categories.

{prompt}"""
        
        response = llm.call([{"role": "user", "content": full_prompt}])
        
        # Step 5: Parse response
        response_text = response.strip()
        
        # Extract JSON from response - improved parsing
        # First try to find JSON between curly braces
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if not json_match:
            # If no match, try a more flexible approach
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(0)
            
            # Clean the JSON text
            json_text = json_text.strip()
            
            # Remove any trailing content after the last }
            last_brace = json_text.rfind('}')
            if last_brace != -1:
                json_text = json_text[:last_brace + 1]
            
            try:
                skills_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # If parsing fails, try to extract just the content between first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_text = response_text[start:end+1]
                    try:
                        skills_data = json.loads(json_text)
                    except json.JSONDecodeError:
                        return {"Error": [f"JSON parsing failed: {str(e)}. Response: {response_text[:200]}..."]}
                else:
                    return {"Error": [f"No valid JSON found in response: {response_text[:200]}..."]}
            
            # Clean and validate the data
            cleaned_skills = {}
            for category, skills_list in skills_data.items():
                if isinstance(skills_list, list) and skills_list:
                    # Remove empty strings and duplicates
                    clean_skills = []
                    for skill in skills_list:
                        if isinstance(skill, str) and skill.strip():
                            skill_clean = skill.strip()
                            if skill_clean not in clean_skills:
                                clean_skills.append(skill_clean)
                    
                    if clean_skills:
                        cleaned_skills[category] = clean_skills
            
            return cleaned_skills
        else:
            return {"Error": [f"No JSON found in response: {response_text[:200]}..."]}
            
    except Exception as e:
        return {"Error": [f"Skill extraction failed: {str(e)}"]}


# Example usage
if __name__ == "__main__":
    # Test with a PDF file
    pdf_file = "C://Users//benlallahom_yo//Desktop//veo-master-douaa//abbes-ghassen-cv.pdf"  # Replace with actual PDF path
    
    if os.path.exists(pdf_file):
        skills = extract_skills_from_pdf(pdf_file)
        print("Extracted Skills:")
        print(json.dumps(skills, indent=2))
    else:
        print(f"PDF file not found: {pdf_file}")
        print("Please provide a valid PDF path to test the extraction.")