"""
Azure AI-powered skill extraction module for resume analysis.
Simple skill extraction that returns technical skills grouped by categories.
"""

import os
import json
import re
import sys
from typing import Dict, List, Optional
from dotenv import load_dotenv
from crewai.llm import LLM

# Fix import issue when running directly
try:
    from .tools.custom_tool import CustomPDFTool
except ImportError:
    # Add current directory to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from tools.custom_tool import CustomPDFTool  # type: ignore[import-not-found]

# Load environment variables
load_dotenv()


def extract_skills_from_pdf(
    pdf_path: Optional[str] = None,
    job_description: Optional[str] = None,
    job_title: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Extract technical skills from either a PDF resume or a raw job description string using Azure AI.

    Args:
        pdf_path: Optional path to the PDF resume file.
    job_description: Optional raw text of a job description.
    job_title: Optional job title context to improve extraction when job_description is provided.

    Notes:
        - Provide exactly one of pdf_path or job_description.
        - Returns only technical skills grouped by standardized categories.

    Returns:
        Dictionary with categories as keys and lists of skills as values.
        Example: {"Business Intelligence": ["Power BI", "Tableau"], "Programming Languages": ["Python", "SQL"]}
    """
    try:
        # Step 1: Acquire input text from either PDF or provided job description
        if (pdf_path is None) and (job_description is None):
            return {"Error": ["No input provided. Pass either pdf_path or job_description."]}
        if (pdf_path is not None) and (job_description is not None):
            return {"Error": ["Provide only one input: either pdf_path or job_description, not both."]}

        if job_description is not None:
            input_text = job_description.strip()
            if not input_text:
                return {"Error": ["Provided job_description is empty."]}
        else:
            pdf_tool = CustomPDFTool(pdf_path=pdf_path)  # type: ignore[arg-type]
            input_text = pdf_tool._run("extract_all")
            if not input_text or "ERROR" in input_text:
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
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            temperature=0.0,
            stream=False,
        )

        # Step 3: Create skill extraction prompt
        context_block = f"RESUME OR JOB DESCRIPTION TEXT:\n{input_text}"
        if job_title:
            context_block = f"JOB TITLE: {job_title}\n\n" + context_block

        prompt = f"""You are an expert HR assistant that extracts technical skills from resumes or job descriptions and groups them into standardized categories.

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
1. Extract ALL technical skills mentioned.
2. Group them into the appropriate categories above.
3. Return ONLY a JSON object with categories as keys and skill arrays as values.
4. Do NOT include soft skills, certifications, or languages.
5. Use exact category names from the list above.

{context_block}

Return ONLY valid JSON in this format:
{{
  "Business Intelligence": ["Power BI", "Tableau"],
  "Programming Languages": ["Python", "SQL", "JavaScript"],
  "Database & Data": ["MySQL", "PostgreSQL"]
}}"""

        # Step 4: Call LLM for skill extraction
        full_prompt = f"""You are an expert resume/job description skill extraction assistant. Extract only technical skills and group them by categories.

{prompt}"""

        response = llm.call([{"role": "user", "content": full_prompt}])

        # Step 5: Parse response
        response_text = response.strip()

        # Extract JSON from response - improved parsing
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)

        if not json_match:
            return {"Error": [f"No JSON found in response: {response_text[:200]}..."]}

        json_text = json_match.group(0).strip()
        last_brace = json_text.rfind('}')
        if last_brace != -1:
            json_text = json_text[:last_brace + 1]

        try:
            skills_data = json.loads(json_text)
        except json.JSONDecodeError as e:
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

        cleaned_skills: Dict[str, List[str]] = {}
        if isinstance(skills_data, dict):
            for category, skills_list in skills_data.items():
                if isinstance(skills_list, list) and skills_list:
                    clean_skills: List[str] = []
                    for skill in skills_list:
                        if isinstance(skill, str):
                            skill_clean = skill.strip()
                            if skill_clean and skill_clean not in clean_skills:
                                clean_skills.append(skill_clean)
                    if clean_skills:
                        cleaned_skills[str(category)] = clean_skills

        return cleaned_skills
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
