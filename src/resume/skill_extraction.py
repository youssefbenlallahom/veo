"""
Skill extraction module for resumes or job descriptions with robust fallbacks.

Order of strategies:
1) Azure AI via crewai.llm (if configured and reachable)
2) Google Gemini via google-genai (if API key present)
3) Deterministic keyword-based extraction (always available)
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

        # Try Azure first if configured
        azure_model = os.getenv("model")
        azure_key = os.getenv("AZURE_AI_API_KEY")
        azure_base = os.getenv("AZURE_AI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_AI_API_VERSION")

        if azure_model and azure_key and azure_base:
            try:
                return _extract_with_azure(input_text, job_title, azure_model, azure_key, azure_base, azure_api_version)
            except Exception as e:
                print(f"[extract_skills] Azure extraction failed, falling back. Reason: {e}")

        # Try Gemini if API key present
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                return _extract_with_gemini(input_text, job_title, gemini_key)
            except Exception as e:
                print(f"[extract_skills] Gemini extraction failed, falling back. Reason: {e}")

        # Fallback: deterministic keyword-based extraction
        return _extract_with_keywords(input_text)
    except Exception as e:
        return {"Error": [f"Skill extraction failed: {str(e)}"]}


def _extract_with_azure(input_text: str, job_title: Optional[str], model: str, api_key: str, base_url: str, api_version: Optional[str]) -> Dict[str, List[str]]:
    # Configure LLM with Azure AI
    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        temperature=0.0,
        stream=False,
    )

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

    full_prompt = f"""You are an expert resume/job description skill extraction assistant. Extract only technical skills and group them by categories.

{prompt}"""

    response = llm.call([{"role": "user", "content": full_prompt}])
    response_text = response.strip()

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in Azure response: {response_text[:200]}...")

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
            skills_data = json.loads(response_text[start:end+1])
        else:
            raise ValueError(f"JSON parsing failed: {str(e)}")

    return _clean_skills_dict(skills_data)


def _extract_with_gemini(input_text: str, job_title: Optional[str], api_key: str) -> Dict[str, List[str]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    jt = job_title or ""
    prompt = f"""
You are an AI that extracts ONLY technical skills and groups them under canonical categories.

Categories:
- Business Intelligence, Programming Languages, Database & Data, Web Development,
- Cloud & DevOps, Machine Learning & AI, Mobile Development, Design & UX,
- Analytics & Reporting, Security

Text Title: {jt}
Text:
{input_text}

Return strict JSON: keys are category names above, values are arrays of skills. No extra text.
"""
    resp = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
    m = re.search(r'\{.*\}', resp.text, re.DOTALL)
    if not m:
        raise ValueError("No JSON in Gemini response")
    skills_data = json.loads(m.group(0))
    return _clean_skills_dict(skills_data)


def _extract_with_keywords(text: str) -> Dict[str, List[str]]:
    """Deterministic keyword-based extractor as a safety net."""
    t = text.lower()
    kb: Dict[str, Dict[str, List[str]]] = {
        "Business Intelligence": {
            "Power BI": ["power bi", "microsoft power bi"],
            "Tableau": ["tableau"],
            "QlikView": ["qlik", "qlikview"],
            "Looker": ["looker"],
        },
        "Programming Languages": {
            "Python": ["python"],
            "Java": ["java"],
            "JavaScript": ["javascript", "js"],
            "TypeScript": ["typescript", "ts"],
            "C++": ["c++"],
            "SQL": ["sql", "postgresql", "mysql", "sqlite", "sql server"],
        },
        "Database & Data": {
            "MySQL": ["mysql"],
            "PostgreSQL": ["postgresql", "postgres"],
            "MongoDB": ["mongodb"],
            "Data Analysis": ["data analysis", "data analytics"],
            "ETL": ["etl"],
            "Data Modeling": ["data modeling", "data modelling"],
        },
        "Web Development": {
            "HTML": ["html"],
            "CSS": ["css"],
            "React": ["react"],
            "Angular": ["angular"],
            "Node.js": ["node.js", "nodejs", "node"],
            "Vue": ["vue", "vue.js", "nuxt"],
            "Next.js": ["next.js", "nextjs"],
        },
        "Cloud & DevOps": {
            "AWS": ["aws", "amazon web services"],
            "Azure": ["azure"],
            "GCP": ["gcp", "google cloud"],
            "Docker": ["docker"],
            "Kubernetes": ["kubernetes", "k8s"],
            "CI/CD": ["ci/cd", "cicd"],
            "Git": ["git"],
            "Terraform": ["terraform"],
        },
        "Machine Learning & AI": {
            "TensorFlow": ["tensorflow"],
            "PyTorch": ["pytorch"],
            "scikit-learn": ["scikit-learn", "sklearn"],
            "Machine Learning": ["machine learning", "ml"],
            "AI": ["artificial intelligence", "ai"],
        },
        "Mobile Development": {
            "iOS": ["ios"],
            "Android": ["android"],
            "React Native": ["react native"],
            "Flutter": ["flutter"],
            "Swift": ["swift"],
            "Kotlin": ["kotlin"],
        },
        "Design & UX": {
            "Figma": ["figma"],
            "Adobe XD": ["adobe xd"],
            "Photoshop": ["photoshop"],
            "Illustrator": ["illustrator"],
            "UI/UX Design": ["ui/ux", "ux", "ui"],
        },
        "Analytics & Reporting": {
            "Excel Advanced": ["excel", "excel advanced", "power query", "power pivot"],
            "Google Analytics": ["google analytics"],
            "Dashboards": ["dashboard", "dashboards"],
            "Reporting": ["reporting", "reports"],
        },
        "Security": {
            "Cybersecurity": ["cybersecurity", "information security"],
            "Penetration Testing": ["penetration testing", "pentest", "pen testing"],
            "SIEM": ["siem"],
            "SOC": ["soc"],
            "Splunk": ["splunk"],
        },
    }

    result: Dict[str, List[str]] = {}
    for category, skills in kb.items():
        found: List[str] = []
        for canonical, variants in skills.items():
            for v in variants:
                if v in t:
                    found.append(canonical)
                    break
        if found:
            # dedupe preserve order
            seen = set()
            clean = [s for s in found if not (s in seen or seen.add(s))]
            result[category] = clean
    return result


def _clean_skills_dict(skills_data) -> Dict[str, List[str]]:
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
