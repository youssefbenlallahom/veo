"""import sys
import pysqlite3

# Monkey-patch sqlite3 before ANYTHING imports it
sys.modules["sqlite3"] = pysqlite3"""

import sys
import warnings
import os
import streamlit as st
import PyPDF2
import tempfile
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from crew import Resume
import sys
import os

    
load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def sanitize_text(text):
    """Sanitize text to remove problematic characters for display."""
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


def validate_pdf(file_path):
    """Validate that the PDF is readable."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            if num_pages == 0:
                return False, "PDF has no pages"

            # Try to read first page text
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            return True, f"PDF is valid with {num_pages} pages"
    except Exception as e:
        return False, f"PDF validation failed: {str(e)}"


import re

def clean_extracted_text(text):
    cleaned_lines = []
    prev_blank = False
    for line in text.splitlines():
        # Remove lines that are only pipes and spaces (table delimiters)
        if re.fullmatch(r'\s*\|[\s\|]*\|?\s*', line):
            continue
        # Remove lines that are only pipes or empty after pipes
        if re.fullmatch(r'\s*\|+\s*', line):
            continue
        # Remove lines that are only whitespace
        if not line.strip():
            if prev_blank:
                continue  # Avoid multiple consecutive blanks
            prev_blank = True
            cleaned_lines.append('')
            continue
        prev_blank = False
        # Remove trailing pipes and spaces
        line = re.sub(r'\s*\|+\s*$', '', line)
        # Optionally: collapse multiple spaces
        line = re.sub(r' {2,}', ' ', line)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def analyze_resume(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis and save the report with a unique filename."""
    try:
        # Create Resume instance and process
        resume_crew = Resume(pdf_path=resume_path)
        # Extract raw data from the PDF (no saving to file)
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
            except Exception as e:
                print(f"[DEBUG] Could not extract raw PDF data: {e}")
        inputs = {
            'pdf': resume_path,
            'job_title': job_title,
            'job_description': job_description,
            'current_year': str(datetime.now().year)
        }
        if barem is not None:
            inputs['barem'] = barem

        # Run the analysis
        resume_crew.crew().kickoff(inputs=inputs)

        # Generate a unique report filename
        sanitized_name = re.sub(r'[^\w\-_\. ]', '_', candidate_name)  # Replace invalid characters
        report_filename = f'report_{sanitized_name}.md'

        # Ensure the report is renamed after each analysis
        if os.path.exists('report.md'):
            # Check if the target file exists and remove it
            if os.path.exists(report_filename):
                os.remove(report_filename)
            os.rename('report.md', report_filename)

        return report_filename

    except Exception as e:
        raise RuntimeError(f"Error analyzing resume for {candidate_name}: {str(e)}")


async def analyze_resume_async(resume_path, job_title, job_description, candidate_name, barem=None):
    """Run analysis asynchronously and save the report with a unique filename."""
    try:
        # Create Resume instance and process
        resume_crew = Resume(pdf_path=resume_path)
        # Extract raw data from the PDF (no saving to file)
        if hasattr(resume_crew, 'pdf_tool') and resume_crew.pdf_tool is not None:
            try:
                extracted_data = resume_crew.pdf_tool._run('extract_all')
                # Clean up table delimiters and excess blank lines
                cleaned_data = clean_extracted_text(extracted_data)
            except Exception as e:
                print(f"[DEBUG] Could not extract raw PDF data: {e}")
        inputs = {
            'pdf': resume_path,
            'job_title': job_title,
            'job_description': job_description,
            'current_year': str(datetime.now().year)
        }
        if barem is not None:
            inputs['barem'] = barem

        # Run the analysis asynchronously
        result = await resume_crew.crew().kickoff_async(inputs=inputs)

        # Generate a unique report filename
        sanitized_name = re.sub(r'[^\w\-_\. ]', '_', candidate_name)  # Replace invalid characters
        report_filename = f'report_{sanitized_name}.md'

        # Ensure the report is renamed after each analysis
        if os.path.exists('report.md'):
            # Check if the target file exists and remove it
            if os.path.exists(report_filename):
                os.remove(report_filename)
            os.rename('report.md', report_filename)

        return report_filename

    except Exception as e:
        raise RuntimeError(f"Error analyzing resume for {candidate_name}: {str(e)}")


async def analyze_single_resume_with_error_handling(resume_pdf, job_title, job_description, barem, semaphore):
    """Analyze a single resume with error handling and concurrency control."""
    async with semaphore:  # Limit concurrent executions
        resume_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(resume_pdf.getbuffer())
                resume_path = tmp_file.name

            # Validate PDF
            is_valid, validation_message = validate_pdf(resume_path)
            if not is_valid:
                return {
                    "filename": resume_pdf.name,
                    "valid": False,
                    "error": validation_message,
                    "score": 0,
                    "recommendation": "Failed",
                    "report_content": ""
                }

            # Extract candidate name from the filename
            candidate_name = os.path.splitext(resume_pdf.name)[0]

            # Analyze resume asynchronously
            report_filename = await analyze_resume_async(
                resume_path=resume_path,
                job_title=job_title,
                job_description=job_description,
                candidate_name=candidate_name,
                barem=barem
            )

            # Parse the generated report
            candidate_result = parse_report(report_filename, resume_pdf.name)
            print(f"DEBUG: Parsed result for {resume_pdf.name}: score={candidate_result['score']}")
            return candidate_result

        except Exception as e:
            return {
                "filename": resume_pdf.name,
                "valid": False,
                "error": str(e),
                "score": 0,
                "recommendation": "Error",
                "report_content": ""
            }

        finally:
            # Clean up the temporary file silently
            if resume_path and os.path.exists(resume_path):
                try:
                    os.remove(resume_path)
                except Exception:
                    pass  # Silent cleanup


async def analyze_all_resumes_async(resume_pdfs, job_title, job_description, barem, progress_callback=None):
    """Analyze all resumes concurrently with progress tracking."""
    # Create semaphore to limit concurrent executions (prevent memory issues)
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent analyses
    
    # Create tasks for all resumes
    tasks = [
        analyze_single_resume_with_error_handling(
            resume_pdf, job_title, job_description, barem, semaphore
        ) 
        for resume_pdf in resume_pdfs
    ]
    
    # Execute with progress tracking
    results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        completed += 1
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(completed, len(resume_pdfs))
    
    return results


def parse_report(report_path, filename):
    """Parse the report file to extract scores and recommendations."""
    result = {
        "filename": filename,
        "valid": True,
        "error": "",
        "score": 0,
        "recommendation": "Unknown",
        "strengths": [],
        "gaps": [],
        "report_content": ""
    }

    if not os.path.exists(report_path):
        result["valid"] = False
        result["error"] = "Report file not found"
        return result

    try:
        # Read the report content
        with open(report_path, 'r', encoding='utf-8') as report_file:
            report_content = report_file.read()

        # Store full content for display
        result["report_content"] = report_content

        # Try multiple patterns to extract the score
        # Pattern 1: Look for "Overall Score: X/10" or "TOTAL WEIGHTED SCORE: X/10"
        score_match_10 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*10\]?',
            report_content,
            re.IGNORECASE
        )
        # Pattern 2: Look for "Overall Score: X/100" or "TOTAL WEIGHTED SCORE: X/100"
        score_match_100 = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*100\]?',
            report_content,
            re.IGNORECASE
        )
        # Pattern 3: Look for "Overall Score: X" or "TOTAL WEIGHTED SCORE: X" without denominator
        score_match_plain = re.search(
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\]?(?!\s*/)',
            report_content,
            re.IGNORECASE
        )
        # Pattern 4: Look for "final score for ... is **X/10**" or "final score is **X.X/10**"
        final_score_match = re.search(
            r'final score.*?is\s*\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 5: Look for "The final score for ... is **X.X/10**"
        final_score_match_2 = re.search(
            r'The final score for.*?is\s*\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 6: Look for any "final score" with **X.X/10** format  
        final_score_match_3 = re.search(
            r'final score.*?\*\*([\d\.]+)/10\*\*',
            report_content,
            re.IGNORECASE
        )
        # Pattern 7: Look for "## Final Score" section with score
        final_score_section = re.search(
            r'##\s*Final\s*Score.*?([\d\.]+)/10',
            report_content,
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 8: Look for score after "Final Score" heading
        final_score_heading = re.search(
            r'##\s*Final\s*Score.*?(\d+\.?\d*)',
            report_content,
            re.IGNORECASE | re.DOTALL
        )
        # Pattern 6: Look for "Key Contributors:" followed by weighted scores that sum up
        weighted_scores = re.findall(
            r'\[?([\d\.]+)\]?\s*(?:points|point)\s*\(Source',
            report_content
        )
        
        print(f"DEBUG: Parsing report for {filename}")
        print(f"DEBUG: Report content length: {len(report_content)}")
        print(f"DEBUG: Looking for score patterns...")
        
        if score_match_10:
            result["score"] = float(score_match_10.group(1))
            print(f"Found score /10: {result['score']}")
        elif final_score_match:
            result["score"] = float(final_score_match.group(1))
            print(f"Found final score: {result['score']}")
        elif final_score_match_2:
            result["score"] = float(final_score_match_2.group(1))
            print(f"Found final score (pattern 2): {result['score']}")
        elif final_score_match_3:
            result["score"] = float(final_score_match_3.group(1))
            print(f"Found final score (pattern 3): {result['score']}")
        elif final_score_section:
            result["score"] = float(final_score_section.group(1))
            print(f"Found final score in section: {result['score']}")
        elif final_score_heading:
            result["score"] = float(final_score_heading.group(1))
            print(f"Found final score after heading: {result['score']}")
        elif score_match_100:
            # Convert score from /100 to /10 scale
            result["score"] = float(score_match_100.group(1)) / 10
            print(f"Found score /100: {result['score']}")
        elif score_match_plain:
            # Assume it's out of 10 if no denominator is specified
            result["score"] = float(score_match_plain.group(1))
            print(f"Found plain score: {result['score']}")
        elif weighted_scores:
            # Sum up the weighted scores if individual components are found
            try:
                total = sum(float(score) for score in weighted_scores)
                result["score"] = total
                print(f"Calculated score from components: {result['score']}")
            except ValueError:
                pass
        else:
            print("DEBUG: No score pattern matched!")
            print("DEBUG: First 500 characters of report:")
            print(report_content[:500])

        # Extract recommendation
        decision_match = re.search(r'\*\*Decision:\*\* ([^\n]+)', report_content)
        if decision_match:
            result["recommendation"] = decision_match.group(1).strip()

        # Extract strengths
        strengths_match = re.search(r'### ✅ Strengths\n([\s\S]+?)### ❌ Gaps', report_content)
        if strengths_match:
            strengths = strengths_match.group(1).strip().split('\n')
            result["strengths"] = [s.strip('- ').strip() for s in strengths if s.strip()]

        # Extract gaps
        gaps_match = re.search(r'### ❌ Gaps\n([\s\S]+?)## Weighted Score Analysis', report_content)
        if gaps_match:
            gaps = gaps_match.group(1).strip().split('\n')
            result["gaps"] = [g.strip('- ').strip() for g in gaps if g.strip()]

        return result

    except Exception as e:
        result["valid"] = False
        result["error"] = f"Error parsing report: {str(e)}"
        return result


def display_comparison_table(results_data, placeholder):
    """Display a comparative table of all candidates."""
    # Sort results by score (highest first)
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)

    # Find the highest score for highlighting
    max_score = max([r["score"] for r in sorted_results if r["valid"]], default=0)

    # Create DataFrame for display
    import pandas as pd

    if sorted_results:
        df = pd.DataFrame([{
            "Rank": idx + 1,
            "Candidate": r["filename"],
            "Score": f"{r['score']:.1f}/10"  # Format as X.X/10
        } for idx, r in enumerate(sorted_results)])

        # Use placeholder to display the table
        with placeholder.container():
            st.subheader("Candidates Comparison")
            st.dataframe(df, use_container_width=True)
    else:
        placeholder.info("No valid results to display") 


def display_individual_reports(results_data):
    """Display individual reports in expandable sections."""
    st.subheader("Individual Analysis Reports")

    # Sort by score (highest first)
    sorted_results = sorted(results_data, key=lambda x: x["score"], reverse=True)

    for idx, result in enumerate(sorted_results):
        # Create expander with score and recommendation in title
        with st.expander(f"#{idx+1}: {result['filename']} - Score: {result['score']:.1f}/10"):
            if not result["valid"]:
                st.error(f"Error: {result['error']}")
            else:
                # Only show sections if they have content
                if result["strengths"]:
                    st.markdown(f"**Strengths:**")
                    for strength in result["strengths"]:
                        st.markdown(f"- {strength}")

                if result["gaps"]:
                    st.markdown(f"**Gaps:**")
                    for gap in result["gaps"]:
                        st.markdown(f"- {gap}")

                if result["report_content"]:
                    st.markdown("**Full Report:**")
                    st.markdown(result["report_content"])


def get_key_skills_from_gemini(job_title, job_description, api_key):
    """Extract key skills from job description using Gemini, grouped by Hard and Soft Skills."""
    from google import genai
    import json, re
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are a flexible skill extraction system. Analyze the job description and extract ALL relevant HARD SKILLS only - technical, measurable competencies that can be objectively evaluated.

Job Title: {job_title}
Job Description: {job_description}

STANDARDIZED HARD SKILLS DICTIONARY:
Use ONLY these exact phrases when extracting skills. Focus on technical and measurable competencies:

HARD SKILLS:
- Education: "Bachelor's or Master's Degree", "PhD", "Professional Certification", "Technical Diploma", "Industry Certification"
- Languages: "French B2 Level", "English B2 Level", "Spanish B2 Level", "German B2 Level", "Arabic B2 Level", "Italian B2 Level", "Portuguese B2 Level", "Chinese B2 Level", "Mandarin B2 Level", "Japanese B2 Level"
- Programming & Development: "Python Programming", "Java Programming", "JavaScript", "C++ Programming", "C# Programming", "PHP Programming", "Ruby Programming", "Swift Programming", "Kotlin Programming", "Go Programming", "Rust Programming", "HTML/CSS", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Framework"
- Software & Tools: "Microsoft Office Suite", "SAP", "SQL", "Tableau", "Power BI", "Excel Advanced", "AutoCAD", "SolidWorks", "Adobe Creative Suite", "Salesforce", "HubSpot", "QuickBooks", "Oracle", "MATLAB", "R Programming", "WordPress", "Photoshop", "Illustrator", "InDesign", "Figma", "Sketch", "Jira", "Confluence", "Git", "Docker", "Kubernetes"
- Technical Skills: "Data Analysis", "Business Analysis", "Financial Analysis", "Statistical Analysis", "Project Management", "Risk Management", "Quality Assurance", "Database Management", "Network Administration", "System Administration", "Cloud Computing", "Cybersecurity", "Digital Marketing", "SEO/SEM", "Social Media Marketing", "Content Marketing", "Email Marketing", "Market Research", "Financial Modeling", "Budget Management", "Inventory Management", "Supply Chain Management", "Process Improvement", "Lean Six Sigma", "Agile Methodology", "Scrum", "DevOps", "Machine Learning", "Artificial Intelligence", "Web Development", "Mobile App Development", "UI/UX Design", "Graphic Design", "Video Editing", "3D Modeling", "CAD Design"
- Industry Skills: "Customer Relationship Management", "Sales Management", "Marketing Strategy", "Human Resources Management", "Recruiting", "Training and Development", "Performance Management", "Compliance Management", "Audit", "Tax Preparation", "Investment Analysis", "Insurance", "Real Estate", "Healthcare Administration", "Legal Research", "Contract Management", "Procurement", "Logistics", "Manufacturing", "Quality Control", "Research and Development", "Clinical Research", "Regulatory Affairs", "Public Relations", "Event Management", "Hospitality Management", "Retail Management", "E-commerce", "Import/Export", "International Trade", "Account Management", "Payment Processing", "Banking Operations"

EXTRACTION RULES:
1. Read the entire job description thoroughly, including requirements, responsibilities, qualifications, and preferred skills
2. Extract ONLY hard skills - technical, measurable competencies that can be objectively evaluated
3. IGNORE soft skills like communication, leadership, teamwork, problem-solving, adaptability, creativity, etc.
4. Include skills mentioned in different sections (requirements, responsibilities, qualifications, nice-to-have, etc.)
5. Match each identified skill to the closest term in the standardized dictionary
6. List skills alphabetically
7. Extract ALL relevant hard skills - do not limit the number

ENHANCED MATCHING RULES:
- "Bachelor's OR Master's/Bachelor's and Master's/Bachelor's to Master's/University degree/College degree" → "Bachelor's or Master's Degree"
- "Microsoft Office/MS Office/Office Suite/Word Excel PowerPoint" → "Microsoft Office Suite"
- "Project management/project coordination/project planning" → "Project Management"
- "Data analysis/data analytics/data science/analytics" → "Data Analysis"
- "Financial analysis/financial modeling/financial planning" → "Financial Analysis"
- "Quality assurance/quality control/QA/QC" → "Quality Assurance"
- "Business analysis/business intelligence/BA" → "Business Analysis"
- "Customer service/customer support/client management" → "Customer Relationship Management"
- "Account management/client management/relationship management" → "Account Management"
- "Programming/coding/software development" → Match to specific language if mentioned, otherwise use "Programming" concept
- "Database/databases/DB management" → "Database Management"
- "Cloud/AWS/Azure/Google Cloud" → "Cloud Computing"
- "Security/cybersecurity/information security" → "Cybersecurity"

OUTPUT FORMAT (JSON with hard skills only):
{{
    "hard_skills": [
        "All identified hard skills listed alphabetically"
    ]
}}
"""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Extract JSON from response
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            skills_json = match.group(0)
            skills_data = json.loads(skills_json)
            
            # Ensure we have the expected structure
            hard_skills = skills_data.get('hard_skills', [])
            
            # Remove duplicates and ensure we have lists
            if not isinstance(hard_skills, list):
                hard_skills = []
            
            # Remove any empty strings or None values
            hard_skills = [skill for skill in hard_skills if skill and skill.strip()]
            
            # Return the structured data (only hard skills)
            return {
                'hard_skills': hard_skills
            }
        else:
            raise ValueError("Failed to extract skills JSON from Gemini response")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse skills JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error calling Gemini API: {str(e)}")


def create_custom_barem(skills_weights):
    """Create a barem structure from user-defined skills and weights."""
    barem = {}
    for skill, weight in skills_weights.items():
        barem[skill] = {
            "weight": weight,
            "criteria": [
                f"Has extensive experience and expertise in {skill}",
                f"Has some experience with {skill}",
                f"Shows potential or basic knowledge in {skill}"
            ]
        }
    return barem


def run():
    """
    Run the crew to analyze multiple resumes against a job description.
    """
    # Add GEMINI_API_KEY at the top of the function
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    st.markdown('<style>.stButton>button{background-color:#4CAF50;color:white;}</style>', unsafe_allow_html=True)

    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["HR", "Home", "About"])

    # Display company name
    st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
    st.title("Resume Analyzer")

    # Initialize session state for workflow step
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 'input'
    if 'extracted_skills' not in st.session_state:
        st.session_state.extracted_skills = []
    if 'skills_weights' not in st.session_state:
        st.session_state.skills_weights = {}

    # Step 1: Job Information Input
    if st.session_state.workflow_step == 'input':
        st.subheader("Job Information")
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer, Data Scientist, Product Manager")
        job_description = st.text_area("Job Description", placeholder="Paste the job description...", height=200)
        
        if st.button("Extract Key Skills from Job Description", disabled=not (job_title and job_description)):
            try:
                with st.spinner("Extracting key skills from job description..."):
                    skills_data = get_key_skills_from_gemini(job_title, job_description, GEMINI_API_KEY)
                    
                    # Extract hard skills only
                    all_skills = skills_data.get("hard_skills", [])
                    
                    # Check if any skills were extracted
                    if not all_skills:
                        st.error("No technical skills could be extracted from the job description. Please check the job description and try again.")
                        return
                    
                    st.session_state.extracted_skills = all_skills
                    st.session_state.skills_categories = skills_data  # Store the categorized structure
                    st.session_state.job_title = job_title
                    st.session_state.job_description = job_description
                    
                    # Initialize weights with equal distribution
                    equal_weight = round(100 / len(all_skills))
                    st.session_state.skills_weights = {skill: equal_weight for skill in all_skills}
                    
                    # Adjust for rounding to ensure total is 100
                    total = sum(st.session_state.skills_weights.values())
                    if total != 100 and all_skills:
                        first_skill = list(st.session_state.skills_weights.keys())[0]
                        st.session_state.skills_weights[first_skill] += (100 - total)
                    
                    st.success(f"Successfully extracted {len(all_skills)} technical skills!")
                    st.session_state.workflow_step = 'weighting'
                    st.rerun()
            except Exception as e:
                st.error(f"Error extracting skills: {str(e)}")
                st.write("Please check your job description and try again. Make sure it contains clear skill requirements.")

    # Step 2: Customize skill weights
    elif st.session_state.workflow_step == 'weighting':
        st.subheader("⚖️ Technical Skills Weighting")
        st.markdown("**Adjust the importance of each technical skill. Total must equal 100%.**")
        
        # Clean, minimal CSS
        st.markdown("""
        <style>
        .skill-item {
            background: #f8f9fa;
            padding: 0.8rem;
            border-left: 4px solid #007bff;
            margin: 0.3rem 0;
            border-radius: 0 8px 8px 0;
        }
        .skills-counter {
            background: #e3f2fd;
            color: #1976d2;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: 500;
            border: 1px solid #bbdefb;
        }
        .remove-button {
            background: none;
            border: 1px solid #dc3545;
            color: #dc3545;
            border-radius: 4px;
            padding: 0.2rem 0.5rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .remove-button:hover {
            background: #dc3545;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize removed skills tracker
        if 'removed_skills' not in st.session_state:
            st.session_state.removed_skills = set()
        
        # Create layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get current hard skills excluding removed ones
            skills_categories = st.session_state.get('skills_categories', {})
            active_skills = [skill for skill in skills_categories.get("hard_skills", []) 
                           if skill not in st.session_state.removed_skills]
            
            new_weights = {}
            
            # Display skills counter
            if active_skills:
                st.markdown(f'<div class="skills-counter">🎯 {len(active_skills)} Technical Skills</div>', unsafe_allow_html=True)
                st.markdown("")
                
                for skill in active_skills:
                    col_slider, col_remove = st.columns([5, 1])
                    
                    with col_slider:
                        new_weights[skill] = st.slider(
                            skill,
                            min_value=0,
                            max_value=100,
                            value=st.session_state.skills_weights.get(skill, 0),
                            step=1,
                            key=f"weight_{skill}",
                            help=f"Weight for {skill}"
                        )
                    
                    with col_remove:
                        if st.button("×", key=f"remove_{skill}", help=f"Remove {skill}"):
                            st.session_state.removed_skills.add(skill)
                            if skill in st.session_state.skills_weights:
                                del st.session_state.skills_weights[skill]
                            st.rerun()
            else:
                st.info("No technical skills available. Please go back and extract skills.")
        
        # Right sidebar - clean and minimal
        with col2:
            total_active_skills = len(active_skills)
            total_removed_skills = len(st.session_state.removed_skills)
            
            st.markdown("### Summary")
            if total_removed_skills > 0:
                st.metric("Active Skills", total_active_skills, delta=f"-{total_removed_skills}")
            else:
                st.metric("Active Skills", total_active_skills)
            
            # Current total weight
            current_total = sum(new_weights.values())
            if current_total == 100:
                st.success(f"✅ {current_total}%")
            elif current_total < 100:
                st.warning(f"⚠️ {current_total}%")
            else:
                st.error(f"❌ {current_total}%")
            
            # Simple progress bar
            progress_percentage = min(current_total / 100, 1.0)
            st.progress(progress_percentage)
            
            st.markdown("### Controls")
            
            # Equal weights
            if st.button("Equal Weights", help="Distribute equally"):
                if total_active_skills > 0:
                    equal_weight = round(100 / total_active_skills)
                    for skill in new_weights.keys():
                        st.session_state[f"weight_{skill}"] = equal_weight
                    # Adjust first skill for rounding
                    if new_weights:
                        first_skill = list(new_weights.keys())[0]
                        adjustment = 100 - (equal_weight * total_active_skills)
                        st.session_state[f"weight_{first_skill}"] = equal_weight + adjustment
                    st.rerun()
            
            # Auto-normalize
            if st.button("Normalize", disabled=current_total == 0, help="Auto-adjust to 100%"):
                if current_total > 0:
                    normalized_weights = {skill: round((weight / current_total) * 100) for skill, weight in new_weights.items()}
                    total_normalized = sum(normalized_weights.values())
                    if total_normalized != 100 and normalized_weights:
                        first_skill = list(normalized_weights.keys())[0]
                        normalized_weights[first_skill] += (100 - total_normalized)
                    
                    for skill, weight in normalized_weights.items():
                        st.session_state[f"weight_{skill}"] = weight
                    st.rerun()
            
            # Restore removed skills
            if st.session_state.removed_skills:
                st.markdown("### Restore")
                removed_list = list(st.session_state.removed_skills)
                skill_to_restore = st.selectbox("", ["Select skill..."] + removed_list, key="restore_selector")
                
                if st.button("Restore", disabled=skill_to_restore == "Select skill..."):
                    if skill_to_restore != "Select skill...":
                        st.session_state.removed_skills.discard(skill_to_restore)
                        # Add back with default weight
                        if total_active_skills > 0:
                            default_weight = max(1, round(100 / (total_active_skills + 1)))
                        else:
                            default_weight = 100
                        st.session_state.skills_weights[skill_to_restore] = default_weight
                        st.rerun()
        
        # Update session state
        st.session_state.skills_weights = new_weights
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = 'input'
                st.rerun()
        
        with col2:
            if current_total == 100:
                st.success("Ready!")
            else:
                st.warning("Adjust to 100%")
        
        with col3:
            proceed_disabled = current_total != 100 or total_active_skills == 0
            if st.button("Continue →", disabled=proceed_disabled):
                st.session_state.extracted_skills = list(new_weights.keys())
                st.session_state.workflow_step = 'upload'
                st.rerun()

    # Step 3: Resume upload and analysis
    elif st.session_state.workflow_step == 'upload':
        st.subheader("Upload Resumes for Analysis")
        
        # Display current configuration
        with st.expander("📋 Configuration Summary", expanded=False):
            st.write(f"**🎯 Job Title:** {st.session_state.job_title}")
            
            # Get removed skills for display
            removed_skills = st.session_state.get('removed_skills', set())
            active_skills_count = len([s for s in st.session_state.extracted_skills if s not in removed_skills])
            
            st.write(f"**📊 Active Technical Skills:** {active_skills_count}")
            if removed_skills:
                st.write(f"**🗑️ Excluded Skills:** {len(removed_skills)}")
            
            st.write(f"**⚖️ Skill Weights:**")
            
            skills_categories = st.session_state.get('skills_categories', {})
            
            # Display Hard Skills (excluding removed ones)
            active_hard_skills = [skill for skill in skills_categories.get("hard_skills", []) 
                                if skill not in removed_skills]
            if active_hard_skills:
                for skill in active_hard_skills:
                    weight = st.session_state.skills_weights.get(skill, 0)
                    st.write(f"• {skill}: {weight}%")
            
            # Show removed skills if any
            if removed_skills:
                st.write("**🗑️ Excluded from Analysis:**")
                for skill in removed_skills:
                    st.write(f"• ~~{skill}~~")
            
            # Total weight validation
            total_weight = sum(st.session_state.skills_weights.get(skill, 0) 
                             for skill in st.session_state.skills_weights.keys() 
                             if skill not in removed_skills)
            
            if total_weight == 100:
                st.success(f"✅ Total: {total_weight}%")
            else:
                st.error(f"❌ Total: {total_weight}%")
        
        # File upload
        resume_pdfs = st.file_uploader("Upload your resume PDFs", type=["pdf"], accept_multiple_files=True)
        
        # Async analysis function
        async def run_analysis_async():
            """Run the async analysis process with custom barem."""
            # Use the custom barem from session state
            barem = st.session_state.get('custom_barem', {})
            job_title = st.session_state.get('job_title', '')
            job_description = st.session_state.get('job_description', '')
            
            # Container for displaying progress
            progress_container = st.container()
            progress_bar = st.progress(0)

            # Placeholder for the comparison table
            table_placeholder = st.empty()

            # Progress callback function
            def update_progress(completed, total):
                progress_percent = int((completed / total) * 100)
                progress_bar.progress(progress_percent)
                progress_container.text(f"Processing resumes concurrently... ({completed}/{total})")

            # Run async analysis
            results_data = await analyze_all_resumes_async(
                resume_pdfs, job_title, job_description, barem, update_progress
            )

            # Complete progress bar
            progress_bar.progress(100)
            progress_container.text("Analysis complete!")

            # Display comparison table
            display_comparison_table(results_data, table_placeholder)

            # Display individual reports in expandable sections
            display_individual_reports(results_data)
        
        if resume_pdfs:
            st.write(f"Uploaded {len(resume_pdfs)} file(s)")
            
            if st.button("Analyze All Resumes"):
                # Create barem from custom weights
                barem = create_custom_barem(st.session_state.skills_weights)
                st.session_state['custom_barem'] = barem
                
                # Run the async analysis
                try:
                    asyncio.run(run_analysis_async())
                except Exception as e:
                    st.error(f"Error processing resumes: {str(e)}")
        
        # Navigation
        if st.button("← Back to Skill Weighting"):
            st.session_state.workflow_step = 'weighting'
            st.rerun()


if __name__ == "__main__":
    run()
