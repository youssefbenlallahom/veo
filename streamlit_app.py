import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import sys
from datetime import datetime

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.resume.crew import Resume
    from src.resume.utils.barem_generator import BaremGenerator
    from src.resume.utils.report_parser import ReportParser
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Resume Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def load_report_json(report_path):
    """Load and parse the JSON report file."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading report: {e}")
        return None

def display_executive_summary(executive_summary):
    """Display the executive summary section."""
    st.markdown('<div class="section-header"><h2>🎯 Executive Summary</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall Score",
            value=f"{executive_summary.get('overall_score', 'Unknown')}/10"
        )
    
    with col2:
        recommendation = executive_summary.get('overall_recommendation', 'Unknown')
        st.metric(
            label="Recommendation",
            value=recommendation
        )
    
    with col3:
        confidence = executive_summary.get('confidence_level', 'Unknown')
        st.metric(
            label="Confidence",
            value=confidence
        )
    
    # Additional metrics
    col4, col5 = st.columns(2)
    
    with col4:
        st.metric(
            label="Position Applied",
            value=executive_summary.get('position_applied', 'Unknown')
        )
    
    with col5:
        st.metric(
            label="Candidate",
            value=executive_summary.get('candidate_name', 'Unknown')
        )

def display_key_decision_factors(executive_summary):
    """Display key decision factors."""
    st.markdown('<div class="section-header"><h3>✅ Key Decision Factors</h3></div>', unsafe_allow_html=True)
    
    key_factors = executive_summary.get('key_decision_factors', [])
    if key_factors:
        for factor in key_factors:
            st.markdown(f"• {factor}")
    else:
        st.info("No key decision factors found in the report.")

def display_critical_concerns(executive_summary):
    """Display critical concerns."""
    st.markdown('<div class="section-header"><h3>⚠️ Critical Concerns</h3></div>', unsafe_allow_html=True)
    
    concerns = executive_summary.get('critical_concerns', [])
    if concerns:
        for concern in concerns:
            st.markdown(f"• {concern}")
    else:
        st.info("No critical concerns found in the report.")

def display_candidate_profile(candidate_profile):
    """Display candidate profile information."""
    st.markdown('<div class="section-header"><h2>👤 Candidate Profile</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Education:**")
        st.write(candidate_profile.get('education_level', 'Unknown'))
        
        st.markdown("**Total Experience:**")
        st.write(candidate_profile.get('total_experience', 'Unknown'))
        
        st.markdown("**Current Status:**")
        st.write(candidate_profile.get('current_status', 'Unknown'))
    
    with col2:
        st.markdown("**Career Trajectory:**")
        st.write(candidate_profile.get('career_trajectory', 'Unknown'))
        
        st.markdown("**Relevant Experience:**")
        st.write(candidate_profile.get('relevant_experience', 'Unknown'))
        
        st.markdown("**Industry Experience:**")
        industries = candidate_profile.get('industry_experience', [])
        if industries:
            st.write(", ".join(industries))
        else:
            st.write("Not specified")

def display_detailed_analysis(report_data):
    """Display detailed analysis sections."""
    with st.expander("📊 Detailed Scoring Analysis", expanded=False):
        scoring = report_data.get('detailed_scoring_analysis', {})
        if scoring:
            score_dist = scoring.get('score_distribution', {})
            if score_dist:
                st.subheader("Score Distribution")
                for category, score in score_dist.items():
                    st.metric(label=category.replace('_', ' ').title(), value=score)
            
            st.subheader("Scoring Methodology")
            st.write(scoring.get('scoring_methodology', 'Not available'))
        else:
            st.info("Detailed scoring analysis not available.")
    
    with st.expander("💪 Strengths and Differentiators", expanded=False):
        strengths = report_data.get('strengths_and_differentiators', {})
        if strengths:
            core_strengths = strengths.get('core_strengths', [])
            if core_strengths:
                st.subheader("Core Strengths")
                for strength in core_strengths:
                    if isinstance(strength, dict):
                        st.markdown(f"**{strength.get('strength', 'Unknown')}**")
                        st.write(strength.get('evidence', ''))
                        st.write(f"*Business Impact:* {strength.get('business_impact', '')}")
                        st.divider()
        else:
            st.info("Strengths analysis not available.")
    
    with st.expander("⚠️ Gaps and Risk Assessment", expanded=False):
        gaps = report_data.get('gaps_and_risk_assessment', {})
        if gaps:
            critical_gaps = gaps.get('critical_gaps', [])
            if critical_gaps:
                st.subheader("Critical Gaps")
                for gap in critical_gaps:
                    if isinstance(gap, dict):
                        st.markdown(f"**{gap.get('gap', 'Unknown')}** (Impact: {gap.get('impact_level', 'Unknown')})")
                        st.write(gap.get('severity_assessment', ''))
                        st.divider()
        else:
            st.info("Gaps and risk assessment not available.")

def analyze_resume(pdf_file, job_title, job_description):
    """Analyze the uploaded resume using the multi-agent system."""
    try:
        # Create a temporary file for the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize the Resume crew
        with st.spinner("Initializing analysis system..."):
            resume_crew = Resume(pdf_path=tmp_file_path)
        
        # Prepare inputs for the crew
        inputs = {
            'pdf': tmp_file_path,
            'job_title': job_title,
            'job_description': job_description,
            'current_year': str(datetime.now().year)
        }
        
        # Generate barem (scoring rubric) if needed
        with st.spinner("Generating scoring rubric..."):
            try:
                barem_generator = BaremGenerator(output_dir)
                barem = barem_generator.get_barem(job_title, job_description)
                if barem:
                    inputs['barem'] = barem
            except Exception as e:
                st.warning(f"Could not generate scoring rubric: {e}")
        
        # Execute the analysis
        with st.spinner("Analyzing resume with multi-agent system... This may take a few minutes."):
            result = resume_crew.crew().kickoff(inputs=inputs)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Look for the generated report.json file
        report_file = Path("report.json")
        if report_file.exists():
            return load_report_json(report_file)
        else:
            st.error("Report file was not generated. Please check the system logs.")
            return None
            
    except Exception as e:
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        st.error(f"Error during analysis: {e}")
        return None

def main():
    """Main Streamlit application."""
    st.markdown('<div class="main-header"><h1>📄 Resume Analysis System</h1><p>Powered by Multi-Agent AI</p></div>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Job Information")
        
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the position title you're hiring for"
        )
        
        job_description = st.text_area(
            "Job Description",
            placeholder="Enter the complete job description including requirements, responsibilities, and qualifications...",
            height=200,
            help="Provide detailed job requirements for accurate analysis"
        )
        
        st.header("📄 Resume Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=['pdf'],
            help="Upload the candidate's resume in PDF format"
        )
        
        analyze_button = st.button(
            "🔍 Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=not (job_title and job_description and uploaded_file)
        )
    
    # Main content area
    if analyze_button:
        if not job_title or not job_description or not uploaded_file:
            st.error("Please fill in all required fields: Job Title, Job Description, and upload a PDF resume.")
            return
        
        # Perform analysis
        report_data = analyze_resume(uploaded_file, job_title, job_description)
        
        if report_data:
            # Store report data in session state for persistence
            st.session_state.report_data = report_data
            st.success("✅ Analysis completed successfully!")
        else:
            st.error("❌ Analysis failed. Please try again.")
            return
    
    # Display results if available
    if 'report_data' in st.session_state:
        report_data = st.session_state.report_data
        
        # Executive Summary
        executive_summary = report_data.get('executive_summary', {})
        display_executive_summary(executive_summary)
        
        # Key Decision Factors
        display_key_decision_factors(executive_summary)
        
        # Critical Concerns  
        display_critical_concerns(executive_summary)
        
        # Candidate Profile
        candidate_profile = report_data.get('candidate_profile', {})
        display_candidate_profile(candidate_profile)
        
        # Additional Analysis (Expandable sections)
        st.markdown('<div class="section-header"><h2>📈 Detailed Analysis</h2></div>', unsafe_allow_html=True)
        display_detailed_analysis(report_data)
        
        # Download option
        st.markdown("---")
        st.subheader("📥 Download Report")
        
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="Download Full Report (JSON)",
            data=report_json,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    else:
        # Show placeholder content
        st.info("👆 Please fill in the job information and upload a resume to start the analysis.")
        
        # Show example/demo content
        st.markdown("---")
        st.subheader("🎯 What This System Does")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📄 Document Analysis**
            - Extracts all information from PDF resumes
            - Identifies skills, experience, and qualifications
            - Calculates career progression metrics
            """)
        
        with col2:
            st.markdown("""
            **🎯 Job Matching**
            - Compares candidate against job requirements
            - Provides detailed scoring and analysis
            - Identifies strengths and gaps
            """)
        
        with col3:
            st.markdown("""
            **📊 Comprehensive Reporting**
            - Executive summary with scores
            - Detailed analysis and recommendations
            - Exportable JSON reports
            """)

if __name__ == "__main__":
    main()
