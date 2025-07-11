#!/usr/bin/env python
"""
Streamlit web interface for the Resume Analysis CrewAI project.
This file contains only UI logic, with business logic handled by main.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
import requests
import json
from typing import List, Dict, Any
from urllib.parse import urlparse
import re

# Import the main analysis engine
from main import ResumeAnalysisEngine, run_batch_analysis
from utils.file_handler import FileHandler

# Configure Streamlit page
st.set_page_config(
    page_title="VEO Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .candidate-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .score-high { color: #4CAF50; font-weight: bold; }
    .score-medium { color: #ff9800; font-weight: bold; }
    .score-low { color: #f44336; font-weight: bold; }
    .recommendation-strong { color: #2e7d32; font-weight: bold; }
    .recommendation-recommended { color: #4CAF50; font-weight: bold; }
    .recommendation-conditional { color: #ff9800; font-weight: bold; }
    .recommendation-not { color: #f44336; font-weight: bold; }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .executive-summary {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .critical-concern {
        background: #ffebee;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #f44336;
        margin: 0.25rem 0;
    }
    .strength-item {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #4CAF50;
        margin: 0.25rem 0;
    }
    .gap-item {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #ff9800;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application for resume analysis."""
    
    def __init__(self):
        self.file_handler = FileHandler(Path("output"))
    
    def _fetch_job_from_recruitee(self, api_url: str) -> Dict[str, Any]:
        """
        Fetch job data from Recruitee API via FastAPI backend.
        
        Args:
            api_url: Recruitee API URL
            
        Returns:
            Dictionary containing job data or error information
        """
        try:
            backend_url = os.getenv('FASTAPI_BACKEND_URL', 'http://localhost:8000')
            auth_token = os.getenv('RECRUITEE_API_TOKEN')
            
            if not auth_token:
                return {
                    'success': False,
                    'error': 'RECRUITEE_API_TOKEN environment variable not set!'
                }
            # Make request to FastAPI backend
            params = {
                'api_url': api_url,
                'token': auth_token
            }
            
            response = requests.get(
                f"{backend_url}/job/fetch",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'description_html': result.get('description_html', ''),
                    'debug_info': result.get('debug_info', {})
                }
            else:
                # Handle error response
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', f'Backend returned status {response.status_code}')
                except:
                    error_msg = f'Backend returned status {response.status_code}: {response.text}'
                
                return {
                    'success': False,
                    'error': f'Backend API error: {error_msg}'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Could not connect to backend API. Please make sure the FastAPI server is running on localhost:8000'
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request to backend API timed out'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request to backend API failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def _validate_recruitee_url(self, url: str) -> bool:
        """
        Validate if the URL matches Recruitee API format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid Recruitee API URL, False otherwise
        """
        pattern = r'^https://api\.recruitee\.com/c/\d+/offers/\d+$'
        return bool(re.match(pattern, url))
    
    def _clean_html_description(self, html_content: str) -> str:
        """
        Clean HTML content to extract plain text description.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Clean text description
        """
        try:
            # Simple HTML tag removal (you might want to use BeautifulSoup for more complex HTML)
            import re
            
            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', '', html_content)
            
            # Replace HTML entities
            html_entities = {
                '&nbsp;': ' ',
                '&amp;': '&',
                '&lt;': '<',
                '&gt;': '>',
                '&quot;': '"',
                '&#39;': "'",
                '&apos;': "'"
            }
            
            for entity, replacement in html_entities.items():
                clean_text = clean_text.replace(entity, replacement)
            
            # Clean up whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = clean_text.strip()
            
            return clean_text
            
        except Exception as e:
            st.warning(f"Error cleaning HTML description: {str(e)}")
            return html_content

    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
        st.title("🤖 AI Resume Analyzer")
        st.markdown("*Powered by CrewAI - Intelligent Resume Analysis*")
        # Sidebar navigation
        self._render_sidebar()
        # Only render the home page
        self._render_home_page()
    
    def _render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("📋 Navigation")
        # Only Home page navigation
        if st.sidebar.button('🏠 Home', key='nav_Home'):
            st.session_state.page = 'Home'
            st.rerun()
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Status")
        required_vars = ['GEMINI_API_KEY']
        optional_vars = ['RECRUITEE_API_TOKEN']
        all_good = True
        for var in required_vars:
            if os.getenv(var):
                st.sidebar.success(f"✅ {var}")
            else:
                st.sidebar.error(f"❌ {var}")
                all_good = False
        for var in optional_vars:
            if os.getenv(var):
                st.sidebar.success(f"✅ {var} (Custom)")
            else:
                st.sidebar.info(f"ℹ️ {var} (Default)")
        if all_good:
            st.sidebar.success("🚀 System Ready")
        else:
            st.sidebar.error("⚠️ Configuration Issues")
    
    def _render_home_page(self):
        """Render the home page with resume upload and analysis."""
        st.header("📄 Resume Analysis")
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 Upload Resumes")
            resume_files = st.file_uploader(
                "Select resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more PDF resumes for analysis"
            )
            
            if resume_files:
                st.success(f"📊 {len(resume_files)} resume(s) uploaded")
                
                # Display uploaded files
                for i, file in enumerate(resume_files, 1):
                    st.write(f"{i}. {file.name}")
        
        with col2:
            st.subheader("📊 Upload Summary")
            if resume_files:
                st.write("✅ Ready for analysis")
                st.write(f"**Files:** {len(resume_files)} resume(s)")
            else:
                st.write("📁 No files uploaded yet")
                st.write("Upload PDF resumes to begin")
        
        # Job information section
        st.subheader("📋 Job Information from Recruitee")
        
        # API URL input
        api_url = st.text_input(
            "Recruitee API URL",
            placeholder="https://api.recruitee.com/c/105443/offers/2199294",
            help="Enter the Recruitee API URL for the job posting"
        )
        
        # Initialize session state for job data
        if 'job_data' not in st.session_state:
            st.session_state.job_data = None
        if 'job_title' not in st.session_state:
            st.session_state.job_title = ""
        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""
        
        # Fetch job data button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Fetch Job Data", type="secondary"):
                if api_url:
                    with st.spinner("Fetching job data from Recruitee..."):
                        job_result = self._fetch_job_from_recruitee(api_url)
                        
                        if job_result['success']:
                            st.session_state.job_data = job_result
                            st.session_state.job_title = job_result['title']
                            st.session_state.job_description = job_result['description']
                            st.success("✅ Job data fetched successfully!")
                        else:
                            st.error(f"❌ Failed to fetch job data: {job_result['error']}")
                            st.session_state.job_data = None
                else:
                    st.warning("Please enter a Recruitee API URL")
        
        with col2:
            if st.button("🔧 Test API", type="secondary"):
                if api_url:
                    with st.spinner("Testing API connection..."):
                        self._test_api_connection(api_url)
                else:
                    st.warning("Please enter a Recruitee API URL")
        
        with col3:
            if st.session_state.job_data:
                st.info(f"📋 Job loaded: {st.session_state.job_title}")
        
        # Display job information if fetched
        if st.session_state.job_data:
            st.markdown("---")
            
            # Job details display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📋 Job Details")
                st.markdown(f"**Title:** {st.session_state.job_title}")
                
                # Show description preview
                description_preview = st.session_state.job_description[:300] + "..." if len(st.session_state.job_description) > 300 else st.session_state.job_description
                st.markdown(f"**Description Preview:** {description_preview}")
                
                # Option to show full description
                if st.checkbox("Show Full Description"):
                    st.text_area(
                        "Full Job Description",
                        value=st.session_state.job_description,
                        height=200,
                        disabled=True
                    )
            
            with col2:
                st.markdown("### ⚙️ Options")
                
                # Option to edit job data
                if st.checkbox("Edit Job Data"):
                    st.session_state.job_title = st.text_input(
                        "Edit Job Title",
                        value=st.session_state.job_title,
                        key="edit_title"
                    )
                    
                    st.session_state.job_description = st.text_area(
                        "Edit Job Description",
                        value=st.session_state.job_description,
                        height=150,
                        key="edit_description"
                    )
                
                # Clear job data button
                if st.button("🗑️ Clear Job Data"):
                    st.session_state.job_data = None
                    st.session_state.job_title = ""
                    st.session_state.job_description = ""
                    st.rerun()
        
        else:
            # Fallback to manual input if no job data fetched
            st.markdown("---")
            st.markdown("### ✏️ Manual Job Entry (Optional)")
            st.info("💡 You can also enter job details manually if the API fetch didn't work")
            
            manual_job_title = st.text_input(
                "Job Title (Manual)",
                placeholder="e.g., Senior Software Engineer",
                help="Enter the position title manually"
            )
            
            manual_job_description = st.text_area(
                "Job Description (Manual)",
                placeholder="Paste the complete job description here...",
                height=200,
                help="Provide detailed job requirements, responsibilities, and qualifications"
            )
            
            if manual_job_title and manual_job_description:
                st.session_state.job_title = manual_job_title
                st.session_state.job_description = manual_job_description
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            self._run_analysis(resume_files, st.session_state.job_title, st.session_state.job_description)
    
    # HR Dashboard section removed
    
    def _run_analysis(self, resume_files, job_title, job_description):
        """Run the resume analysis process."""
        # For testing: allow manual loading of report.json if no resumes are uploaded
        if not resume_files:
            st.warning("No resumes uploaded. Loading sample report.json for demonstration.")
            try:
                with open('report.json', 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                st.info("Loaded report.json data:")
                st.json(report_data)
                self._display_results([report_data])
                self._update_session_state([report_data])
            except Exception as e:
                st.error(f"Could not load report.json: {str(e)}")
            return
        if not job_title or not job_description:
            st.error("Please provide job title and description by fetching from Recruitee API or entering manually.")
            return
        # ...existing code for real analysis...
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            engine = ResumeAnalysisEngine()
            if not engine.validate_requirements():
                st.error("System requirements not met. Please check environment variables.")
                return
            resume_file_list = []
            total_files = len(resume_files)
            for i, uploaded_file in enumerate(resume_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    candidate_name = Path(uploaded_file.name).stem
                    resume_file_list.append((tmp_file.name, candidate_name))
            status_text.text("Starting analysis...")
            results = engine.analyze_multiple_resumes(
                resume_file_list,
                job_title,
                job_description,
                progress_callback=lambda idx, total, candidate: self._update_progress(progress_bar, status_text, idx, total, candidate)
            )
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            self._display_results(results)
            self._update_session_state(results)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            for file_path, _ in resume_file_list:
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def _run_batch_analysis(self, batch_files, job_title, job_description):
        """Run batch analysis for multiple resumes."""
        # Convert uploaded files to format expected by run_batch_analysis
        resume_file_list = []
        
        for uploaded_file in batch_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                candidate_name = Path(uploaded_file.name).stem
                resume_file_list.append((tmp_file.name, candidate_name))
        
        # Run batch analysis
        with st.spinner("Running batch analysis..."):
            results = run_batch_analysis(resume_file_list, job_title, job_description)
        
        # Display results
        self._display_results(results)
        
        # Update session state
        self._update_session_state(results)
        
        # Clean up temporary files
        for file_path, _ in resume_file_list:
            try:
                os.unlink(file_path)
            except:
                pass
    
    def _display_results(self, results):
        """Display analysis results in a formatted way using the new JSON structure."""
        if not results:
            st.warning("No results to display.")
            return
        
        st.success(f"✅ Analysis completed for {len(results)} candidates")
        
        # Results summary
        st.subheader("📊 Results Summary")
        
        # Create results dataframe (plain text, no HTML)
        df_data = []
        for result in results:
            executive_summary = result.get('executive_summary', {})
            candidate_profile = result.get('candidate_profile', {})
            score = executive_summary.get('overall_score', 0)
            recommendation = executive_summary.get('overall_recommendation', 'Unknown')
            confidence = executive_summary.get('confidence_level', 'Unknown')
            # Use plain text for table
            df_data.append({
                'Candidate': executive_summary.get('candidate_name', 'Unknown'),
                'Score': f"{score:.1f}/10" if isinstance(score, (int, float)) else str(score),
                'Recommendation': str(recommendation),
                'Confidence': str(confidence),
                'Experience': candidate_profile.get('total_experience', 'Unknown'),
                'Status': 'Success' if result.get('valid', True) else 'Failed'
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for i, result in enumerate(results, 1):
            # Extract key information from new structure
            executive_summary = result.get('executive_summary', {})
            candidate_profile = result.get('candidate_profile', {})
            strengths_data = result.get('strengths_and_differentiators', {})
            gaps_data = result.get('gaps_and_risk_assessment', {})
            
            candidate_name = executive_summary.get('candidate_name', 'Unknown')
            overall_score = executive_summary.get('overall_score', 0)
            
            with st.expander(f"#{i}: {candidate_name} - {overall_score:.1f}/10"):
                if result.get('valid', True):
                    # Executive Summary
                    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Executive Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Overall Score:** {self._format_score_display(overall_score)}", unsafe_allow_html=True)
                        st.markdown(f"**Recommendation:** {self._format_recommendation_display(executive_summary.get('overall_recommendation', 'Unknown'))}", unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {self._format_confidence_display(executive_summary.get('confidence_level', 'Unknown'))}", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**Position:** {executive_summary.get('position_applied', 'Unknown')}")
                        st.markdown(f"**Experience:** {candidate_profile.get('total_experience', 'Unknown')}")
                        st.markdown(f"**Career Level:** {candidate_profile.get('career_level', 'Unknown')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Key Decision Factors
                    decision_factors = executive_summary.get('key_decision_factors', [])
                    if decision_factors:
                        st.markdown("**Key Decision Factors:**")
                        for factor in decision_factors:
                            st.markdown(f"- {factor}")
                    
                    # Critical Concerns
                    critical_concerns = executive_summary.get('critical_concerns', [])
                    if critical_concerns:
                        st.markdown("**Critical Concerns:**")
                        for concern in critical_concerns:
                            st.markdown(f'<div class="critical-concern">⚠️ {concern}</div>', unsafe_allow_html=True)
                    
                    # Candidate Profile
                    st.markdown("### 👤 Candidate Profile")
                    professional_summary = candidate_profile.get('professional_summary', '')
                    if professional_summary:
                        st.markdown(f"**Professional Summary:** {professional_summary}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Career Trajectory:** {candidate_profile.get('career_trajectory', 'Unknown')}")
                        st.markdown(f"**Relevant Experience:** {candidate_profile.get('relevant_experience', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Education Level:** {candidate_profile.get('education_level', 'Unknown')}")
                        st.markdown(f"**Current Status:** {candidate_profile.get('current_status', 'Unknown')}")
                    
                    # Strengths and Differentiators
                    st.markdown("### ✅ Strengths and Differentiators")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Core Strengths:**")
                        core_strengths = strengths_data.get('core_strengths', [])
                        if core_strengths:
                            for strength in core_strengths:
                                if isinstance(strength, dict):
                                    strength_text = strength.get('strength', 'Unknown')
                                    st.markdown(f'<div class="strength-item">✅ {strength_text}</div>', unsafe_allow_html=True)
                                    if strength.get('quantified_impact'):
                                        st.markdown(f"  *Impact: {strength.get('quantified_impact')}*")
                                else:
                                    st.markdown(f'<div class="strength-item">✅ {strength}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No strengths found.")
                    with col2:
                        st.markdown("**Unique Value Propositions:**")
                        unique_values = strengths_data.get('unique_value_propositions', [])
                        if unique_values:
                            for value in unique_values:
                                if isinstance(value, dict):
                                    value_text = value.get('value_proposition', 'Unknown')
                                    st.markdown(f'<div class="strength-item">🌟 {value_text}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="strength-item">🌟 {value}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No unique value propositions found.")
                    
                    # Gaps and Risk Assessment
                    st.markdown("### ❌ Gaps and Risk Assessment")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Critical Gaps:**")
                        critical_gaps = gaps_data.get('critical_gaps', [])
                        if critical_gaps:
                            for gap in critical_gaps:
                                if isinstance(gap, dict):
                                    gap_text = gap.get('gap', 'Unknown')
                                    st.markdown(f'<div class="gap-item">⚠️ {gap_text}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="gap-item">⚠️ {gap}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No critical gaps found.")
                    with col2:
                        st.markdown("**Risk Factors:**")
                        risk_factors = gaps_data.get('performance_risk_factors', [])
                        if risk_factors:
                            for risk in risk_factors:
                                if isinstance(risk, dict):
                                    risk_text = risk.get('risk', 'Unknown')
                                    st.markdown(f'<div class="gap-item">🔴 {risk_text}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="gap-item">🔴 {risk}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No risk factors found.")
                    
                    # Business Impact Assessment
                    business_impact = result.get('business_impact_assessment', {})
                    if business_impact:
                        st.markdown("### 💼 Business Impact Assessment")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Immediate Impact:** {business_impact.get('immediate_impact_potential', 'Unknown')}")
                            st.markdown(f"**Ramp-up Time:** {business_impact.get('ramp_up_time_estimate', 'Unknown')}")
                        
                        with col2:
                            st.markdown(f"**Long-term Value:** {business_impact.get('long_term_value_creation', 'Unknown')}")
                            st.markdown(f"**Team Impact:** {business_impact.get('team_dynamic_impact', 'Unknown')}")
                    
                    # Competitive Analysis
                    competitive_analysis = result.get('comparative_analysis', {})
                    if competitive_analysis:
                        st.markdown("### 🏆 Competitive Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Market Position:** {competitive_analysis.get('market_positioning', 'Unknown')}")
                            st.markdown(f"**Competitive Advantage:** {competitive_analysis.get('competitive_advantage', 'Unknown')}")
                        
                        with col2:
                            st.markdown(f"**Salary Alignment:** {competitive_analysis.get('salary_market_alignment', 'Unknown')}")
                            
                        scarcity_factors = competitive_analysis.get('market_scarcity_factors', [])
                        if scarcity_factors:
                            st.markdown("**Market Scarcity Factors:**")
                            for factor in scarcity_factors:
                                st.markdown(f"- {factor}")
                    
                    # Job Requirements Match
                    job_requirements = result.get('job_requirements_analysis', {})
                    if job_requirements:
                        st.markdown("### 📋 Job Requirements Match")
                        
                        satisfaction_score = job_requirements.get('requirements_satisfaction_score', 0)
                        if satisfaction_score:
                            st.markdown(f"**Requirements Satisfaction:** {satisfaction_score}%")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            missing_reqs = job_requirements.get('critical_missing_requirements', [])
                            if missing_reqs:
                                st.markdown("**Missing Requirements:**")
                                for req in missing_reqs:
                                    st.markdown(f'<div class="gap-item">❌ {req}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            exceeds_in = job_requirements.get('exceeds_expectations_in', [])
                            if exceeds_in:
                                st.markdown("**Exceeds Expectations In:**")
                                for area in exceeds_in:
                                    st.markdown(f'<div class="strength-item">🎯 {area}</div>', unsafe_allow_html=True)
                    
                    # Detailed Scoring Analysis
                    scoring_analysis = result.get('detailed_scoring_analysis', {})
                    if scoring_analysis:
                        st.markdown("### 📊 Detailed Scoring")
                        
                        score_distribution = scoring_analysis.get('score_distribution', {})
                        if score_distribution:
                            # Create columns for score distribution
                            num_cols = min(len(score_distribution), 3)  # Max 3 columns for better display
                            cols = st.columns(num_cols)
                            
                            for idx, (criterion, score) in enumerate(score_distribution.items()):
                                with cols[idx % num_cols]:
                                    criterion_name = criterion.replace('_', ' ').title()
                                    score_color = self._get_score_color_class(score)
                                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown(f"**{criterion_name}**")
                                    st.markdown(f'<span class="{score_color}">{score:.1f}/10</span>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            total_weighted = scoring_analysis.get('total_weighted_score', 0)
                            st.markdown(f"**Total Weighted Score:** {self._format_score_display(total_weighted)}", unsafe_allow_html=True)
                        
                        with col2:
                            percentile = scoring_analysis.get('percentile_ranking', 'Unknown')
                            st.markdown(f"**Percentile Ranking:** {percentile}")
                        
                        score_reliability = scoring_analysis.get('score_reliability', '')
                        if score_reliability:
                            st.markdown(f"**Score Reliability:** {score_reliability}")
                    
                    # Final Recommendation
                    final_recommendation = result.get('final_recommendation', {})
                    if final_recommendation:
                        st.markdown("### 📋 Final Recommendation")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Status:** {final_recommendation.get('status', 'Unknown')}")
                            st.markdown(f"**Primary Reason:** {final_recommendation.get('primary_reason', 'Unknown')}")
                        
                        with col2:
                            st.markdown(f"**Confidence Level:** {final_recommendation.get('confidence_level', 'Unknown')}")
                            
                        supporting_factors = final_recommendation.get('supporting_factors', [])
                        if supporting_factors:
                            st.markdown("**Supporting Factors:**")
                            for factor in supporting_factors:
                                st.markdown(f"- {factor}")
                        
                        conditions = final_recommendation.get('conditions_if_applicable', [])
                        if conditions:
                            st.markdown("**Conditions:**")
                            for condition in conditions:
                                st.markdown(f"- {condition}")
                        
                        next_steps = final_recommendation.get('next_steps', [])
                        if next_steps:
                            st.markdown("**Next Steps:**")
                            for step in next_steps:
                                st.markdown(f"- {step}")
                    
                    # Full JSON Report (expandable)
                    if st.checkbox(f"Show Full JSON Report for {candidate_name}", key=f"json_{i}"):
                        st.json(result)
                        
                else:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
    
    def _update_session_state(self, results):
        """Update session state with analysis results using the new JSON structure."""
        if not results:
            return
        
        # Update metrics
        valid_results = [r for r in results if r.get('valid', True)]
        
        st.session_state.total_analyses = st.session_state.get('total_analyses', 0) + len(results)
        st.session_state.success_rate = (len(valid_results) / len(results)) * 100 if results else 0
        
        # Calculate average score from new structure
        total_score = 0
        score_count = 0
        for result in valid_results:
            executive_summary = result.get('executive_summary', {})
            score = executive_summary.get('overall_score', 0)
            if score > 0:
                total_score += score
                score_count += 1
        
        st.session_state.avg_score = total_score / score_count if score_count > 0 else 0
    
    def _test_api_connection(self, api_url: str) -> None:
        """
        Test API connection through FastAPI backend and display results.
        
        Args:
            api_url: Recruitee API URL to test
        """
        try:
            # Get FastAPI backend URL from environment or use default
            backend_url = os.getenv('FASTAPI_BACKEND_URL', 'http://localhost:8000')
            
            # Get token from environment or use default
            auth_token = os.getenv('RECRUITEE_API_TOKEN', 'bUEyUjhlV3MzSFQzVXJKanJyeWpQUT09')
            
            # Make request to FastAPI backend test endpoint
            params = {
                'api_url': api_url,
                'token': auth_token
            }
            
            response = requests.get(
                f"{backend_url}/job/test",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    st.success("✅ API Connection Test Successful!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Status Code:** {result.get('status_code', 'Unknown')}")
                        st.markdown(f"**Auth Method:** {result.get('auth_method', 'Unknown')}")
                    
                    with col2:
                        if result.get('response_preview'):
                            st.markdown("**Response Preview:**")
                            st.code(result['response_preview'][:200] + "..." if len(result['response_preview']) > 200 else result['response_preview'])
                    
                    # Show debug info if available
                    debug_info = result.get('debug_info', {})
                    if debug_info:
                        if st.checkbox("Show Debug Information"):
                            st.json(debug_info)
                else:
                    st.error("❌ API Connection Test Failed")
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                    
                    # Show debug info if available
                    debug_info = result.get('debug_info', {})
                    if debug_info:
                        if st.checkbox("Show Debug Information"):
                            st.json(debug_info)
            else:
                st.error("❌ Backend API Test Failed")
                try:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', f'Backend returned status {response.status_code}')}")
                except:
                    st.error(f"Backend returned status {response.status_code}: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to backend API")
            st.error("Please make sure the FastAPI server is running on localhost:8000")
            st.info("💡 To start the backend server, run: `python src/api/recruitee_proxy.py`")
        except requests.exceptions.Timeout:
            st.error("❌ Request to backend API timed out")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request to backend API failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

    def _update_progress(self, progress_bar, status_text, current_index, total_count, current_candidate):
        """Update progress bar and status text."""
        progress = (current_index + 1) / total_count
        progress_bar.progress(progress)
        status_text.text(f"Analyzing candidate {current_index + 1} of {total_count}: {current_candidate}")

    def _format_score_display(self, score):
        """Format score with appropriate color."""
        if score >= 8:
            return f'<span class="score-high">{score:.1f}/10</span>'
        elif score >= 6:
            return f'<span class="score-medium">{score:.1f}/10</span>'
        else:
            return f'<span class="score-low">{score:.1f}/10</span>'

    def _format_recommendation_display(self, recommendation):
        """Format recommendation with appropriate color."""
        if recommendation.lower() in ['strongly recommended', 'strong hire']:
            return f'<span class="recommendation-strong">{recommendation}</span>'
        elif recommendation.lower() in ['recommended', 'hire']:
            return f'<span class="recommendation-recommended">{recommendation}</span>'
        elif recommendation.lower() in ['conditional', 'conditional hire']:
            return f'<span class="recommendation-conditional">{recommendation}</span>'
        else:
            return f'<span class="recommendation-not">{recommendation}</span>'

    def _format_confidence_display(self, confidence):
        """Format confidence level with appropriate color."""
        if confidence.lower() == 'high':
            return f'<span class="confidence-high">{confidence}</span>'
        elif confidence.lower() == 'medium':
            return f'<span class="confidence-medium">{confidence}</span>'
        else:
            return f'<span class="confidence-low">{confidence}</span>'

    def _get_score_color_class(self, score):
        """Get appropriate CSS class for score color."""
        if score >= 8:
            return 'score-high'
        elif score >= 6:
            return 'score-medium'
        else:
            return 'score-low'


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()