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
from typing import List, Dict, Any

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
        
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 style="color:#4CAF50;">VEO</h1>', unsafe_allow_html=True)
        st.title("🤖 AI Resume Analyzer")
        st.markdown("*Powered by CrewAI - Intelligent Resume Analysis*")
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content area
        page = st.session_state.get('page', 'Home')
        
        if page == 'Home':
            self._render_home_page()
        elif page == 'HR':
            self._render_hr_page()
        elif page == 'About':
            self._render_about_page()
    
    def _render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("📋 Navigation")
        
        pages = {
            'Home': '🏠 Home',
            'HR': '👥 HR Dashboard',
            'About': 'ℹ️ About'
        }
        
        for key, display_name in pages.items():
            if st.sidebar.button(display_name, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Status")
        
        # Check environment variables
        required_vars = ['GEMINI_API_KEY']
        all_good = True
        
        for var in required_vars:
            if os.getenv(var):
                st.sidebar.success(f"✅ {var}")
            else:
                st.sidebar.error(f"❌ {var}")
                all_good = False
        
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
        st.subheader("💼 Job Requirements")
        
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the position title"
        )
        
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste the complete job description here...",
            height=200,
            help="Provide detailed job requirements, responsibilities, and qualifications"
        )
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            self._run_analysis(resume_files, job_title, job_description)
    
    def _render_hr_page(self):
        """Render the HR dashboard page."""
        st.header("👥 HR Dashboard")
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Analyses", st.session_state.get('total_analyses', 0))
        
        with col2:
            st.metric("📈 Success Rate", f"{st.session_state.get('success_rate', 0):.1f}%")
        
        with col3:
            st.metric("⭐ Avg Score", f"{st.session_state.get('avg_score', 0):.1f}/10")
        
        with col4:
            st.metric("🕒 Last Analysis", st.session_state.get('last_analysis', 'None'))
        
        # Recent analyses section
        st.subheader("📋 Recent Analyses")
        
        if 'recent_analyses' in st.session_state and st.session_state.recent_analyses:
            # Display recent analyses in a table
            df = pd.DataFrame(st.session_state.recent_analyses)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent analyses available. Start analyzing resumes to see results here.")
        
        # Batch analysis section
        st.subheader("📦 Batch Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Upload multiple resumes for batch processing:")
            batch_files = st.file_uploader(
                "Select multiple resume PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="batch_upload"
            )
        
        with col2:
            if batch_files:
                st.success(f"📊 {len(batch_files)} files ready for batch analysis")
                
                batch_job_title = st.text_input("Batch Job Title", key="batch_job_title")
                batch_job_description = st.text_area(
                    "Batch Job Description", 
                    height=100, 
                    key="batch_job_description"
                )
                
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    if batch_job_title and batch_job_description:
                        self._run_batch_analysis(batch_files, batch_job_title, batch_job_description)
                    else:
                        st.error("Please provide job title and description for batch analysis.")
    
    def _render_about_page(self):
        """Render the about page with system information."""
        st.header("ℹ️ About VEO Resume Analyzer")
        
        # System overview
        st.markdown("""
        ### 🤖 AI-Powered Resume Analysis
        
        **VEO Resume Analyzer** leverages CrewAI and advanced LLMs to automate and standardize 
        resume screening for HR teams and recruiters with comprehensive JSON-based reporting.
        
        #### Key Features:
        - **📊 Comprehensive Scoring**: Multi-dimensional evaluation with detailed scoring breakdown
        - **📋 Executive Summary**: Clear recommendations with confidence levels and decision factors
        - **🎯 Business Impact Assessment**: Immediate and long-term value analysis
        - **📈 Competitive Analysis**: Market positioning and salary alignment insights
        - **📄 Multi-format Support**: Analyze PDF resumes with advanced text extraction
        - **🔍 Detailed Reports**: Structured JSON reports with strengths, gaps, and recommendations
        - **� Batch Processing**: Handle multiple resumes simultaneously
        - **🔒 Privacy-First**: No data storage, all processing is temporary
        
        #### New JSON Structure Benefits:
        - **📊 Structured Data**: Machine-readable format for easy integration
        - **🎯 Executive Focus**: Clear executive summary for quick decision-making
        - **📈 Business Metrics**: Impact assessment and competitive analysis
        - **🔍 Quality Assurance**: Built-in validation and confidence scoring
        - **📋 Comprehensive Coverage**: Multi-dimensional candidate evaluation
        """)
        
        # Technical specifications
        with st.expander("🔧 Technical Specifications"):
            st.markdown("""
            - **AI Models**: Google Gemini, Azure OpenAI
            - **Framework**: CrewAI for multi-agent orchestration
            - **PDF Processing**: pypdf, PDFplumber for text extraction
            - **Interface**: Streamlit for web UI
            - **Output Format**: Structured JSON with comprehensive analysis
            - **Scoring System**: Multi-dimensional weighted scoring
            - **Security**: API-key based authentication, no data persistence
            """)
        
        # Usage instructions
        with st.expander("📚 How to Use"):
            st.markdown("""
            1. **Setup**: Ensure GEMINI_API_KEY is configured in environment
            2. **Upload**: Select one or more PDF resumes
            3. **Job Details**: Provide job title and description
            4. **Analysis**: Click 'Start Analysis' to begin processing
            5. **Results**: View comprehensive JSON-structured reports including:
               - Executive Summary with recommendations
               - Detailed scoring breakdown
               - Business impact assessment
               - Competitive analysis
               - Job requirements matching
            6. **Export**: Download structured reports for record-keeping
            """)
        
        # New features highlight
        with st.expander("🆕 New Features in JSON Format"):
            st.markdown("""
            - **🎯 Executive Summary**: Quick decision-making insights
            - **📊 Detailed Scoring**: Multi-dimensional evaluation metrics
            - **💼 Business Impact**: Immediate and long-term value assessment
            - **🏆 Competitive Analysis**: Market positioning and advantages
            - **📋 Requirements Matching**: Detailed job fit analysis
            - **🔍 Quality Assurance**: Confidence levels and data validation
            - **📈 Stakeholder Impact**: Analysis for different organizational levels
            """)
        
        # Contact and support
        st.markdown("""
        ---
        **Contact**: [Your Company](mailto:contact@yourcompany.com)  
        **Version**: 1.0.0  
        **Last Updated**: July 2025
        """)
    
    def _run_analysis(self, resume_files, job_title, job_description):
        """Run the resume analysis process."""
        if not resume_files:
            st.error("Please upload at least one resume.")
            return
        
        if not job_title or not job_description:
            st.error("Please provide job title and description.")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize analysis engine
            engine = ResumeAnalysisEngine()
            
            if not engine.validate_requirements():
                st.error("System requirements not met. Please check environment variables.")
                return
            
            # Prepare file list for batch processing
            resume_file_list = []
            total_files = len(resume_files)
            
            for i, uploaded_file in enumerate(resume_files):
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    candidate_name = Path(uploaded_file.name).stem
                    resume_file_list.append((tmp_file.name, candidate_name))
            
            # Update initial status
            status_text.text("Starting analysis...")
            
            # Run batch analysis
            results = engine.analyze_multiple_resumes(
                resume_file_list, 
                job_title, 
                job_description,
                progress_callback=lambda idx, total, candidate: self._update_progress(progress_bar, status_text, idx, total, candidate)
            )
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            # Display results
            self._display_results(results)
            
            # Update session state
            self._update_session_state(results)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            # Clean up temporary files
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
        
        # Create results dataframe
        df_data = []
        for result in results:
            # Extract data from new JSON structure
            executive_summary = result.get('executive_summary', {})
            candidate_profile = result.get('candidate_profile', {})
            
            score = executive_summary.get('overall_score', 0)
            recommendation = executive_summary.get('overall_recommendation', 'Unknown')
            confidence = executive_summary.get('confidence_level', 'Unknown')
            
            df_data.append({
                'Candidate': executive_summary.get('candidate_name', 'Unknown'),
                'Score': self._format_score_display(score),
                'Recommendation': self._format_recommendation_display(recommendation),
                'Confidence': self._format_confidence_display(confidence),
                'Experience': candidate_profile.get('total_experience', 'Unknown'),
                'Status': '✅ Success' if result.get('valid', True) else '❌ Failed'
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
                        for strength in core_strengths:
                            if isinstance(strength, dict):
                                strength_text = strength.get('strength', 'Unknown')
                                st.markdown(f'<div class="strength-item">✅ {strength_text}</div>', unsafe_allow_html=True)
                                if strength.get('quantified_impact'):
                                    st.markdown(f"  *Impact: {strength.get('quantified_impact')}*")
                            else:
                                st.markdown(f'<div class="strength-item">✅ {strength}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Unique Value Propositions:**")
                        unique_values = strengths_data.get('unique_value_propositions', [])
                        for value in unique_values:
                            if isinstance(value, dict):
                                value_text = value.get('value_proposition', 'Unknown')
                                st.markdown(f'<div class="strength-item">🌟 {value_text}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="strength-item">🌟 {value}</div>', unsafe_allow_html=True)
                    
                    # Gaps and Risk Assessment
                    st.markdown("### ❌ Gaps and Risk Assessment")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Critical Gaps:**")
                        critical_gaps = gaps_data.get('critical_gaps', [])
                        for gap in critical_gaps:
                            if isinstance(gap, dict):
                                gap_text = gap.get('gap', 'Unknown')
                                st.markdown(f'<div class="gap-item">⚠️ {gap_text}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="gap-item">⚠️ {gap}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Risk Factors:**")
                        risk_factors = gaps_data.get('performance_risk_factors', [])
                        for risk in risk_factors:
                            if isinstance(risk, dict):
                                risk_text = risk.get('risk', 'Unknown')
                                st.markdown(f'<div class="gap-item">🔴 {risk_text}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="gap-item">🔴 {risk}</div>', unsafe_allow_html=True)
                    
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
        st.session_state.last_analysis = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        # Update recent analyses
        recent_analyses = st.session_state.get('recent_analyses', [])
        
        for result in results:
            executive_summary = result.get('executive_summary', {})
            
            recent_analyses.append({
                'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'Candidate': executive_summary.get('candidate_name', 'Unknown'),
                'Score': f"{executive_summary.get('overall_score', 0):.1f}/10",
                'Recommendation': executive_summary.get('overall_recommendation', 'Unknown'),
                'Confidence': executive_summary.get('confidence_level', 'Unknown'),
                'Status': '✅ Success' if result.get('valid', True) else '❌ Failed'
            })
        
        # Keep only last 10 analyses
        st.session_state.recent_analyses = recent_analyses[-10:]
    
    def _update_progress(self, progress_bar, status_text, idx, total, candidate):
        """Update progress bar and status text during analysis."""
        if total > 0:
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {candidate} ({idx + 1}/{total})...")
        
    def _get_score_color_class(self, score: float) -> str:
        """Get CSS class for score color based on value."""
        if score >= 8.0:
            return "score-high"
        elif score >= 6.0:
            return "score-medium"
        else:
            return "score-low"
    
    def _get_recommendation_color_class(self, recommendation: str) -> str:
        """Get CSS class for recommendation color based on value."""
        recommendation_lower = recommendation.lower()
        if "strongly recommended" in recommendation_lower:
            return "recommendation-strong"
        elif "recommended" in recommendation_lower and "not" not in recommendation_lower:
            return "recommendation-recommended"
        elif "conditional" in recommendation_lower:
            return "recommendation-conditional"
        else:
            return "recommendation-not"
    
    def _get_confidence_color_class(self, confidence: str) -> str:
        """Get CSS class for confidence color based on value."""
        confidence_lower = confidence.lower()
        if confidence_lower == "high":
            return "confidence-high"
        elif confidence_lower == "medium":
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _format_score_display(self, score: float) -> str:
        """Format score for display with color coding."""
        color_class = self._get_score_color_class(score)
        return f'<span class="{color_class}">{score:.1f}/10</span>'
    
    def _format_recommendation_display(self, recommendation: str) -> str:
        """Format recommendation for display with color coding."""
        color_class = self._get_recommendation_color_class(recommendation)
        return f'<span class="{color_class}">{recommendation}</span>'
    
    def _format_confidence_display(self, confidence: str) -> str:
        """Format confidence for display with color coding."""
        color_class = self._get_confidence_color_class(confidence)
        return f'<span class="{color_class}">{confidence}</span>'


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()