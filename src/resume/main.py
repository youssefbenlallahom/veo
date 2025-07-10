#!/usr/bin/env python
"""
Main entry point for the Resume Analysis CrewAI project.
Following CrewAI best practices for clean separation of concerns.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from crew import Resume
from utils.file_handler import FileHandler
from utils.barem_generator import BaremGenerator
from utils.report_parser import ReportParser
from utils.pdf_validator import PDFValidator

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'resume_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResumeAnalysisEngine:
    """
    Main engine for orchestrating resume analysis using CrewAI.
    Follows CrewAI best practices for clean architecture.
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize the analysis engine."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "extracts").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize utilities
        self.file_handler = FileHandler(self.output_dir)
        self.barem_generator = BaremGenerator(self.output_dir)
        self.report_parser = ReportParser()
        self.pdf_validator = PDFValidator()
        
        logger.info(f"Resume Analysis Engine initialized with output directory: {self.output_dir}")
    
    def validate_requirements(self) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = ['GEMINI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True
    
    def generate_or_load_barem(self, job_title: str, job_description: str) -> Optional[Dict]:
        """Generate or load existing barem for the job."""
        try:
            return self.barem_generator.get_barem(job_title, job_description)
        except Exception as e:
            logger.error(f"Error generating barem: {e}")
            return None
    
    def analyze_single_resume(
        self, 
        resume_path: str, 
        job_title: str, 
        job_description: str,
        candidate_name: str,
        barem: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single resume using CrewAI.
        
        Args:
            resume_path: Path to the PDF resume
            job_title: Job title for the position
            job_description: Job description text
            candidate_name: Name of the candidate
            barem: Optional scoring rubric
            
        Returns:
            Dict containing analysis results
        """
        logger.info(f"Starting analysis for candidate: {candidate_name}")
        
        try:
            # Validate PDF
            is_valid, validation_message = self.pdf_validator.validate_pdf(resume_path)
            if not is_valid:
                logger.warning(f"PDF validation failed for {candidate_name}: {validation_message}")
                return {
                    "candidate_name": candidate_name,
                    "valid": False,
                    "error": validation_message,
                    "score": 0,
                    "recommendation": "Failed"
                }
            
            # Create Resume crew instance
            resume_crew = Resume(pdf_path=resume_path)
            
            # Prepare inputs for crew
            inputs = {
                'pdf': resume_path,
                'job_title': job_title,
                'job_description': job_description,
                'current_year': str(datetime.now().year)
            }
            
            if barem:
                inputs['barem'] = barem
            
            # Execute crew analysis
            logger.info(f"Executing crew analysis for {candidate_name}")
            result = resume_crew.crew().kickoff(inputs=inputs)
            
            # Save token usage
            self._save_token_usage(candidate_name, result.token_usage)
            
            # Generate unique report filename and move report
            report_filename = self._handle_report_file(candidate_name)
            
            # Parse the report
            analysis_result = self.report_parser.parse_report(report_filename, candidate_name)
            
            logger.info(f"Analysis completed for {candidate_name} - Score: {analysis_result.get('score', 0)}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing resume for {candidate_name}: {e}")
            return {
                "candidate_name": candidate_name,
                "valid": False,
                "error": str(e),
                "score": 0,
                "recommendation": "Error"
            }
    
    def analyze_multiple_resumes(
        self,
        resume_files: List[tuple],  # [(file_path, candidate_name), ...]
        job_title: str,
        job_description: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes in batch.
        
        Args:
            resume_files: List of tuples (file_path, candidate_name)
            job_title: Job title for the position
            job_description: Job description text
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(resume_files)} resumes")
        
        # Generate barem once for all resumes
        barem = self.generate_or_load_barem(job_title, job_description)
        if not barem:
            logger.error("Failed to generate barem, aborting batch analysis")
            return []
        
        results = []
        total_files = len(resume_files)
        
        for idx, (resume_path, candidate_name) in enumerate(resume_files):
            # Update progress
            if progress_callback:
                progress_callback(idx, total_files, candidate_name)
            
            # Analyze single resume
            result = self.analyze_single_resume(
                resume_path, job_title, job_description, candidate_name, barem
            )
            results.append(result)
            
            # Clean up temporary file if needed
            self.file_handler.cleanup_temp_file(resume_path)
        
        # Sort results by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Save batch results
        self._save_batch_results(results, job_title)
        
        logger.info(f"Batch analysis completed. Processed {len(results)} resumes")
        return results
    
    def _save_token_usage(self, candidate_name: str, token_usage: Any) -> None:
        """Save token usage information."""
        try:
            sanitized_name = self.file_handler.sanitize_filename(candidate_name)
            token_file = self.output_dir / "logs" / f"token_usage_{sanitized_name}.txt"
            
            with open(token_file, 'w', encoding='utf-8') as f:
                f.write(f"Token Usage for {candidate_name}:\n")
                f.write(f"{token_usage}\n")
        except Exception as e:
            logger.warning(f"Failed to save token usage for {candidate_name}: {e}")
    
    def _handle_report_file(self, candidate_name: str) -> str:
        """Handle moving and renaming the report file."""
        sanitized_name = self.file_handler.sanitize_filename(candidate_name)
        report_filename = self.output_dir / "reports" / f"report_{sanitized_name}.md"
        
        # Move report.md to unique filename
        if os.path.exists('report.md'):
            if report_filename.exists():
                report_filename.unlink()
            os.rename('report.md', report_filename)
        
        return str(report_filename)
    
    def _save_batch_results(self, results: List[Dict], job_title: str) -> None:
        """Save batch analysis results summary."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"batch_analysis_{timestamp}.md"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# Batch Analysis Results - {job_title}\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Summary\n\n")
                f.write(f"Total Candidates Analyzed: {len(results)}\n\n")
                f.write("## Rankings\n\n")
                
                for idx, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    recommendation = result.get('recommendation', 'Unknown')
                    f.write(f"{idx}. **{result['candidate_name']}** - Score: {score:.1f}/10 - {recommendation}\n")
                
            logger.info(f"Batch results saved to {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to save batch results: {e}")


def run_single_analysis(
    resume_path: str,
    job_title: str,
    job_description: str,
    candidate_name: str = None
) -> Dict[str, Any]:
    """
    Run analysis for a single resume.
    
    Args:
        resume_path: Path to the PDF resume
        job_title: Job title for the position
        job_description: Job description text
        candidate_name: Optional candidate name (defaults to filename)
        
    Returns:
        Analysis result dictionary
    """
    if not candidate_name:
        candidate_name = Path(resume_path).stem
    
    engine = ResumeAnalysisEngine()
    
    if not engine.validate_requirements():
        logger.error("Requirements validation failed")
        return {"error": "Missing required environment variables"}
    
    return engine.analyze_single_resume(
        resume_path, job_title, job_description, candidate_name
    )


def run_batch_analysis(
    resume_files: List[tuple],
    job_title: str,
    job_description: str
) -> List[Dict[str, Any]]:
    """
    Run batch analysis for multiple resumes.
    
    Args:
        resume_files: List of tuples (file_path, candidate_name)
        job_title: Job title for the position
        job_description: Job description text
        
    Returns:
        List of analysis results
    """
    engine = ResumeAnalysisEngine()
    
    if not engine.validate_requirements():
        logger.error("Requirements validation failed")
        return []
    
    return engine.analyze_multiple_resumes(resume_files, job_title, job_description)


def main():
    """
    Main entry point for command-line usage.
    Example usage for single resume analysis.
    """
    # Example usage - replace with your actual values
    resume_path = "path/to/resume.pdf"
    job_title = "Senior Software Engineer"
    job_description = """
    We are looking for a senior software engineer with expertise in Python,
    machine learning, and cloud technologies...
    """
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run single analysis
    result = run_single_analysis(resume_path, job_title, job_description)
    
    # Print results
    print("\n" + "="*50)
    print("RESUME ANALYSIS RESULTS")
    print("="*50)
    
    if result.get('valid', False):
        print(f"Candidate: {result.get('candidate_name', 'Unknown')}")
        print(f"Score: {result.get('score', 0):.1f}/10")
        print(f"Recommendation: {result.get('recommendation', 'Unknown')}")
        print("\nReport saved to output/reports/")
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()