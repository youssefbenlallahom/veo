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
import json

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from crew import Resume
from utils.file_handler import FileHandler
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
        
        # Convert output_dir to Path object for compatibility
        output_path = Path(output_dir)
        
        # Initialize utilities
        self.file_handler = FileHandler(output_path)
        self.report_parser = ReportParser()
        self.pdf_validator = PDFValidator()
        
        # Load barem from JSON file
        self.barem = self._load_barem_from_json()
        
        logger.info(f"Resume Analysis Engine initialized")
    
    def _load_barem_from_json(self) -> Optional[Dict]:
        """Load barem directly from barem_gemini.json file."""
        try:
            # Look for barem_gemini.json in the project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            barem_file = project_root / "barem_gemini.json"
            
            if not barem_file.exists():
                logger.error(f"Barem file not found at {barem_file}")
                return None
                
            with open(barem_file, 'r', encoding='utf-8') as f:
                barem_data = json.load(f)
                
            logger.info(f"Barem loaded successfully from {barem_file}")
            return barem_data
            
        except Exception as e:
            logger.error(f"Error loading barem from JSON: {e}")
            return None
    
    
    def _parse_result_directly(self, result, candidate_name: str) -> Dict[str, Any]:
        """Parse crew result directly without saving to file."""
        try:
            # Extract the raw result from crew output
            if hasattr(result, 'raw'):
                raw_result = result.raw
            else:
                raw_result = result
                
            # If it's a string, try to parse as JSON
            if isinstance(raw_result, str):
                try:
                    parsed_result = json.loads(raw_result)
                except json.JSONDecodeError:
                    # If not JSON, create a basic structure
                    parsed_result = {
                        "candidate_name": candidate_name,
                        "score": 0,
                        "recommendation": "Error parsing result",
                        "raw_output": raw_result
                    }
            elif isinstance(raw_result, dict):
                parsed_result = raw_result
            else:
                # Handle other types
                parsed_result = {
                    "candidate_name": candidate_name,
                    "score": 0,
                    "recommendation": "Unknown result format",
                    "raw_output": str(raw_result)
                }
            
            # Ensure required fields are present
            if "candidate_name" not in parsed_result:
                parsed_result["candidate_name"] = candidate_name
            if "valid" not in parsed_result:
                parsed_result["valid"] = True
            if "score" not in parsed_result:
                parsed_result["score"] = 0
            if "recommendation" not in parsed_result:
                parsed_result["recommendation"] = "Unknown"
                
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error parsing result for {candidate_name}: {e}")
            return {
                "candidate_name": candidate_name,
                "valid": False,
                "error": str(e),
                "score": 0,
                "recommendation": "Error"
            }

    def generate_or_load_barem(self, job_title: str, job_description: str) -> Optional[Dict]:
        """Return the loaded barem from JSON file."""
        return self.barem
    
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
            
            # Parse the report directly from result
            analysis_result = self._parse_result_directly(result, candidate_name)
            
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
        
        logger.info(f"Batch analysis completed. Processed {len(results)} resumes")
        return results

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
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()