"""
Utility functions for working with Pydantic models and JSON validation.
"""

import json
from typing import Dict, Any, Union, Type
from pathlib import Path
import logging
from pydantic import BaseModel, ValidationError

from schemas import (
    DocumentAnalysisOutput, 
    CandidateMatchingOutput, 
    ReportGenerationOutput
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """Handles validation and conversion of JSON data to Pydantic models."""
    
    MODEL_MAPPING = {
        'document_analysis': DocumentAnalysisOutput,
        'candidate_matching': CandidateMatchingOutput,
        'report_generation': ReportGenerationOutput
    }
    
    @staticmethod
    def validate_json_string(json_str: str, model_type: str) -> Union[BaseModel, Dict[str, Any]]:
        """
        Validate a JSON string against a specific model type.
        
        Args:
            json_str: JSON string to validate
            model_type: Type of model to validate against ('document_analysis', 'candidate_matching', 'report_generation')
            
        Returns:
            Validated Pydantic model instance or original dict if validation fails
        """
        try:
            # Parse JSON string
            data = json.loads(json_str)
            return ModelValidator.validate_dict(data, model_type)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string: {e}")
            return {"error": f"Invalid JSON: {e}"}
    
    @staticmethod
    def validate_dict(data: Dict[str, Any], model_type: str) -> Union[BaseModel, Dict[str, Any]]:
        """
        Validate a dictionary against a specific model type.
        
        Args:
            data: Dictionary to validate
            model_type: Type of model to validate against
            
        Returns:
            Validated Pydantic model instance or original dict if validation fails
        """
        if model_type not in ModelValidator.MODEL_MAPPING:
            logger.error(f"Unknown model type: {model_type}")
            return {"error": f"Unknown model type: {model_type}"}
        
        model_class = ModelValidator.MODEL_MAPPING[model_type]
        
        try:
            # Create and validate model instance
            model_instance = model_class(**data)
            logger.info(f"Successfully validated {model_type} model")
            return model_instance
        except ValidationError as e:
            logger.error(f"Validation error for {model_type}: {e}")
            return {"error": f"Validation error: {e}", "original_data": data}
    
    @staticmethod
    def validate_file(file_path: Union[str, Path], model_type: str) -> Union[BaseModel, Dict[str, Any]]:
        """
        Validate a JSON file against a specific model type.
        
        Args:
            file_path: Path to JSON file
            model_type: Type of model to validate against
            
        Returns:
            Validated Pydantic model instance or error dict
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelValidator.validate_dict(data, model_type)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            return {"error": f"Invalid JSON in file: {e}"}
    
    @staticmethod
    def model_to_dict(model: BaseModel) -> Dict[str, Any]:
        """Convert a Pydantic model to a dictionary."""
        return model.model_dump()
    
    @staticmethod
    def model_to_json(model: BaseModel, indent: int = 2) -> str:
        """Convert a Pydantic model to a JSON string."""
        return model.model_dump_json(indent=indent)
    
    @staticmethod
    def save_model_to_file(model: BaseModel, file_path: Union[str, Path]) -> bool:
        """
        Save a Pydantic model to a JSON file.
        
        Args:
            model: Pydantic model instance
            file_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(ModelValidator.model_to_json(model))
            logger.info(f"Successfully saved model to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to file {file_path}: {e}")
            return False


class SchemaDocumentationGenerator:
    """Generates documentation for the schemas."""
    
    @staticmethod
    def generate_field_documentation(model_class: Type[BaseModel]) -> Dict[str, Any]:
        """Generate documentation for all fields in a model."""
        fields_doc = {}
        
        for field_name, field_info in model_class.model_fields.items():
            fields_doc[field_name] = {
                "type": str(field_info.annotation),
                "description": field_info.description or "No description provided",
                "required": field_info.is_required(),
                "default": field_info.default if field_info.default is not None else "No default"
            }
        
        return fields_doc
    
    @staticmethod
    def generate_schema_documentation() -> Dict[str, Any]:
        """Generate complete documentation for all schemas."""
        documentation = {}
        
        for model_name, model_class in ModelValidator.MODEL_MAPPING.items():
            documentation[model_name] = {
                "model_class": model_class.__name__,
                "description": model_class.__doc__ or "No description provided",
                "fields": SchemaDocumentationGenerator.generate_field_documentation(model_class)
            }
        
        return documentation
    
    @staticmethod
    def save_documentation_to_file(file_path: Union[str, Path]) -> bool:
        """Save schema documentation to a JSON file."""
        try:
            docs = SchemaDocumentationGenerator.generate_schema_documentation()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2, default=str)
            logger.info(f"Documentation saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving documentation: {e}")
            return False


def create_example_data():
    """Create example data for testing the schemas."""
    
    # Example document analysis data
    document_example = {
        "contact_information": {
            "full_name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-123-4567",
            "linkedin": "https://linkedin.com/in/johndoe",
            "github": "https://github.com/johndoe",
            "portfolio": "https://johndoe.dev",
            "address": "123 Main St, City, State 12345",
            "other_profiles": ["https://twitter.com/johndoe"]
        },
        "education": [{
            "degree": "Bachelor of Science in Computer Science",
            "institution": "University of Technology",
            "graduation_date": "2020-05-15",
            "gpa": "3.8/4.0",
            "honors": "Magna Cum Laude",
            "relevant_coursework": ["Data Structures", "Algorithms", "Database Systems"],
            "thesis_project": "Machine Learning for Text Analysis"
        }],
        "work_experience": [{
            "job_title": "Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "start_date": "2020-06-01",
            "end_date": "Present",
            "duration_months": "42",
            "employment_type": "full-time",
            "key_responsibilities": ["Develop web applications", "Code reviews", "Mentoring junior developers"],
            "achievements": ["Improved system performance by 30%", "Led team of 5 developers"],
            "tools_used": ["Python", "React", "PostgreSQL", "Docker"]
        }],
        "skills": {
            "professional_skills": ["Software Development", "Team Leadership", "Project Management"],
            "technical_skills": ["Python", "JavaScript", "SQL", "Machine Learning"],
            "software_tools": ["Git", "Docker", "Kubernetes", "AWS"],
            "methodologies": ["Agile", "Scrum", "TDD", "CI/CD"],
            "soft_skills": ["Communication", "Problem Solving", "Teamwork"],
            "domain_expertise": ["Web Development", "Data Science", "Cloud Computing"]
        },
        "languages": [{
            "language": "English",
            "proficiency_level": "Native",
            "certification": "Not found"
        }],
        "certifications": [{
            "name": "AWS Certified Developer",
            "issuer": "Amazon Web Services",
            "date_obtained": "2021-03-15",
            "expiration_date": "2024-03-15",
            "credential_id": "AWS-DEV-123456",
            "verification_url": "https://aws.amazon.com/certification/verify"
        }],
        "projects": [{
            "name": "E-commerce Platform",
            "description": "Full-stack e-commerce solution",
            "skills_used": ["React", "Node.js", "MongoDB"],
            "role": "Lead Developer",
            "team_size": "4",
            "duration": "6 months",
            "url": "https://github.com/johndoe/ecommerce",
            "achievements": ["Processed 1000+ orders daily", "99.9% uptime"]
        }],
        "additional_sections": {
            "publications": ["Machine Learning in Web Development - Tech Journal 2022"],
            "patents": ["Not found"],
            "awards": ["Employee of the Month - Tech Corp 2021"],
            "volunteering": ["Code for Good - Teaching programming to underprivileged youth"],
            "professional_associations": ["IEEE Computer Society", "ACM"],
            "licenses": ["Not found"],
            "references": ["Available upon request"]
        },
        "analysis_summary": {
            "total_experience_years": "3.5",
            "total_experience_months": "42",
            "career_level": "mid",
            "primary_domain": "Software Development",
            "key_strengths": ["Technical Leadership", "Full-stack Development", "Problem Solving"],
            "employment_gaps": [],
            "career_progression": "upward",
            "most_recent_role": "Software Engineer at Tech Corp"
        }
    }
    
    return document_example


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test validation with example data
    example_data = create_example_data()
    result = ModelValidator.validate_dict(example_data, 'document_analysis')
    
    if isinstance(result, BaseModel):
        print("✅ Validation successful!")
        print(f"Model type: {type(result).__name__}")
        print(f"Candidate name: {result.contact_information.full_name}")
    else:
        print("❌ Validation failed!")
        print(result)
    
    # Generate documentation
    print("\n📚 Generating schema documentation...")
    SchemaDocumentationGenerator.save_documentation_to_file("schema_documentation.json")
