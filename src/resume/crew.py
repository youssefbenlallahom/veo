import sys
import os
from pathlib import Path

# Add current directory and parent directory to Python path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))

try:
    from crewai import Agent, Crew, Process, Task
except ImportError as e:
    print("Import Error:", e)
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import *
from crewai.llm import LLM
import os
from datetime import datetime
from dotenv import load_dotenv

# Try to import with better error handling
try:
    from .tools.custom_tool import CustomPDFTool
except ImportError:
    try:
        from .tools.custom_tool import CustomPDFTool
    except ImportError:
        import sys
        tools_path = os.path.join(os.path.dirname(__file__), 'tools')
        sys.path.append(tools_path)
        from custom_tool import CustomPDFTool

try:
    from .schemas import DocumentAnalysisOutput, CandidateMatchingOutput, ReportGenerationOutput
except ImportError:
    try:
        from .schemas import DocumentAnalysisOutput, CandidateMatchingOutput, ReportGenerationOutput
    except ImportError:
        # As a last resort, try absolute import
        schemas_path = os.path.join(os.path.dirname(__file__), 'schemas.py')
        import importlib.util
        spec = importlib.util.spec_from_file_location("schemas", schemas_path)
        schemas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schemas)
        DocumentAnalysisOutput = schemas.DocumentAnalysisOutput
        CandidateMatchingOutput = schemas.CandidateMatchingOutput
        ReportGenerationOutput = schemas.ReportGenerationOutput

load_dotenv()

# Use Gemini for more reliable structured output
"""
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.0,
    api_key=os.getenv("GEMINI_API_KEY"),
)"""

# Backup Azure configuration if needed
llm = LLM(
    model=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    temperature=0.0,
)
    
@CrewBase
class Resume():
    """Resume crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self, pdf_path=None):
        super().__init__()
        print(f"[DEBUG] Resume class initialized with PDF path: {pdf_path}")
        
        if pdf_path:
            try:
                self.pdf_tool = CustomPDFTool(pdf_path=pdf_path)
                print(f"[DEBUG] CustomPDFTool created successfully")
                
                test_result = self.pdf_tool._run("extract_all")
                print(f"[DEBUG] PDF tool test result length: {len(test_result)} characters")
                
                if test_result and len(test_result) > 0:
                    print(f"[DEBUG] PDF content preview: {test_result[:200]}...")
                else:
                    print("[DEBUG] Warning: PDF extraction returned empty content")
                
            except Exception as e:
                print(f"[DEBUG] Error creating CustomPDFTool: {e}")
                raise e
        else:
            self.pdf_tool = None
            print("[DEBUG] No PDF path provided")

    @agent
    def document_analyzer(self) -> Agent:
        if self.pdf_tool is None:
            raise ValueError("PDF tool not initialized. Please provide a PDF path.")
        
        print(f"[DEBUG] Creating document_analyzer agent with PDF tool")
        
        return Agent(
            config=self.agents_config['document_analyzer'],
            verbose=True,
            tools=[self.pdf_tool],
            llm=llm,
        )

    @agent
    def matching_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['matching_specialist'],
            reasoning=True,
            verbose=False,
            llm=llm,
        )

    @agent
    def report_generator(self) -> Agent:
        # NOTE: The report_generator agent uses the ReportGenerationOutput schema for structured JSON output.
        # This ensures consistent, machine-readable reports that can be easily processed by the UI.
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=False,
            llm=llm,
        )

    @task
    def document_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['document_analysis_task'],
            output_file='document_analysis_task.json',
            output_json=DocumentAnalysisOutput
        )
    
    @task
    def candidate_matching_task(self) -> Task:
        return Task(
            config=self.tasks_config['candidate_matching_task'],
            output_file='candidate_matching_task.json',
            output_json=CandidateMatchingOutput
        )
    
    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_generation_task'], 
            output_file='report.json',
            output_json=ReportGenerationOutput
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Resume crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )