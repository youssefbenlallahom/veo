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
from typing import List, Tuple, Any
from crewai_tools import *
from crewai.llm import LLM
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from langchain_openai import AzureChatOpenAI


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
os.environ["GEMINI_API_KEY"]="AIzaSyD4wr8nrFBM9scvZTAFvEzbpDZtFIR2qHw"
"""llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.0,
    api_key=os.getenv("GEMINI_API_KEY"),
)"""

os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'

# Disable all OpenTelemetry (including CrewAI)
os.environ['OTEL_SDK_DISABLED'] = 'true'

import os
os.environ["AZURE_API_KEY"] = "4Yyy8h5DwLynTdUgvzaDR9MEosUoonomgDFyt0bsXbApsiugWyPtJQQJ99BGACHYHv6XJ3w3AAAAACOG31ig"
os.environ["AZURE_API_BASE"] = "https://youss-mcpff1c2-eastus2.cognitiveservices.azure.com"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

llm = LLM(
    model="azure/gpt-4.1",  # This should match your deployment name
    api_key="4Yyy8h5DwLynTdUgvzaDR9MEosUoonomgDFyt0bsXbApsiugWyPtJQQJ99BGACHYHv6XJ3w3AAAAACOG31ig",
    base_url="https://youss-mcpff1c2-eastus2.cognitiveservices.azure.com",
    api_version="2024-12-01-preview"
)
    
def ensure_dict_guardrail(result) -> Tuple[bool, Any]:
    """Guardrail to ensure output is a dict, parsing from JSON string if needed."""
    # CrewAI passes TaskOutput or raw output depending on version
    raw = getattr(result, 'raw', result)
    if isinstance(raw, dict):
        return (True, raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return (True, parsed)
        except Exception as e:
            return (False, f"Output is not valid JSON: {e}")
    return (False, "Output is neither a dict nor a JSON string")

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
            output_json=DocumentAnalysisOutput,
            guardrail=ensure_dict_guardrail
        )
    
    @task
    def candidate_matching_task(self) -> Task:
        return Task(
            config=self.tasks_config['candidate_matching_task'],
            output_file='candidate_matching_task.json',
            output_json=CandidateMatchingOutput,
            guardrail=ensure_dict_guardrail
        )
    
    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_generation_task'], 
            output_file='report.json',
            output_json=ReportGenerationOutput,
            guardrail=ensure_dict_guardrail
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Resume crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disable memory for now
        )