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
import re
from datetime import datetime
from dotenv import load_dotenv
from src.resume.tools.custom_tool import CustomPDFTool
from src.resume.report_schema import ReportModel  # Enforce standardized report structure

load_dotenv()

# Reduced temperature for more consistent results
"""llm = LLM(
    model="gemini/gemini-2.5-pro",
    temperature=0.0,
    stream=False
)"""
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'

# Disable all OpenTelemetry (including CrewAI)
os.environ['OTEL_SDK_DISABLED'] = 'true'


llm = LLM(
    model=os.getenv("model"),  # This should match your deployment name
    api_key=os.getenv("AZURE_AI_API_KEY"),
    base_url=os.getenv("AZURE_AI_ENDPOINT"),
    api_version=os.getenv("AZURE_AI_API_VERSION"),
    temperature=0.0,  # Reduced for more consistent outputs
    stream=False,
)
@CrewBase
class Resume():
    """Resume crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self, pdf_path=None, candidate_name=None):
        super().__init__()
        self.candidate_name = candidate_name or "unknown_candidate"
        
        if pdf_path:
            try:
                self.pdf_tool = CustomPDFTool(pdf_path=pdf_path)
            except Exception as e:
                raise e
        else:
            self.pdf_tool = None

    @agent
    def document_analyzer(self) -> Agent:
        if self.pdf_tool is None:
            raise ValueError("PDF tool not initialized. Please provide a PDF path.")
        
        return Agent(
            config=self.agents_config['document_analyzer'],
            verbose=False,
            tools=[self.pdf_tool],
            llm=llm,
        )

    @agent
    def matching_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['matching_specialist'],
            tools=[],
            verbose=False,
            max_iter=3,
            reasoning=True,
            llm=llm,
        )

    @agent
    def report_generator(self) -> Agent:
        # NOTE: The report_generator agent MUST use the ReportModel schema for all report output.
        # The agent should fill a ReportModel instance, then call to_markdown() to render the report.
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=False,
            max_iter=1,
            llm=llm,
        )

    @task
    def document_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['document_analysis_task']
        )
    
    @task
    def candidate_matching_task(self) -> Task:
        return Task(
            config=self.tasks_config['candidate_matching_task']
        )
    
    @task
    def report_generation_task(self) -> Task:
        # Create reports directory if it doesn't exist
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with candidate name
        safe_name = re.sub(r'[^\w\-_\. ]', '_', self.candidate_name)
        filename = f"{reports_dir}/report_{safe_name}.json"
        
        return Task(
            config=self.tasks_config['report_generation_task'], 
            output_file=filename,
            output_json=ReportModel
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Resume crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False
        )