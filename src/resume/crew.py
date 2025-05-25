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
from tools.custom_tool import CustomPDFTool
from report_schema import ReportModel  # Enforce standardized report structure

load_dotenv()

# Reduced temperature for more consistent results
llm1 = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1  # Reduced from 0.3 for more consistency
)
llm2 = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1  # Reduced from 0.3 for more consistency
)
llm3 = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1  # Reduced from 0.3 for more consistency
)
llm4 = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.1  # Reduced from 0.3 for more consistency
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
            llm=llm1,
        )

    @agent
    def matching_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['matching_specialist'],
            tools=[],
            verbose=False,
            llm=llm2,
        )

    @agent
    def report_generator(self) -> Agent:
        # NOTE: The report_generator agent MUST use the ReportModel schema for all report output.
        # The agent should fill a ReportModel instance, then call to_markdown() to render the report.
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=False,
            llm=llm4,
        )

    @task
    def document_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['document_analysis_task'],
            output_file='document_analysis_task.md'
        )
    
    @task
    def candidate_matching_task(self) -> Task:
        return Task(
            config=self.tasks_config['candidate_matching_task']
        )
    
    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_generation_task'], 
            output_file='report.md'
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