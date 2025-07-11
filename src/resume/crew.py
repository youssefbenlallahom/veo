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
from .tools.custom_tool import CustomPDFTool
from .schemas import DocumentAnalysisOutput, CandidateMatchingOutput, ReportGenerationOutput

load_dotenv()

# Reduced temperature for more consistent results
llm = LLM(
    model="gemini/gemini-2.5-pro",
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