from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class ScoreDetail(BaseModel):
    section: str  # Section name from Gemini rubric
    raw_score: float  # Always out of 10
    weight: float  # As percentage (e.g., 25 for 25%)
    weighted_score: float  # Contribution to total (normalized to /10)

class ReportModel(BaseModel):
    applied_job_title: str = Field(..., description="Job title from the user.")
    applied_job_description: str = Field(..., description="Job description provided by the user.")
    candidate_name: str = Field(..., description="Candidate's full name.")
    candidate_job_title: Optional[str] = Field(None, description="Candidate's current job title/role.")
    candidate_experience: str = Field(..., description="Total years of experience.")
    candidate_background: str = Field(..., description="Short background or current role.")
    requirements_analysis: List[str] = Field(..., description="List of job requirements based only on user-provided text.")
    match_results: Dict[str, str] = Field(..., description="Key match results (skills, experience, education, etc.).")
    scoring_weights: Dict[str, float] = Field(..., description="Scoring weights for each criterion.")
    score_details: List[ScoreDetail] = Field(..., description="Detailed scoring breakdown.")
    total_weighted_score: float = Field(..., description="Total weighted score out of 10.")
    strengths: List[str] = Field(..., description="List of candidate strengths.")
    gaps: List[str] = Field(..., description="List of gaps or missing requirements.")
    rationale: str = Field(..., description="Analysis rationale and insights.")
    risk: Optional[str] = Field(None, description="Potential risks or caveats.")
    next_steps: Optional[List[str]] = Field(None, description="Recommended next steps if hired.")

    def to_markdown(self) -> str:
        """Render the report as a standardized markdown string."""
        md = []
        md.append(f"# Hiring Report: {self.applied_job_title}")
        md.append("\n## Job Description (User-Provided)")
        md.append(self.applied_job_description)
        md.append("\n## Candidate")
        md.append(f"- **Name:** {self.candidate_name}")
        if self.candidate_job_title:
            md.append(f"- **Current Role:** {self.candidate_job_title}")
        md.append(f"- **Experience:** {self.candidate_experience}")
        md.append(f"- **Background:** {self.candidate_background}")
        md.append("\n## Requirements Analysis")
        for req in self.requirements_analysis:
            md.append(f"- {req}")
        md.append("\n## Match Results")
        for k, v in self.match_results.items():
            md.append(f"- **{k}:** {v}")
        md.append("\n## Scoring Weights")
        for k, v in self.scoring_weights.items():
            md.append(f"- **{k}:** {v*100:.0f}%")
        md.append("\n## Weighted Score Analysis")
        md.append("| Section | Weight (%) | Raw Score (/10) | Weighted Score |")
        md.append("|---------|------------|-----------------|---------------|")
        for s in self.score_details:
            md.append(f"| {s.section} | {s.weight:.0f}% | {s.raw_score:.2f} | {s.weighted_score:.2f} |")
        md.append(f"\n**Overall Score:** {self.total_weighted_score:.2f}/10")
        md.append("\n## Strengths")
        for s in self.strengths:
            md.append(f"- {s}")
        md.append("\n## Gaps")
        for g in self.gaps:
            md.append(f"- {g}")
        md.append(f"\n## Decision\n**Status:** {self.recommendation}\n**Reason:** {self.rationale}")
        if self.risk:
            md.append(f"\n**Risk:** {self.risk}")
        if self.next_steps:
            md.append("\n**Next Steps:**")
            for step in self.next_steps:
                md.append(f"- {step}")
        return "\n".join(md)
