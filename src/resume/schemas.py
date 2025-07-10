"""
Pydantic models for Resume Analysis CrewAI project output validation.
These models enforce the exact structure defined in tasks.yaml.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum


# =============================================================================
# Document Analysis Task Models
# =============================================================================

class ContactInformation(BaseModel):
    full_name: str = Field(description="Extracted name or 'Not found'")
    email: str = Field(description="Email address or 'Not found'")
    phone: str = Field(description="Phone number or 'Not found'")
    linkedin: str = Field(description="LinkedIn URL or 'Not found'")
    github: str = Field(description="GitHub URL or 'Not found'")
    portfolio: str = Field(description="Portfolio URL or 'Not found'")
    address: str = Field(description="Physical address or 'Not found'")
    other_profiles: List[str] = Field(default=[], description="Array of other social/professional profiles")


class Education(BaseModel):
    degree: str = Field(description="Degree type and field of study")
    institution: str = Field(description="School/university name")
    graduation_date: str = Field(description="Graduation date or expected date")
    gpa: str = Field(description="GPA if mentioned")
    honors: str = Field(description="Honors, distinctions, or awards")
    relevant_coursework: List[str] = Field(default=[], description="Array of relevant courses")
    thesis_project: str = Field(description="Thesis or capstone project details")


class WorkExperience(BaseModel):
    job_title: str = Field(description="Position title")
    company: str = Field(description="Company name")
    location: str = Field(description="Work location")
    start_date: str = Field(description="Start date")
    end_date: str = Field(description="End date or 'Present'")
    duration_months: str = Field(description="Calculated duration in months")
    employment_type: str = Field(description="full-time, part-time, contract, internship, etc.")
    key_responsibilities: List[str] = Field(default=[], description="Array of main responsibilities")
    achievements: List[str] = Field(default=[], description="Array of quantified achievements")
    tools_used: List[str] = Field(default=[], description="Array of tools, technologies, or methods used")


class Skills(BaseModel):
    professional_skills: List[str] = Field(default=[], description="Array of job-specific professional skills")
    technical_skills: List[str] = Field(default=[], description="Array of technical competencies")
    software_tools: List[str] = Field(default=[], description="Array of software and tools")
    methodologies: List[str] = Field(default=[], description="Array of methodologies, frameworks, or approaches")
    soft_skills: List[str] = Field(default=[], description="Array of interpersonal and soft skills")
    domain_expertise: List[str] = Field(default=[], description="Array of industry/domain knowledge")


class Language(BaseModel):
    language: str = Field(description="Language name")
    proficiency_level: str = Field(description="Proficiency level (A1, A2, B1, B2, C1, C2, Native, Fluent, etc.)")
    certification: str = Field(description="Language certification if any")


class Certification(BaseModel):
    name: str = Field(description="Certification name")
    issuer: str = Field(description="Issuing organization")
    date_obtained: str = Field(description="Date obtained")
    expiration_date: str = Field(description="Expiration date if applicable")
    credential_id: str = Field(description="Credential ID if provided")
    verification_url: str = Field(description="Verification URL if provided")


class Project(BaseModel):
    name: str = Field(description="Project name")
    description: str = Field(description="Brief description")
    skills_used: List[str] = Field(default=[], description="Array of skills/tools/technologies used")
    role: str = Field(description="Your role in the project")
    team_size: str = Field(description="Team size if mentioned")
    duration: str = Field(description="Project duration")
    url: str = Field(description="Project URL if available")
    achievements: List[str] = Field(default=[], description="Array of project outcomes/achievements")


class AdditionalSections(BaseModel):
    publications: List[str] = Field(default=[], description="Array of publications")
    patents: List[str] = Field(default=[], description="Array of patents")
    awards: List[str] = Field(default=[], description="Array of awards and recognition")
    volunteering: List[str] = Field(default=[], description="Array of volunteer experience")
    professional_associations: List[str] = Field(default=[], description="Array of memberships")
    licenses: List[str] = Field(default=[], description="Array of professional licenses")
    references: List[str] = Field(default=[], description="Array of references or 'Available upon request'")


class EmploymentGap(BaseModel):
    gap_period: str = Field(description="Period of gap")
    duration_months: str = Field(description="Gap duration in months")
    potential_reason: str = Field(description="Inferred reason if obvious")


class AnalysisSummary(BaseModel):
    total_experience_years: str = Field(description="Calculated total years of experience")
    total_experience_months: str = Field(description="Calculated total months of experience")
    career_level: str = Field(description="entry, mid, senior, executive")
    primary_domain: str = Field(description="Main area of expertise")
    key_strengths: List[str] = Field(default=[], description="Array of top 3-5 strengths")
    employment_gaps: List[EmploymentGap] = Field(default=[], description="Array of employment gaps")
    career_progression: str = Field(description="upward, lateral, mixed, declining")
    most_recent_role: str = Field(description="Most recent job title and company")


class DocumentAnalysisOutput(BaseModel):
    """Complete output model for document analysis task"""
    contact_information: ContactInformation
    education: List[Education] = Field(default=[])
    work_experience: List[WorkExperience] = Field(default=[])
    skills: Skills
    languages: List[Language] = Field(default=[])
    certifications: List[Certification] = Field(default=[])
    projects: List[Project] = Field(default=[])
    additional_sections: AdditionalSections
    analysis_summary: AnalysisSummary


# =============================================================================
# Candidate Matching Task Models
# =============================================================================

class JobAnalysis(BaseModel):
    job_title: str = Field(description="Extracted job title")
    job_description: str = Field(description="Full job description provided")
    critical_requirements: List[str] = Field(default=[], description="Array of must-have requirements from job description")
    preferred_requirements: List[str] = Field(default=[], description="Array of nice-to-have requirements from job description")
    experience_required: str = Field(description="Years and type of experience specified")
    education_required: str = Field(description="Education requirements if specified")


class CandidateOverview(BaseModel):
    candidate_name: str = Field(description="Candidate name from resume")
    current_title: str = Field(description="Most recent job title")
    total_experience_years: str = Field(description="Total years of experience")
    relevant_experience_years: str = Field(description="Years of relevant experience for this role")
    education_level: str = Field(description="Highest education level")
    key_skills: List[str] = Field(default=[], description="Array of candidate's key skills relevant to job")


class RubricEvaluation(BaseModel):
    section_name: str = Field(description="Rubric section name")
    weight_percentage: str = Field(description="Weight from provided rubric")
    candidate_evidence: List[str] = Field(default=[], description="Array of specific evidence from resume")
    section_score: str = Field(description="Score based on rubric criteria")
    score_justification: str = Field(description="Detailed explanation of why this score was assigned")
    weighted_score: str = Field(description="Calculated weighted score (section_score × weight)")


class SkillsMatch(BaseModel):
    matched_skills: List[str] = Field(default=[], description="Array of skills candidate has that match job requirements")
    missing_skills: List[str] = Field(default=[], description="Array of required skills candidate lacks")
    additional_skills: List[str] = Field(default=[], description="Array of relevant skills candidate has beyond requirements")
    match_percentage: str = Field(description="Percentage of required skills matched")


class ExperienceMatch(BaseModel):
    required_experience: str = Field(description="Experience requirement from job")
    candidate_experience: str = Field(description="Candidate's relevant experience")
    experience_gap: str = Field(description="Gap analysis if any")
    match_level: str = Field(description="exceeds, meets, below, or significantly below")


class EducationMatch(BaseModel):
    required_education: str = Field(description="Education requirement from job")
    candidate_education: str = Field(description="Candidate's education")
    match_level: str = Field(description="exceeds, meets, below, or not specified")


class MatchAnalysis(BaseModel):
    skills_match: SkillsMatch
    experience_match: ExperienceMatch
    education_match: EducationMatch


class ScoreBreakdown(BaseModel):
    criterion: str = Field(description="Criterion name")
    raw_score: str = Field(description="Raw score out of 10")
    weight: str = Field(description="Weight percentage")
    weighted_points: str = Field(description="raw_score × weight")


class ScoringSummary(BaseModel):
    total_weighted_score: str = Field(description="Sum of all weighted scores")
    normalized_score: str = Field(description="Score out of 10")
    score_breakdown: List[ScoreBreakdown] = Field(default=[])
    weights_validation: str = Field(description="Confirmation that weights sum to 100%")


class EvaluationResults(BaseModel):
    critical_missing_requirements: List[str] = Field(default=[], description="Array of critical missing requirements")
    key_strengths: List[str] = Field(default=[], description="Array of candidate's key strengths for this role")
    areas_for_improvement: List[str] = Field(default=[], description="Array of areas where candidate could improve")
    unique_value_propositions: List[str] = Field(default=[], description="Array of unique qualities candidate brings")
    risk_factors: List[str] = Field(default=[], description="Array of potential concerns or risks")


class FinalRecommendation(BaseModel):
    status: str = Field(description="RECOMMENDED | NOT RECOMMENDED | CONDITIONALLY RECOMMENDED")
    confidence_level: str = Field(description="high, medium, low")
    primary_reason: str = Field(description="Main reason for recommendation decision")
    supporting_factors: List[str] = Field(default=[], description="Array of factors supporting the decision")
    conditions_if_applicable: List[str] = Field(default=[], description="Array of conditions if conditionally recommended")
    next_steps: List[str] = Field(default=[], description="Array of suggested next steps in hiring process")


class DetailedFeedback(BaseModel):
    for_candidate: str = Field(description="Constructive feedback for candidate improvement")
    for_hiring_manager: str = Field(description="Insights and recommendations for hiring manager")
    interview_focus_areas: List[str] = Field(default=[], description="Array of areas to focus on during interview")
    reference_check_priorities: List[str] = Field(default=[], description="Array of areas to verify with references")


class CandidateMatchingOutput(BaseModel):
    """Complete output model for candidate matching task"""
    job_analysis: JobAnalysis
    candidate_overview: CandidateOverview
    rubric_evaluation: List[RubricEvaluation] = Field(default=[])
    match_analysis: MatchAnalysis
    scoring_summary: ScoringSummary
    evaluation_results: EvaluationResults
    final_recommendation: FinalRecommendation
    detailed_feedback: DetailedFeedback


# =============================================================================
# Report Generation Task Models
# =============================================================================

class ExecutiveSummary(BaseModel):
    candidate_name: str = Field(description="Candidate full name")
    position_applied: str = Field(description="Job title")
    overall_recommendation: str = Field(description="STRONGLY RECOMMENDED | RECOMMENDED | CONDITIONALLY RECOMMENDED | NOT RECOMMENDED")
    overall_score: str = Field(description="Normalized score out of 10")
    confidence_level: str = Field(description="HIGH | MEDIUM | LOW")
    key_decision_factors: List[str] = Field(default=[], description="Array of top 3-5 factors influencing recommendation")
    critical_concerns: List[str] = Field(default=[], description="Array of major concerns if any")
    recommendation_summary: str = Field(description="2-3 sentence summary of final recommendation and rationale")


class CandidateProfile(BaseModel):
    professional_summary: str = Field(description="Comprehensive 3-4 sentence candidate overview")
    career_trajectory: str = Field(description="ascending, stable, transitioning, declining with explanation")
    total_experience: str = Field(description="Total years of experience")
    relevant_experience: str = Field(description="Years of directly relevant experience")
    education_level: str = Field(description="Highest degree and field")
    current_status: str = Field(description="employed, unemployed, seeking transition")
    geographic_considerations: str = Field(description="Location compatibility assessment")
    career_progression_analysis: str = Field(description="Detailed analysis of career growth pattern")
    industry_experience: List[str] = Field(default=[], description="Array of industries worked in")
    company_sizes_worked: List[str] = Field(default=[], description="startup, mid-size, enterprise, etc.")


class MustHaveRequirement(BaseModel):
    requirement: str = Field(description="Specific requirement")
    candidate_match: str = Field(description="EXCEEDS | MEETS | PARTIALLY MEETS | DOES NOT MEET")
    evidence: str = Field(description="Specific evidence from resume")
    gap_analysis: str = Field(description="Description of any gaps")
    risk_level: str = Field(description="HIGH | MEDIUM | LOW")
    impact_on_role_success: str = Field(description="How this requirement affects job performance")


class PreferredRequirement(BaseModel):
    requirement: str = Field(description="Specific requirement")
    candidate_match: str = Field(description="EXCEEDS | MEETS | PARTIALLY MEETS | DOES NOT MEET")
    evidence: str = Field(description="Specific evidence from resume")
    added_value: str = Field(description="Description of additional value if met")


class JobRequirementsAnalysis(BaseModel):
    must_have_requirements: List[MustHaveRequirement] = Field(default=[])
    preferred_requirements: List[PreferredRequirement] = Field(default=[])
    requirements_satisfaction_score: str = Field(description="Percentage of requirements met")
    critical_missing_requirements: List[str] = Field(default=[], description="Array of missing must-have requirements")
    exceeds_expectations_in: List[str] = Field(default=[], description="Array of areas where candidate exceeds requirements")


class RubricBreakdown(BaseModel):
    criterion: str = Field(description="Criterion name")
    raw_score: str = Field(description="Score out of 10")
    weight_percentage: str = Field(description="Weight in evaluation")
    weighted_points: str = Field(description="Calculated weighted points")
    performance_level: str = Field(description="EXCELLENT | GOOD | SATISFACTORY | NEEDS IMPROVEMENT | POOR")
    supporting_evidence: List[str] = Field(default=[], description="Array of specific evidence")
    score_justification: str = Field(description="Detailed explanation of score assignment")
    benchmark_comparison: str = Field(description="How this compares to typical candidates")


class ScoreDistribution(BaseModel):
    technical_skills: str = Field(description="Weighted score for technical competencies")
    experience_relevance: str = Field(description="Weighted score for experience match")
    educational_background: str = Field(description="Weighted score for education match")
    achievements_impact: str = Field(description="Weighted score for demonstrated achievements")
    skill_depth_breadth: str = Field(description="Weighted score for skill comprehensiveness")


class DetailedScoringAnalysis(BaseModel):
    rubric_breakdown: List[RubricBreakdown] = Field(default=[])
    scoring_methodology: str = Field(description="Explanation of how scores were calculated")
    score_distribution: ScoreDistribution
    total_weighted_score: str = Field(description="Final calculated score")
    percentile_ranking: str = Field(description="Estimated percentile vs typical candidates for this role")
    score_reliability: str = Field(description="Confidence in scoring accuracy")


class CoreStrength(BaseModel):
    strength: str = Field(description="Specific strength")
    evidence: str = Field(description="Supporting evidence from resume")
    business_impact: str = Field(description="How this strength benefits the role/company")
    uniqueness_factor: str = Field(description="How rare/valuable this strength is in market")
    quantified_impact: str = Field(description="Measurable outcomes if available")


class UniqueValueProposition(BaseModel):
    value_proposition: str = Field(description="Unique aspect candidate brings")
    market_advantage: str = Field(description="Competitive advantage this provides")
    differentiation_level: str = Field(description="How this sets candidate apart")
    strategic_value: str = Field(description="Long-term strategic benefit to organization")


class StandoutAchievement(BaseModel):
    achievement: str = Field(description="Specific achievement")
    quantified_impact: str = Field(description="Measurable impact if available")
    relevance_to_role: str = Field(description="How this achievement applies to target role")
    skill_demonstration: str = Field(description="What skills this achievement demonstrates")
    complexity_level: str = Field(description="Sophistication of the achievement")


class StrengthsAndDifferentiators(BaseModel):
    core_strengths: List[CoreStrength] = Field(default=[])
    unique_value_propositions: List[UniqueValueProposition] = Field(default=[])
    standout_achievements: List[StandoutAchievement] = Field(default=[])
    leadership_indicators: List[str] = Field(default=[], description="Array of leadership qualities demonstrated")
    innovation_examples: List[str] = Field(default=[], description="Array of innovative contributions or thinking")


class CriticalGap(BaseModel):
    gap: str = Field(description="Specific missing requirement")
    impact_level: str = Field(description="HIGH | MEDIUM | LOW")
    severity_assessment: str = Field(description="How critical this gap is to role success")
    likelihood_of_success_despite_gap: str = Field(description="Assessment of success probability")
    gap_category: str = Field(description="technical, experience, educational, cultural")


class ExperienceGap(BaseModel):
    gap_area: str = Field(description="Specific experience missing")
    years_of_experience_gap: str = Field(description="Quantified experience shortfall")
    complexity_gap: str = Field(description="Sophistication level missing")
    industry_specific_concerns: str = Field(description="Industry-specific experience gaps")


class SkillDeficiency(BaseModel):
    skill: str = Field(description="Specific skill gap")
    proficiency_gap: str = Field(description="Level of proficiency missing")
    criticality_to_role: str = Field(description="How essential this skill is")
    market_availability: str = Field(description="How common this skill is in market")


class PerformanceRiskFactor(BaseModel):
    risk_factor: str = Field(description="Specific risk")
    probability: str = Field(description="Likelihood this risk manifests")
    impact: str = Field(description="Potential impact on performance")
    risk_category: str = Field(description="technical, cultural, motivational, capacity")


class CulturalFitAssessment(BaseModel):
    cultural_alignment_score: str = Field(description="Score out of 10")
    cultural_strengths: List[str] = Field(default=[], description="Array of cultural fit strengths")
    cultural_concerns: List[str] = Field(default=[], description="Array of potential cultural misalignment")
    team_integration_assessment: str = Field(description="Likelihood of successful team integration")


class GapsAndRiskAssessment(BaseModel):
    critical_gaps: List[CriticalGap] = Field(default=[])
    experience_gaps: List[ExperienceGap] = Field(default=[])
    skill_deficiencies: List[SkillDeficiency] = Field(default=[])
    performance_risk_factors: List[PerformanceRiskFactor] = Field(default=[])
    cultural_fit_assessment: CulturalFitAssessment


class ComparativeAnalysis(BaseModel):
    market_positioning: str = Field(description="How candidate compares to typical market candidates")
    salary_market_alignment: str = Field(description="Alignment with market compensation")
    competitive_advantage: str = Field(description="What makes this candidate stand out")
    alternative_candidates_consideration: str = Field(description="How this candidate compares to alternatives")
    urgency_vs_quality_tradeoff: str = Field(description="Assessment of hiring urgency vs candidate quality")
    market_scarcity_factors: List[str] = Field(default=[], description="Array of factors affecting candidate availability")


class CostBenefitAnalysis(BaseModel):
    total_cost_of_hire: str = Field(description="Estimated total hiring cost")
    expected_productivity_value: str = Field(description="Estimated value creation")
    roi_timeline: str = Field(description="Expected timeline for positive ROI")
    investment_risk_level: str = Field(description="Risk level of hiring investment")


class BusinessImpactAssessment(BaseModel):
    immediate_impact_potential: str = Field(description="Ability to contribute immediately")
    ramp_up_time_estimate: str = Field(description="Estimated time to full productivity")
    long_term_value_creation: str = Field(description="Potential for long-term value creation")
    team_dynamic_impact: str = Field(description="Expected impact on team performance")
    organizational_growth_contribution: str = Field(description="Contribution to company growth")
    cost_benefit_analysis: CostBenefitAnalysis


class AlternativeScenario(BaseModel):
    scenario: str = Field(description="Alternative outcome scenario")
    probability: str = Field(description="Likelihood of this scenario")
    implications: str = Field(description="What this scenario would mean")


class DecisionRationale(BaseModel):
    primary_reasons_for_recommendation: List[str] = Field(default=[], description="Array of main reasons supporting recommendation")
    primary_reasons_against_recommendation: List[str] = Field(default=[], description="Array of main concerns")
    decision_confidence_factors: List[str] = Field(default=[], description="Array of factors supporting confidence level")
    decision_uncertainty_factors: List[str] = Field(default=[], description="Array of factors creating uncertainty")
    alternative_scenarios: List[AlternativeScenario] = Field(default=[])
    key_assumptions: List[str] = Field(default=[], description="Array of assumptions underlying the recommendation")
    sensitivity_analysis: str = Field(description="How sensitive the recommendation is to key factors")


class StakeholderImpactAnalysis(BaseModel):
    hiring_manager_considerations: str = Field(description="Specific points for hiring manager")
    team_impact_assessment: str = Field(description="How this hire affects the team")
    departmental_implications: str = Field(description="Broader departmental considerations")
    organizational_alignment: str = Field(description="Alignment with organizational goals")
    budget_impact: str = Field(description="Financial implications of hiring decision")
    timeline_considerations: str = Field(description="Timing factors affecting the hire")


class QualityAssurance(BaseModel):
    data_completeness_assessment: str = Field(description="Assessment of data quality used")
    evaluation_methodology_validation: str = Field(description="Validation of evaluation approach")
    potential_bias_factors: List[str] = Field(default=[], description="Array of potential biases identified")
    confidence_intervals: str = Field(description="Uncertainty ranges for key assessments")
    recommendation_robustness: str = Field(description="How robust the recommendation is to new information")
    evaluation_limitations: List[str] = Field(default=[], description="Array of limitations in the evaluation")


class ReportGenerationOutput(BaseModel):
    """Complete output model for report generation task"""
    executive_summary: ExecutiveSummary
    candidate_profile: CandidateProfile
    job_requirements_analysis: JobRequirementsAnalysis
    detailed_scoring_analysis: DetailedScoringAnalysis
    strengths_and_differentiators: StrengthsAndDifferentiators
    gaps_and_risk_assessment: GapsAndRiskAssessment
    comparative_analysis: ComparativeAnalysis
    business_impact_assessment: BusinessImpactAssessment
    decision_rationale: DecisionRationale
    stakeholder_impact_analysis: StakeholderImpactAnalysis
    quality_assurance: QualityAssurance


# =============================================================================
# Combined Models for Easy Import
# =============================================================================

__all__ = [
    "DocumentAnalysisOutput",
    "CandidateMatchingOutput", 
    "ReportGenerationOutput",
    # Individual models can be imported if needed
    "ContactInformation", "Education", "WorkExperience", "Skills",
    "ExecutiveSummary", "CandidateProfile", "JobRequirementsAnalysis"
]
