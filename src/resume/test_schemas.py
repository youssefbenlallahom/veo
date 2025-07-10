"""
Test script to verify the Pydantic schemas work correctly.
"""

import json
from schemas import DocumentAnalysisOutput, CandidateMatchingOutput, ReportGenerationOutput
from model_utils import ModelValidator, create_example_data

def test_document_analysis_schema():
    """Test the DocumentAnalysisOutput schema."""
    print("🧪 Testing DocumentAnalysisOutput schema...")
    
    example_data = create_example_data()
    
    try:
        # Test validation
        model = DocumentAnalysisOutput(**example_data)
        print("✅ DocumentAnalysisOutput validation successful!")
        
        # Test JSON serialization
        json_str = model.model_dump_json(indent=2)
        print(f"✅ JSON serialization successful! Length: {len(json_str)} characters")
        
        # Test deserialization
        parsed_data = json.loads(json_str)
        model2 = DocumentAnalysisOutput(**parsed_data)
        print("✅ JSON deserialization successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ DocumentAnalysisOutput test failed: {e}")
        return False

def test_candidate_matching_schema():
    """Test the CandidateMatchingOutput schema with minimal data."""
    print("\n🧪 Testing CandidateMatchingOutput schema...")
    
    example_data = {
        "job_analysis": {
            "job_title": "Software Engineer",
            "job_description": "Looking for a skilled software engineer...",
            "critical_requirements": ["Python", "5+ years experience"],
            "preferred_requirements": ["AWS", "Team leadership"],
            "experience_required": "5+ years",
            "education_required": "Bachelor's degree"
        },
        "candidate_overview": {
            "candidate_name": "John Doe",
            "current_title": "Senior Software Engineer",
            "total_experience_years": "6",
            "relevant_experience_years": "5",
            "education_level": "Bachelor's in Computer Science",
            "key_skills": ["Python", "AWS", "Team Leadership"]
        },
        "rubric_evaluation": [{
            "section_name": "Technical Skills",
            "weight_percentage": "40",
            "candidate_evidence": ["5 years Python experience", "AWS certification"],
            "section_score": "8",
            "score_justification": "Strong technical background with relevant certifications",
            "weighted_score": "3.2"
        }],
        "match_analysis": {
            "skills_match": {
                "matched_skills": ["Python", "AWS"],
                "missing_skills": [],
                "additional_skills": ["Docker", "Kubernetes"],
                "match_percentage": "100"
            },
            "experience_match": {
                "required_experience": "5+ years",
                "candidate_experience": "6 years",
                "experience_gap": "None",
                "match_level": "exceeds"
            },
            "education_match": {
                "required_education": "Bachelor's degree",
                "candidate_education": "Bachelor's in Computer Science",
                "match_level": "meets"
            }
        },
        "scoring_summary": {
            "total_weighted_score": "8.0",
            "normalized_score": "8.0",
            "score_breakdown": [{
                "criterion": "Technical Skills",
                "raw_score": "8",
                "weight": "40",
                "weighted_points": "3.2"
            }],
            "weights_validation": "100%"
        },
        "evaluation_results": {
            "critical_missing_requirements": [],
            "key_strengths": ["Strong technical skills", "Relevant experience"],
            "areas_for_improvement": ["Could improve leadership skills"],
            "unique_value_propositions": ["Cloud architecture expertise"],
            "risk_factors": []
        },
        "final_recommendation": {
            "status": "RECOMMENDED",
            "confidence_level": "high",
            "primary_reason": "Strong technical fit with relevant experience",
            "supporting_factors": ["Meets all requirements", "Exceeds experience threshold"],
            "conditions_if_applicable": [],
            "next_steps": ["Schedule technical interview", "Check references"]
        },
        "detailed_feedback": {
            "for_candidate": "Strong technical profile with excellent experience match",
            "for_hiring_manager": "Excellent candidate with strong technical skills",
            "interview_focus_areas": ["Leadership experience", "Team collaboration"],
            "reference_check_priorities": ["Technical competency", "Team leadership"]
        }
    }
    
    try:
        model = CandidateMatchingOutput(**example_data)
        print("✅ CandidateMatchingOutput validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ CandidateMatchingOutput test failed: {e}")
        return False

def test_report_generation_schema():
    """Test the ReportGenerationOutput schema with minimal data."""
    print("\n🧪 Testing ReportGenerationOutput schema...")
    
    example_data = {
        "executive_summary": {
            "candidate_name": "John Doe",
            "position_applied": "Software Engineer",
            "overall_recommendation": "RECOMMENDED",
            "overall_score": "8.0",
            "confidence_level": "HIGH",
            "key_decision_factors": ["Strong technical skills", "Relevant experience"],
            "critical_concerns": [],
            "recommendation_summary": "Highly qualified candidate with excellent technical skills and relevant experience."
        },
        "candidate_profile": {
            "professional_summary": "Experienced software engineer with strong technical skills and team leadership experience.",
            "career_trajectory": "ascending",
            "total_experience": "6 years",
            "relevant_experience": "5 years",
            "education_level": "Bachelor's in Computer Science",
            "current_status": "employed",
            "geographic_considerations": "Willing to relocate",
            "career_progression_analysis": "Steady upward progression with increasing responsibilities",
            "industry_experience": ["Technology", "Software Development"],
            "company_sizes_worked": ["startup", "mid-size"]
        },
        "job_requirements_analysis": {
            "must_have_requirements": [{
                "requirement": "Python programming",
                "candidate_match": "EXCEEDS",
                "evidence": "5+ years Python experience",
                "gap_analysis": "No gap identified",
                "risk_level": "LOW",
                "impact_on_role_success": "Critical for success"
            }],
            "preferred_requirements": [{
                "requirement": "AWS experience",
                "candidate_match": "MEETS",
                "evidence": "AWS certification",
                "added_value": "Enables cloud architecture decisions"
            }],
            "requirements_satisfaction_score": "95",
            "critical_missing_requirements": [],
            "exceeds_expectations_in": ["Technical skills", "Experience level"]
        },
        "detailed_scoring_analysis": {
            "rubric_breakdown": [{
                "criterion": "Technical Skills",
                "raw_score": "8",
                "weight_percentage": "40",
                "weighted_points": "3.2",
                "performance_level": "EXCELLENT",
                "supporting_evidence": ["5+ years Python", "AWS certification"],
                "score_justification": "Strong technical background",
                "benchmark_comparison": "Above average"
            }],
            "scoring_methodology": "Weighted scoring based on rubric criteria",
            "score_distribution": {
                "technical_skills": "8.0",
                "experience_relevance": "8.5",
                "educational_background": "7.0",
                "achievements_impact": "8.0",
                "skill_depth_breadth": "8.0"
            },
            "total_weighted_score": "8.0",
            "percentile_ranking": "85th percentile",
            "score_reliability": "High confidence"
        },
        "strengths_and_differentiators": {
            "core_strengths": [{
                "strength": "Technical expertise",
                "evidence": "5+ years Python experience",
                "business_impact": "Enables rapid development",
                "uniqueness_factor": "Above market average",
                "quantified_impact": "Improved system performance by 30%"
            }],
            "unique_value_propositions": [{
                "value_proposition": "Cloud architecture expertise",
                "market_advantage": "Competitive advantage in cloud-first environment",
                "differentiation_level": "High",
                "strategic_value": "Enables digital transformation initiatives"
            }],
            "standout_achievements": [{
                "achievement": "Led team of 5 developers",
                "quantified_impact": "Improved team productivity by 25%",
                "relevance_to_role": "Directly applicable to leadership requirements",
                "skill_demonstration": "Leadership and management skills",
                "complexity_level": "High"
            }],
            "leadership_indicators": ["Team leadership", "Mentoring junior developers"],
            "innovation_examples": ["Implemented new CI/CD pipeline", "Introduced automated testing"]
        },
        "gaps_and_risk_assessment": {
            "critical_gaps": [],
            "experience_gaps": [],
            "skill_deficiencies": [],
            "performance_risk_factors": [],
            "cultural_fit_assessment": {
                "cultural_alignment_score": "8",
                "cultural_strengths": ["Team collaboration", "Growth mindset"],
                "cultural_concerns": [],
                "team_integration_assessment": "High likelihood of successful integration"
            }
        },
        "comparative_analysis": {
            "market_positioning": "Above average candidate",
            "salary_market_alignment": "Aligned with market standards",
            "competitive_advantage": "Strong technical skills and experience",
            "alternative_candidates_consideration": "Better than 80% of candidates",
            "urgency_vs_quality_tradeoff": "High quality candidate worth waiting for",
            "market_scarcity_factors": ["High demand for Python developers"]
        },
        "business_impact_assessment": {
            "immediate_impact_potential": "High - can contribute immediately",
            "ramp_up_time_estimate": "2-3 weeks",
            "long_term_value_creation": "High potential for long-term contributions",
            "team_dynamic_impact": "Positive impact on team performance",
            "organizational_growth_contribution": "Significant contribution to growth",
            "cost_benefit_analysis": {
                "total_cost_of_hire": "$15,000",
                "expected_productivity_value": "$200,000 annually",
                "roi_timeline": "3-6 months",
                "investment_risk_level": "LOW"
            }
        },
        "decision_rationale": {
            "primary_reasons_for_recommendation": ["Strong technical fit", "Relevant experience"],
            "primary_reasons_against_recommendation": [],
            "decision_confidence_factors": ["Clear evidence of skills", "Strong track record"],
            "decision_uncertainty_factors": [],
            "alternative_scenarios": [{
                "scenario": "Candidate accepts offer",
                "probability": "High",
                "implications": "Positive impact on team and projects"
            }],
            "key_assumptions": ["Candidate is available", "Salary expectations are reasonable"],
            "sensitivity_analysis": "Recommendation is robust to minor variations"
        },
        "stakeholder_impact_analysis": {
            "hiring_manager_considerations": "Excellent technical fit",
            "team_impact_assessment": "Will strengthen team capabilities",
            "departmental_implications": "Supports department growth goals",
            "organizational_alignment": "Aligns with company technical strategy",
            "budget_impact": "Within approved budget range",
            "timeline_considerations": "Can start within 2 weeks"
        },
        "quality_assurance": {
            "data_completeness_assessment": "Comprehensive data available",
            "evaluation_methodology_validation": "Rigorous evaluation process followed",
            "potential_bias_factors": ["No significant bias identified"],
            "confidence_intervals": "High confidence in assessment",
            "recommendation_robustness": "Recommendation is robust",
            "evaluation_limitations": ["Limited to resume information only"]
        }
    }
    
    try:
        model = ReportGenerationOutput(**example_data)
        print("✅ ReportGenerationOutput validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ ReportGenerationOutput test failed: {e}")
        return False

def test_model_validator():
    """Test the ModelValidator utility."""
    print("\n🧪 Testing ModelValidator...")
    
    example_data = create_example_data()
    
    try:
        # Test validation
        result = ModelValidator.validate_dict(example_data, 'document_analysis')
        
        if isinstance(result, DocumentAnalysisOutput):
            print("✅ ModelValidator validation successful!")
            
            # Test conversion to dict
            dict_result = ModelValidator.model_to_dict(result)
            print(f"✅ Model to dict conversion successful! Keys: {len(dict_result)}")
            
            # Test JSON conversion
            json_result = ModelValidator.model_to_json(result)
            print(f"✅ Model to JSON conversion successful! Length: {len(json_result)}")
            
            return True
        else:
            print(f"❌ ModelValidator validation failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ModelValidator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Pydantic Schema Tests...")
    
    tests = [
        test_document_analysis_schema,
        test_candidate_matching_schema,
        test_report_generation_schema,
        test_model_validator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Schemas are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
