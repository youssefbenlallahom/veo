import os
import re
import json
from typing import Dict, List, Any


class ReportParser:
    """Parses analysis reports to extract structured data."""
    
    def parse_report(self, report_path: str, candidate_name: str) -> Dict[str, Any]:
        """Parse the report file to extract scores and recommendations."""
        result = {
            "candidate_name": candidate_name,
            "valid": True,
            "error": "",
            "score": 0,
            "recommendation": "Unknown",
            "strengths": [],
            "gaps": [],
            "report_content": ""
        }
        
        if not os.path.exists(report_path):
            result["valid"] = False
            result["error"] = "Report file not found"
            return result
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            result["report_content"] = report_content
            
            # Try to parse as JSON first (new format)
            try:
                json_data = json.loads(report_content)
                return self._parse_json_report(json_data, candidate_name)
            except json.JSONDecodeError:
                # Fall back to markdown parsing (old format)
                return self._parse_markdown_report(report_content, candidate_name)
            
        except Exception as e:
            result["valid"] = False
            result["error"] = f"Error parsing report: {str(e)}"
            return result
    
    def _parse_json_report(self, json_data: Dict[str, Any], candidate_name: str) -> Dict[str, Any]:
        """Parse JSON format report (new structure)."""
        result = {
            "candidate_name": candidate_name,
            "valid": True,
            "error": "",
            "score": 0,
            "recommendation": "Unknown",
            "strengths": [],
            "gaps": [],
            "report_content": json.dumps(json_data, indent=2)
        }
        
        # Extract executive summary information
        executive_summary = json_data.get('executive_summary', {})
        result["candidate_name"] = executive_summary.get('candidate_name', candidate_name)
        result["score"] = executive_summary.get('overall_score', 0)
        result["recommendation"] = executive_summary.get('overall_recommendation', 'Unknown')
        
        # Extract strengths from the new structure
        strengths_data = json_data.get('strengths_and_differentiators', {})
        core_strengths = strengths_data.get('core_strengths', [])
        unique_values = strengths_data.get('unique_value_propositions', [])
        
        strengths = []
        for strength in core_strengths:
            if isinstance(strength, dict):
                strengths.append(strength.get('strength', ''))
            else:
                strengths.append(str(strength))
        
        for value in unique_values:
            if isinstance(value, dict):
                strengths.append(value.get('value_proposition', ''))
            else:
                strengths.append(str(value))
        
        result["strengths"] = [s for s in strengths if s]
        
        # Extract gaps from the new structure
        gaps_data = json_data.get('gaps_and_risk_assessment', {})
        critical_gaps = gaps_data.get('critical_gaps', [])
        risk_factors = gaps_data.get('performance_risk_factors', [])
        
        gaps = []
        for gap in critical_gaps:
            if isinstance(gap, dict):
                gaps.append(gap.get('gap', ''))
            else:
                gaps.append(str(gap))
        
        for risk in risk_factors:
            if isinstance(risk, dict):
                gaps.append(risk.get('risk', ''))
            else:
                gaps.append(str(risk))
        
        result["gaps"] = [g for g in gaps if g]
        
        # Add the full JSON data to the result for comprehensive access
        result.update(json_data)
        
        return result
    
    def _parse_markdown_report(self, content: str, candidate_name: str) -> Dict[str, Any]:
        """Parse markdown format report (old structure)."""
        result = {
            "candidate_name": candidate_name,
            "valid": True,
            "error": "",
            "score": 0,
            "recommendation": "Unknown",
            "strengths": [],
            "gaps": [],
            "report_content": content
        }
        
        # Extract score using multiple patterns
        score = self._extract_score(content)
        result["score"] = score
        
        # Extract recommendation
        recommendation = self._extract_recommendation(content)
        result["recommendation"] = recommendation
        
        # Extract strengths and gaps
        result["strengths"] = self._extract_strengths(content)
        result["gaps"] = self._extract_gaps(content)
        
        return result
    
    def _extract_score(self, content: str) -> float:
        """Extract numeric score from report content."""
        # Multiple patterns for score extraction
        patterns = [
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*10\]?',
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\s*/\s*100\]?',
            r'(?:\*\*Overall Score:\*\*|Overall Score:|TOTAL WEIGHTED SCORE:)\s*\[?([\d\.]+)\]?(?!\s*/)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Convert from /100 to /10 scale if needed
                if '/100' in pattern:
                    score = score / 10
                return score
        
        return 0.0
    
    def _extract_recommendation(self, content: str) -> str:
        """Extract recommendation from report content."""
        match = re.search(r'\*\*Decision:\*\* ([^\n]+)', content)
        return match.group(1).strip() if match else "Unknown"
    
    def _extract_strengths(self, content: str) -> List[str]:
        """Extract strengths from report content."""
        match = re.search(r'### ✅ Strengths\n([\s\S]+?)### ❌ Gaps', content)
        if match:
            strengths = match.group(1).strip().split('\n')
            return [s.strip('- ').strip() for s in strengths if s.strip()]
        return []
    
    def _extract_gaps(self, content: str) -> List[str]:
        """Extract gaps from report content."""
        match = re.search(r'### ❌ Gaps\n([\s\S]+?)## Weighted Score Analysis', content)
        if match:
            gaps = match.group(1).strip().split('\n')
            return [g.strip('- ').strip() for g in gaps if g.strip()]
        return []