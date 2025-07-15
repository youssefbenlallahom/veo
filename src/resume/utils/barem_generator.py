import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from google import genai
from google.genai import types


class BaremGenerator:
    """Generates and manages scoring rubrics (barem) for job positions."""
    
    def __init__(self, output_dir: Path = None):
        # Store reference to temporary file for cleanup
        self.temp_file = None
    
    def get_barem(self, job_title: str, job_description: str) -> Optional[Dict]:
        """Generate barem and save as temporary file."""
        # Always generate new barem
        barem = self._generate_barem_from_gemini(job_title, job_description)
        
        # Save as temporary file
        if barem:
            try:
                # Create a temporary file with .json extension
                self.temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.json', 
                    prefix=f'barem_{self._sanitize_job_title(job_title)}_',
                    delete=False,  # Don't delete automatically, we'll handle cleanup
                    encoding='utf-8'
                )
                
                json.dump(barem, self.temp_file, ensure_ascii=False, indent=2)
                self.temp_file.close()
                
                print(f"Temporary barem saved to: {self.temp_file.name}")
            except Exception as e:
                print(f"Warning: Could not save temporary barem file: {e}")
        
        return barem
    
    def cleanup(self):
        """Clean up temporary barem file."""
        if self.temp_file and os.path.exists(self.temp_file.name):
            try:
                os.unlink(self.temp_file.name)
                print(f"Cleaned up temporary barem file: {self.temp_file.name}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary file: {e}")
            finally:
                self.temp_file = None
    
    def _sanitize_job_title(self, job_title: str) -> str:
        """Sanitize job title for filename."""
        import re
        return re.sub(r'[^\w\-_\. ]', '_', job_title.lower())
    
    def _generate_barem_from_gemini(self, job_title: str, job_description: str) -> Optional[Dict]:
        """Generate barem using Gemini API with retry logic."""
        import time
        import random
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables")
                
                client = genai.Client(api_key=api_key)
                
                prompt = f"""
                You are a structured hiring evaluation assistant.
                Generate a resume evaluation barème (scoring rubric) out of 100 points for:
                
                Job Title: {job_title}
                Job Description: {job_description}
                
                Return only a JSON object with the rubric structure containing 6 sections:
                1. Relevant Work Experience (30 points)
                2. Skills and Technical Expertise (25 points)
                3. Educational Background and Certifications (15 points)
                4. Achievements and Impact (15 points)
                5. Soft Skills and Cultural Fit (10 points)
                6. Bonus (5 points)
                
                Each section should have scoring criteria for full_points, partial_points, and zero_points.
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        top_p=1.0,
                    ),
                )
                
                # Extract JSON from response
                import re
                match = re.search(r'\{.*\}', response.text or "", re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                
                return None
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error generating barem (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check if it's a 503 error (rate limiting)
                if "503" in error_msg or "overloaded" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit, waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                        continue
                
                # For other errors, don't retry
                if attempt == max_retries - 1:
                    print(f"Failed to generate barem after {max_retries} attempts")
                    return None
                
        return None

