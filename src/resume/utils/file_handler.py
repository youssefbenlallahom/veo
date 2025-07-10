import os
import re
import tempfile
from pathlib import Path
from typing import Any


class FileHandler:
    """Handles file operations for the resume analysis project."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove problematic characters."""
        return re.sub(r'[^\w\-_\. ]', '_', filename)
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary files safely."""
        try:
            if os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
                os.remove(file_path)
        except Exception:
            pass  # Silent cleanup

