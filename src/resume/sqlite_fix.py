"""
SQLite compatibility fix for Streamlit Cloud.
This MUST be imported before any other modules that use ChromaDB.
"""
import sys
import os

def apply_sqlite_fix():
    """Apply SQLite compatibility fixes for Streamlit Cloud and ChromaDB."""
    
    # Method 1: Try to use pysqlite3-binary
    try:
        import pysqlite3
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("✅ Successfully replaced sqlite3 with pysqlite3-binary")
        return True
    except ImportError:
        print("⚠️ pysqlite3-binary not available")
    
    # Method 2: Set ChromaDB environment variables to use alternative backend
    os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["ALLOW_RESET"] = "True"
    
    # Method 3: Disable ChromaDB entirely for CrewAI if needed
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["CREWAI_STORAGE_DIR"] = ""
    
    # Method 4: Try to mock problematic ChromaDB components
    try:
        import sqlite3
        version = sqlite3.sqlite_version_info
        if version < (3, 35, 0):
            print(f"⚠️ SQLite version {sqlite3.sqlite_version} is too old for ChromaDB")
            # Try to disable ChromaDB features that require newer SQLite
            os.environ["CHROMA_DISABLE"] = "true"
    except Exception as e:
        print(f"SQLite version check failed: {e}")
    
    print("🔧 Applied ChromaDB compatibility settings")
    return False

# Apply the fix immediately when this module is imported
apply_sqlite_fix()