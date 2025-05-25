# Streamlit Cloud Deployment Instructions

## ChromaDB Compatibility Issue

The main application (`src/resume/main.py`) uses CrewAI which depends on ChromaDB. ChromaDB has known compatibility issues with Streamlit Cloud's sandboxed environment.

## Solution: Cloud-Compatible Version

Use the simplified version that bypasses ChromaDB entirely.

### Deployment Steps:

1. **Use the cloud-compatible version**: `streamlit_app.py`
2. **Use simplified requirements**: Copy `requirements_cloud.txt` to `requirements.txt` for deployment
3. **Set environment variables in Streamlit Cloud**:
   - `GEMINI_API_KEY`: Your Google Gemini API key

### Files for Streamlit Cloud Deployment:

```
streamlit_app.py          # Main application file (point Streamlit Cloud here)
requirements_cloud.txt    # Simplified dependencies
runtime.txt              # Python version specification
```

### Key Differences from Full Version:

- ✅ **Works on Streamlit Cloud**: No ChromaDB dependency issues
- ✅ **Direct Gemini API**: Uses google-generativeai directly
- ✅ **Same Core Features**: PDF processing, batch analysis, scoring
- ❌ **No CrewAI Multi-Agent**: Uses single Gemini model instead
- ❌ **Simpler Analysis**: Less sophisticated than multi-agent approach

### Performance:

- **Faster startup**: No ChromaDB initialization
- **More reliable**: Fewer dependencies to fail
- **Good results**: Still provides comprehensive resume analysis

### Usage:

The cloud version works exactly like the full version:
1. Upload PDF resumes
2. Enter job title and description  
3. Click "Analyze All Resumes"
4. View comparative results and individual reports

### Environment Variables Required:

- `GEMINI_API_KEY`: Your Google Gemini API key

That's it! The cloud version should deploy successfully on Streamlit Cloud without ChromaDB issues.
