# VEO Resume Analyzer 📄✨

An intelligent resume screening and analysis system powered by CrewAI and Google Gemini. This tool helps HR professionals and recruiters efficiently evaluate multiple resumes against job descriptions using AI-powered analysis with automatic scoring rubric generation.

## 🚀 Features

- **Batch Resume Analysis**: Upload and analyze multiple PDF resumes simultaneously
- **Multi-LLM AI System**: Uses Google Gemini 2.0 Flash and Groq Gemma2-9b for comprehensive analysis
- **Smart Barème Generation**: Automatically generates evaluation criteria from job descriptions using Gemini
- **Comparative Analysis**: Side-by-side comparison of candidates with rankings
- **Detailed Reports**: Comprehensive analysis with strengths, gaps, and weighted scoring
- **PDF Validation**: Built-in PDF validation and robust text extraction
- **Interactive UI**: Clean Streamlit interface for easy use
- **Progress Tracking**: Real-time progress updates during batch processing

## 🛠️ Technology Stack

- **AI Framework**: CrewAI for multi-agent orchestration
- **LLMs**: 
  - Google Gemini 2.0 Flash (primary analysis and barème generation)
  - Groq Gemma2-9b (report generation)
- **Frontend**: Streamlit for web interface
- **PDF Processing**: PyPDF2 and pdfplumber for robust text extraction
- **Package Management**: UV for fast dependency management

## 📋 Prerequisites

- Python 3.10-3.12
- Google Gemini API key
- Groq API key (for report generation)
- UV package manager (recommended) or pip

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd resume
```

### 2. Set Up Environment
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
# Using UV
uv run streamlit run src/resume/main.py

# Or using pip
streamlit run src/resume/main.py
```

## 📖 How to Use

1. **Launch the Application**: Open your browser to the Streamlit URL (typically `http://localhost:8501`)

2. **Upload Resumes**: Use the file uploader to select multiple PDF resumes

3. **Enter Job Details**: 
   - Provide the job title
   - Paste the complete job description

4. **Analyze**: Click "Analyze All Resumes" to start the process

5. **Review Results**: 
   - View the comparative ranking table with scores and status
   - Expand individual reports for detailed analysis
   - Check scores, strengths, and improvement areas

## 🏗️ Project Structure

```
resume/
├── src/resume/
│   ├── main.py              # Streamlit application entry point
│   ├── crew.py              # CrewAI crew configuration
│   ├── report_schema.py     # Pydantic report structure definition
│   ├── config/
│   │   ├── agents.yaml      # AI agent configurations (3 agents)
│   │   └── tasks.yaml       # Task definitions and workflows
│   └── tools/
│       └── custom_tool.py   # PDF processing tools
├── requirements.txt         # Project dependencies
├── pyproject.toml          # Project configuration
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## 🤖 AI Agents (3 Specialized Agents)

The system uses **3 specialized AI agents** working in sequence:

1. **Document Analyzer**: 
   - Extracts and structures resume data from PDFs using custom PDF tools
   - Handles contact info, education, experience, skills, certifications
   - Calculates experience years and identifies employment gaps

2. **Matching Specialist**: 
   - Compares resumes against job requirements using provided scoring rubric
   - Uses the auto-generated barème for weighted evaluation
   - Provides detailed scoring breakdown per criteria

3. **Report Generator**: 
   - Creates comprehensive analysis reports using Pydantic schema
   - Combines all analysis into final hiring recommendations
   - Outputs structured markdown reports

## 📊 Scoring System

- **Automatic Barème Generation**: Uses Gemini to create evaluation criteria from job descriptions
- **Weighted Scoring**: Different aspects weighted by importance (totaling 100%)
- **0-10 Scale**: Easy to understand scoring system with weighted calculations
- **Bonus Points**: Recognition for exceptional qualifications (up to 5 points)
- **JSON Caching**: Generated barème is cached for consistent evaluation across candidates

## 🔧 Configuration

### Agent Configuration (`config/agents.yaml`)
- Defines 3 AI agent roles, goals, and constraints
- Specifies tool usage and validation requirements

### Task Configuration (`config/tasks.yaml`)
- Defines 3 sequential tasks: document analysis, matching, and report generation
- Specifies expected outputs and markdown formatting

### Environment Variables Required
- `GEMINI_API_KEY`: Required for Google Gemini API access (barème generation and analysis)
- `GROQ_API_KEY`: Required for Groq API access (report generation with Gemma2-9b)

## 📈 Output Files

The system generates several files during analysis:
- `report_[candidate_name].md`: Individual analysis reports
- `extracted_[candidate_name].txt`: Raw PDF text extraction
- `barem_gemini.json`: Generated evaluation criteria (cached for consistency)
- `document_analysis_task.md`: Detailed document extraction results

## 🚀 Advanced Usage

### Custom PDF Processing
The system includes robust PDF processing with:
- Multiple extraction methods (PyPDF2 + pdfplumber fallback)
- Text cleaning and sanitization for display
- Table delimiter removal and formatting cleanup
- Comprehensive error handling for corrupted files

### Batch Processing Features
- Processes multiple resumes efficiently with progress tracking
- Real-time comparative table updates
- Handles PDF validation failures gracefully
- Generates unique reports per candidate with sanitized filenames
- Automatic cleanup of temporary files

### Multi-LLM Architecture
- **Gemini 2.0 Flash**: Used for document analysis and matching (low temperature for consistency)
- **Groq Gemma2-9b**: Used for report generation (optimized for structured output)
- **Temperature Control**: Set to 0.1 for consistent, reliable results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- **API Requirements**: Both Gemini and Groq API keys are required for full functionality
- **PDF Requirements**: Ensure PDF files contain extractable text (not scanned images)
- **API Costs**: Usage costs apply for both Google Gemini and Groq APIs
- **Security**: Keep your API keys secure and never commit them to version control
- **Consistency**: The system caches barème generation to ensure consistent evaluation across candidates

## 🆘 Troubleshooting

### Common Issues

1. **PDF Extraction Fails**: 
   - Ensure PDFs contain selectable text, not just images
   - Check for file corruption or password protection

2. **API Errors**: 
   - Verify both Gemini and Groq API keys are correctly set
   - Check API quotas and billing status
   - Ensure environment variables are loaded properly

3. **Memory Issues**: 
   - For large batches, process in smaller groups
   - Monitor system resources during processing

4. **Import Errors**: 
   - Ensure all dependencies are installed correctly
   - Check Python version compatibility (3.10-3.12)
   - Verify CrewAI and related packages are up to date

5. **Scoring Inconsistencies**:
   - Delete `barem_gemini.json` to regenerate evaluation criteria
   - Ensure job descriptions are detailed and specific

### Debug Features

- The system includes extensive debug logging for PDF processing
- Progress tracking shows real-time processing status
- Error messages are displayed in the Streamlit interface
- Failed analyses are clearly marked in the results table

---

**VEO Resume Analyzer** - Streamlining talent acquisition with AI-powered resume analysis and intelligent scoring. 🎯
