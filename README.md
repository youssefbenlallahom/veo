# VEO Resume Analyzer 📄✨

An intelligent resume screening and analysis system powered by CrewAI and Google Gemini. This tool helps HR professionals and recruiters efficiently evaluate multiple resumes against job descriptions using AI-powered analysis.

## 🚀 Features

- **Batch Resume Analysis**: Upload and analyze multiple PDF resumes simultaneously
- **AI-Powered Scoring**: Automatic scoring based on job requirements using Google Gemini
- **Smart Barème Generation**: Automatically generates evaluation criteria from job descriptions
- **Comparative Analysis**: Side-by-side comparison of candidates with rankings
- **Detailed Reports**: Comprehensive analysis with strengths, gaps, and recommendations
- **PDF Validation**: Built-in PDF validation and text extraction
- **Interactive UI**: Clean Streamlit interface for easy use

## 🛠️ Technology Stack

- **AI Framework**: CrewAI for multi-agent orchestration
- **LLM**: Google Gemini 2.0 Flash for intelligent analysis
- **Frontend**: Streamlit for web interface
- **PDF Processing**: PyPDF2 and pdfplumber for text extraction
- **Package Management**: UV for fast dependency management

## 📋 Prerequisites

- Python 3.10-3.12
- Google Gemini API key
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
   - View the comparative ranking table
   - Expand individual reports for detailed analysis
   - Check scores, strengths, and improvement areas

## 🏗️ Project Structure

```
resume/
├── src/resume/
│   ├── main.py              # Streamlit application entry point
│   ├── crew.py              # CrewAI crew configuration
│   ├── report_schema.py     # Report structure definition
│   ├── config/
│   │   ├── agents.yaml      # AI agent configurations
│   │   └── tasks.yaml       # Task definitions
│   └── tools/
│       └── custom_tool.py   # PDF processing tools
├── requirements.txt         # Project dependencies
├── pyproject.toml          # Project configuration
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

## 🤖 AI Agents

The system uses specialized AI agents:

- **Document Analyzer**: Extracts and structures resume data from PDFs
- **Matching Specialist**: Compares resumes against job requirements
- **Scoring Agent**: Provides numerical scoring based on criteria
- **Report Generator**: Creates comprehensive analysis reports

## 📊 Scoring System

- **Automatic Barème**: Generates evaluation criteria from job descriptions
- **Weighted Scoring**: Different aspects weighted by importance
- **0-10 Scale**: Easy to understand scoring system
- **Bonus Points**: Recognition for exceptional qualifications

## 🔧 Configuration

### Agent Configuration (`config/agents.yaml`)
Define AI agent roles, goals, and constraints

### Task Configuration (`config/tasks.yaml`)
Specify task workflows and expected outputs

### Environment Variables
- `GEMINI_API_KEY`: Required for Google Gemini API access

## 📈 Output Files

The system generates several files during analysis:
- `report_[candidate_name].md`: Individual analysis reports
- `extracted_[candidate_name].txt`: Raw PDF text extraction
- `barem_gemini.json`: Generated evaluation criteria (cached)

## 🚀 Advanced Usage

### Custom PDF Processing
The system includes robust PDF processing with:
- Multiple extraction methods (PyPDF2 + pdfplumber)
- Text cleaning and sanitization
- Table delimiter removal
- Error handling for corrupted files

### Batch Processing
- Processes multiple resumes efficiently
- Real-time progress tracking
- Handles failures gracefully
- Generates unique reports per candidate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- Ensure PDF files are readable and contain extractable text
- The system works best with well-structured resumes
- API costs apply for Google Gemini usage
- Keep your API keys secure and never commit them to version control

## 🆘 Troubleshooting

### Common Issues

1. **PDF Extraction Fails**: Ensure PDFs contain selectable text, not just images
2. **API Errors**: Check your Gemini API key and quota
3. **Memory Issues**: For large batches, process in smaller groups
4. **Import Errors**: Ensure all dependencies are installed correctly

### Support

For issues and questions:
- Check the troubleshooting section above
- Review error messages in the Streamlit interface
- Ensure all environment variables are set correctly

---

**VEO Resume Analyzer** - Streamlining talent acquisition with AI-powered resume analysis. 🎯
