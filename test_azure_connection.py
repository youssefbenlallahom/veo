#!/usr/bin/env python3
"""
Test Azure AI connection and model configuration
"""
import os
import sys
from dotenv import load_dotenv
from crewai.llm import LLM

# Import SSL configuration
try:
    from src.resume.ssl_config import disable_ssl_verification
    disable_ssl_verification()
except ImportError:
    # Fallback SSL handling
    import warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_VERIFY'] = 'false'

# Load environment variables
load_dotenv()

def test_azure_connection():
    """Test Azure AI connection with current configuration"""
    print("Testing Azure AI connection...")
    
    # Get credentials
    model = os.getenv("model")
    api_key = os.getenv("AZURE_AI_API_KEY")
    base_url = os.getenv("AZURE_AI_ENDPOINT")
    api_version = os.getenv("AZURE_AI_API_VERSION")
    
    print(f"Model: {model}")
    print(f"API Key: {api_key[:10]}..." if api_key else "No API Key")
    print(f"Base URL: {base_url}")
    print(f"API Version: {api_version}")
    
    if not all([model, api_key, base_url]):
        print("❌ Missing required Azure credentials")
        return False
    
    # Test different model name formats
    base_model = model.replace("azure_ai/", "").replace("azure/", "")
    model_formats = [
        f"azure_ai/{base_model}",  # Azure AI Inference format
        f"azure/{base_model}",     # Azure OpenAI format
        base_model,                # Raw model name
        f"azure_ai/{base_model.lower()}",  # Lowercase version
    ]
    
    for test_model in model_formats:
        print(f"\n🔧 Testing model format: '{test_model}'")
        try:
            llm = LLM(
                model=test_model,
                api_key=api_key,
                base_url=base_url,
                api_version=api_version,
                temperature=0.0,
                stream=False,
            )
            
            # Test a simple call
            response = llm.call([{"role": "user", "content": "Say 'Hello, Azure!' and nothing else."}])
            print(f"✅ Success with model '{test_model}'")
            print(f"Response: {response}")
            return True
            
        except Exception as e:
            print(f"❌ Failed with model '{test_model}': {str(e)}")
            continue
    
    return False

def test_gemini_fallback():
    """Test Gemini as fallback"""
    print("\n🔄 Testing Gemini fallback...")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ No Gemini API key found")
        return False
    
    try:
        # Try using CrewAI LLM with Gemini
        llm = LLM(
            model="gemini/gemini-2.5-pro",
            api_key=gemini_key,
            temperature=0.0,
            stream=False,
        )
        
        response = llm.call([{"role": "user", "content": "Say 'Hello, Gemini!' and nothing else."}])
        print(f"✅ Gemini fallback works via CrewAI")
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini via CrewAI failed: {str(e)}")
        
        # Try direct Google GenAI
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents="Say 'Hello, Gemini!' and nothing else."
            )
            print(f"✅ Gemini fallback works via direct API")
            print(f"Response: {response.text}")
            return True
        except Exception as e2:
            print(f"❌ Direct Gemini also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    print("🚀 Azure AI Connection Test\n")
    
    azure_works = test_azure_connection()
    
    if not azure_works:
        print("\n⚠️  Azure AI failed, testing fallback...")
        gemini_works = test_gemini_fallback()
        
        if gemini_works:
            print("\n✅ System will use Gemini as fallback")
        else:
            print("\n❌ Both Azure and Gemini failed")
            print("Consider checking your credentials or using keyword-based extraction")
    else:
        print("\n🎉 Azure AI connection successful!")
