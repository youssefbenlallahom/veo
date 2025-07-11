#!/usr/bin/env python
"""
FastAPI backend for Recruitee API integration.
This service acts as a proxy between the Streamlit app and Recruitee API,
handling authentication, error handling, and data processing.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, Optional
import requests
import os
import re
import logging
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VEO Recruitee API Proxy",
    description="FastAPI backend for Recruitee API integration with VEO Resume Analyzer",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class JobRequest(BaseModel):
    api_url: HttpUrl
    token: Optional[str] = None

class JobResponse(BaseModel):
    success: bool
    title: Optional[str] = None
    description: Optional[str] = None
    description_html: Optional[str] = None
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

class APITestResponse(BaseModel):
    success: bool
    status_code: Optional[int] = None
    response_preview: Optional[str] = None
    error: Optional[str] = None
    auth_method: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

class RecruiteeAPIHandler:
    """Handler for Recruitee API operations."""
    
    def __init__(self):
        self.default_token = os.getenv('RECRUITEE_API_TOKEN')
        if not self.default_token:
            logger.error("RECRUITEE_API_TOKEN environment variable is not set!")
        else:
            logger.info("Recruitee token loaded from environment")
        self.timeout = 10
    
    def validate_recruitee_url(self, url: str) -> bool:
        """
        Validate if the URL matches Recruitee API format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid Recruitee API URL, False otherwise
        """
        pattern = r'^https://api\.recruitee\.com/c/\d+/offers/\d+$'
        return bool(re.match(pattern, url))
    
    def clean_html_description(self, html_content: str) -> str:
        """
        Clean HTML content to extract plain text description.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Clean text description
        """
        try:
            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', '', html_content)
            
            # Replace HTML entities
            html_entities = {
                '&nbsp;': ' ',
                '&amp;': '&',
                '&lt;': '<',
                '&gt;': '>',
                '&quot;': '"',
                '&#39;': "'",
                '&apos;': "'"
            }
            
            for entity, replacement in html_entities.items():
                clean_text = clean_text.replace(entity, replacement)
            
            # Clean up whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = clean_text.strip()
            
            return clean_text
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML description: {str(e)}")
            return html_content
    
    def try_decode_token(self, token: str) -> list:
        """
        Try different ways to decode/format the token.
        
        Args:
            token: The token to decode
            
        Returns:
            List of possible token values to try
        """
        tokens_to_try = [token]  # Start with original token
        
        # Try base64 decoding
        try:
            decoded_bytes = base64.b64decode(token)
            decoded_str = decoded_bytes.decode('utf-8')
            tokens_to_try.append(decoded_str)
        except:
            pass
        
        # Try removing common prefixes if present
        if token.startswith('Bearer '):
            tokens_to_try.append(token[7:])
        
        return tokens_to_try
    
    def get_headers_options(self, token: str) -> list:
        """
        Get different authentication header options to try.
        
        Args:
            token: Authentication token
            
        Returns:
            List of header dictionaries to try
        """
        possible_tokens = self.try_decode_token(token)
        headers_list = []
        
        for token_variant in possible_tokens:
            # Option 1: Bearer token (recommended by Recruitee docs)
            headers_list.append({
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'VEO-Resume-Analyzer/1.0',
                'Authorization': f'Bearer {token_variant}'
            })
            
            # Option 2: Direct token in Authorization header
            headers_list.append({
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'VEO-Resume-Analyzer/1.0',
                'Authorization': token_variant
            })
        
        # Option 3: No authentication (public endpoint test)
        headers_list.append({
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'VEO-Resume-Analyzer/1.0'
        })
        
        return headers_list
    
    def extract_job_data(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract job title and description from Recruitee API response.
        
        Args:
            api_response: Raw API response
            
        Returns:
            Dictionary with extracted job data
        """
        # Extract the offer object from the response
        offer = api_response.get('offer', {})
        if not offer:
            raise ValueError('No offer data found in API response')
        
        # Extract required fields from the offer object
        title = ''
        description_html = offer.get('description_html', '')
        
        # Check if translations exist and get title from default translation
        translations = offer.get('translations', {})
        if translations:
            # Find default translation
            for lang_code, translation in translations.items():
                if translation.get('default', False):
                    title = translation.get('title', '')
                    break
            
            # If no default found, use the first translation
            if not title:
                first_translation = list(translations.values())[0] if translations else {}
                title = first_translation.get('title', '')
        
        # If still no title, try to extract from adminapp_url or other fields
        if not title:
            adminapp_url = offer.get('adminapp_url', '')
            if adminapp_url:
                # Extract title from URL (e.g., commercial-team-leader-with-english-2-4)
                url_parts = adminapp_url.split('/')
                if url_parts:
                    title_from_url = url_parts[-1].replace('-', ' ').title()
                    title = title_from_url
        
        # Final fallback - use a placeholder if nothing found
        if not title:
            title = f"Job Position ID: {offer.get('id', 'Unknown')}"
        
        if not description_html:
            raise ValueError('Job description not found in API response')
        
        # Clean HTML description
        description_clean = self.clean_html_description(description_html)
        
        return {
            'title': title,
            'description': description_clean,
            'description_html': description_html,
            'offer_data': offer
        }
    
    def fetch_job_data(self, api_url: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch job data from Recruitee API.
        
        Args:
            api_url: Recruitee API URL
            token: Optional custom token, uses default if not provided
            
        Returns:
            Dictionary containing job data or error information
        """
        try:
            # Validate URL format
            if not self.validate_recruitee_url(api_url):
                return {
                    'success': False, 
                    'error': 'Invalid Recruitee API URL format. Expected: https://api.recruitee.com/c/{company_id}/offers/{offer_id}'
                }
            
            # Use provided token or default
            auth_token = token or self.default_token
            logger.info(f"Fetching job data from: {api_url}")
            logger.info(f"Token format: {auth_token[:10]}..." if len(auth_token) > 10 else f"Token: {auth_token}")
            
            # Try different authentication methods
            headers_options = self.get_headers_options(auth_token)
            last_error = None
            successful_method = None
            
            # Try each authentication method
            for i, headers in enumerate(headers_options, 1):
                try:
                    logger.info(f"Trying authentication method {i}/{len(headers_options)}")
                    response = requests.get(api_url, headers=headers, timeout=self.timeout)
                    
                    if response.status_code == 200:
                        successful_method = f"Method {i}"
                        logger.info(f"✅ Authentication method {i} succeeded")
                        break
                    else:
                        logger.warning(f"❌ Authentication method {i} failed with status {response.status_code}: {response.text[:100]}")
                        last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                        
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    logger.warning(f"❌ Authentication method {i} failed: {e}")
                    
                # If this is the last attempt and we still failed
                if i == len(headers_options):
                    error_msg = f"All authentication methods failed. Last error: {last_error}"
                    if "401" in str(last_error):
                        error_msg += "\n\n🔑 AUTHENTICATION ISSUE:"
                        error_msg += "\n1. Your token may be invalid or expired"
                        error_msg += "\n2. Please generate a new token from Recruitee:"
                        error_msg += "\n   - Go to Settings > Apps and plugins > Personal API tokens"
                        error_msg += "\n   - Click '+ New token' to generate a new token"
                        error_msg += "\n   - Update your RECRUITEE_API_TOKEN environment variable"
                        error_msg += f"\n3. Current token format: {auth_token[:10]}..."
                    return {
                        'success': False,
                        'error': error_msg
                    }
            
            # Parse JSON response
            job_data = response.json()
            logger.info("Successfully fetched and parsed job data")
            
            # Extract job information
            extracted_data = self.extract_job_data(job_data)
            
            return {
                'success': True,
                'title': extracted_data['title'],
                'description': extracted_data['description'],
                'description_html': extracted_data['description_html'],
                'debug_info': {
                    'auth_method': successful_method,
                    'response_size': len(str(job_data)),
                    'offer_id': extracted_data['offer_data'].get('id'),
                    'timestamp': datetime.now().isoformat(),
                    'token_format': f"{auth_token[:10]}..." if len(auth_token) > 10 else auth_token
                }
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f'API request failed: {str(e)}'
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except ValueError as e:
            logger.error(f"Data extraction error: {str(e)}")
            return {
                'success': False,
                'error': f'Data extraction failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def test_api_connection(self, api_url: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Test API connection with detailed debugging information.
        
        Args:
            api_url: Recruitee API URL
            token: Optional custom token
            
        Returns:
            Dictionary with test results
        """
        try:
            # Validate URL format
            if not self.validate_recruitee_url(api_url):
                return {
                    'success': False, 
                    'error': 'Invalid Recruitee API URL format'
                }
            
            auth_token = token or self.default_token
            headers_options = self.get_headers_options(auth_token)
            
            test_results = []
            
            # Try each authentication method
            for i, headers in enumerate(headers_options, 1):
                try:
                    response = requests.get(api_url, headers=headers, timeout=self.timeout)
                    
                    # Record result for this method
                    method_result = {
                        'method': i,
                        'status_code': response.status_code,
                        'success': response.status_code == 200,
                        'response_size': len(response.content) if response.content else 0,
                        'headers_used': {k: v for k, v in headers.items() if k != 'Authorization'},
                        'auth_header': headers.get('Authorization', 'None')[:20] + '...' if headers.get('Authorization') else 'None'
                    }
                    
                    # Add preview of response content
                    if response.status_code == 200:
                        try:
                            json_data = response.json()
                            method_result['response_preview'] = str(json_data)[:200] + "..." if len(str(json_data)) > 200 else str(json_data)
                        except:
                            method_result['response_preview'] = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    else:
                        method_result['error'] = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    
                    test_results.append(method_result)
                    
                    # If successful, we can stop here
                    if response.status_code == 200:
                        return {
                            'success': True,
                            'status_code': response.status_code,
                            'auth_method': f"Method {i}",
                            'response_preview': method_result['response_preview'],
                            'debug_info': {
                                'all_methods_tested': test_results,
                                'successful_method': i,
                                'timestamp': datetime.now().isoformat(),
                                'token_info': f"Token format: {auth_token[:10]}..." if len(auth_token) > 10 else auth_token
                            }
                        }
                    
                except requests.exceptions.RequestException as e:
                    method_result = {
                        'method': i,
                        'success': False,
                        'error': str(e),
                        'headers_used': {k: v for k, v in headers.items() if k != 'Authorization'},
                        'auth_header': headers.get('Authorization', 'None')[:20] + '...' if headers.get('Authorization') else 'None'
                    }
                    test_results.append(method_result)
            
            # If we get here, all methods failed
            error_msg = 'All authentication methods failed'
            if any('401' in str(result.get('error', '')) for result in test_results):
                error_msg += '\n\n🔑 TOKEN ISSUE DETECTED:'
                error_msg += '\n1. Generate a new token from Recruitee dashboard'
                error_msg += '\n2. Go to Settings > Apps and plugins > Personal API tokens'
                error_msg += '\n3. Click "+ New token" and copy the generated token'
                error_msg += '\n4. Update your RECRUITEE_API_TOKEN environment variable'
                
            return {
                'success': False,
                'error': error_msg,
                'debug_info': {
                    'all_methods_tested': test_results,
                    'timestamp': datetime.now().isoformat(),
                    'token_info': f"Token format: {auth_token[:10]}..." if len(auth_token) > 10 else auth_token
                }
            }
            
        except Exception as e:
            logger.error(f"API test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }

# Initialize handler
recruitee_handler = RecruiteeAPIHandler()

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VEO Recruitee API Proxy",
        "version": "1.0.0",
        "endpoints": {
            "fetch_job": "/job/fetch",
            "test_api": "/job/test",
            "health": "/health"
        },
        "token_setup": {
            "instructions": "Generate a new token from Recruitee Settings > Apps and plugins > Personal API tokens",
            "format": "Set RECRUITEE_API_TOKEN environment variable with your token"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "VEO Recruitee API Proxy",
        "token_configured": bool(os.getenv('RECRUITEE_API_TOKEN'))
    }

@app.get("/job/fetch", response_model=JobResponse)
async def fetch_job(
    api_url: str = Query(..., description="Recruitee API URL"),
    token: Optional[str] = Query(None, description="Optional custom authentication token")
):
    """
    Fetch job data from Recruitee API.
    
    Args:
        api_url: Recruitee API URL
        token: Optional custom authentication token
        
    Returns:
        Job data including title and description
    """
    try:
        result = recruitee_handler.fetch_job_data(api_url, token)
        
        if result['success']:
            return JobResponse(
                success=True,
                title=result['title'],
                description=result['description'],
                description_html=result['description_html'],
                debug_info=result.get('debug_info')
            )
        else:
            # Return error as HTTP 400 with details
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
            
    except Exception as e:
        logger.error(f"Error in fetch_job endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/job/test", response_model=APITestResponse)
async def test_api(
    api_url: str = Query(..., description="Recruitee API URL to test"),
    token: Optional[str] = Query(None, description="Optional custom authentication token")
):
    """
    Test API connection with detailed debugging information.
    
    Args:
        api_url: Recruitee API URL to test
        token: Optional custom authentication token
        
    Returns:
        Test results with debugging information
    """
    try:
        result = recruitee_handler.test_api_connection(api_url, token)
        
        return APITestResponse(
            success=result['success'],
            status_code=result.get('status_code'),
            response_preview=result.get('response_preview'),
            error=result.get('error'),
            auth_method=result.get('auth_method'),
            debug_info=result.get('debug_info')
        )
        
    except Exception as e:
        logger.error(f"Error in test_api endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/job/fetch", response_model=JobResponse)
async def fetch_job_post(job_request: JobRequest):
    """
    Fetch job data from Recruitee API using POST method.
    
    Args:
        job_request: Job request with API URL and optional token
        
    Returns:
        Job data including title and description
    """
    try:
        result = recruitee_handler.fetch_job_data(str(job_request.api_url), job_request.token)
        
        if result['success']:
            return JobResponse(
                success=True,
                title=result['title'],
                description=result['description'],
                description_html=result['description_html'],
                debug_info=result.get('debug_info')
            )
        else:
            # Return error as HTTP 400 with details
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
            
    except Exception as e:
        logger.error(f"Error in fetch_job_post endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"Starting VEO Recruitee API Proxy on {host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"API Health Check: http://{host}:{port}/health")
    print(f"Token Status: {'✅ Configured' if os.getenv('RECRUITEE_API_TOKEN') else '❌ Not configured'}")
    print("\n🔑 To fix authentication issues:")
    print("1. Go to Recruitee Settings > Apps and plugins > Personal API tokens")
    print("2. Click '+ New token' to generate a new token")
    print("3. Set environment variable: export RECRUITEE_API_TOKEN='your_new_token'")
    
    uvicorn.run(
        "recruitee_proxy:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )