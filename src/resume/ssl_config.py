"""
SSL Configuration for Corporate Networks
This module handles SSL certificate issues common in corporate environments
"""
import os
import ssl
import warnings
import urllib3

def disable_ssl_verification():
    """
    Disable SSL verification for corporate networks with self-signed certificates
    """
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    # Set environment variables to disable SSL verification
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_VERIFY'] = 'false'
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # Configure requests to not verify SSL
    import requests
    requests.packages.urllib3.disable_warnings()
    
    # Create a monkey patch for requests
    original_request = requests.Session.request
    
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    
    requests.Session.request = patched_request
    
    print("🔓 SSL verification disabled for corporate network compatibility")

def configure_ssl_context():
    """
    Create a custom SSL context that doesn't verify certificates
    """
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context
    except Exception as e:
        print(f"Warning: Could not configure SSL context: {e}")
        return None

# Auto-apply SSL fixes when module is imported
disable_ssl_verification()
