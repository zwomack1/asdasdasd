import os
import sys
import requests
import time
from utils.run_logger import get_logger

# Initialize logger
logger = get_logger("openrouter_diagnostic")

logger.log("Starting comprehensive OpenRouter diagnostic")

print("=== OpenRouter Diagnostic Test ===")
print("This script will test all aspects of OpenRouter integration")
print("--------------------------------------------------------")

# 1. Environment check
logger.log("\n[1/5] Environment Check:")
print("\n[1/5] Environment Check:")
print(f"Python Path: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
HRM_PROVIDER = os.getenv('HRM_PROVIDER')
logger.log(f"OPENROUTER_API_KEY: {'*****' + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else 'Not set'}")
logger.log(f"HRM_PROVIDER: {HRM_PROVIDER or 'Not set'}")
print(f"OPENROUTER_API_KEY: {'*****' + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else 'not set'}")
print(f"HRM_PROVIDER: {HRM_PROVIDER or 'not set'}")

if not OPENROUTER_API_KEY:
    logger.log("OPENROUTER_API_KEY environment variable is not set", level="error")
    sys.exit(1)

# 2. Network connectivity
logger.log("\n[2/5] Network Test:")
print("\n[2/5] Network Test:")
try:
    print("Testing connection to openrouter.ai...")
    response = requests.get('https://openrouter.ai', timeout=5)
    logger.log(f"Connectivity to openrouter.ai: Status {response.status_code}")
    print(f"OpenRouter website accessible: {response.status_code == 200}")
    
    print("Testing connection to api.openrouter.ai...")
    response = requests.get('https://api.openrouter.ai', timeout=5)
    logger.log(f"Connectivity to api.openrouter.ai: Status {response.status_code}")
    print(f"OpenRouter API accessible: {response.status_code == 200}")
except Exception as e:
    logger.log(f"Network test failed: {str(e)}", level="error")
    print(f"Network test failed: {str(e)}")

# 3. API key validation
logger.log("\n[3/5] API Key Validation:")
print("\n[3/5] API Key Validation:")
try:
    api_key = os.getenv('OPENROUTER_API_KEY', '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')
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10)
    logger.log(f"API Key Valid: Status {response.status_code}")
    print(f"API Key Valid: {response.status_code == 200}")
    if response.status_code == 200:
        logger.log(f"Key details: {response.json()}")
        print(f"Key details: {response.json()}")
    else:
        logger.log(f"Error: {response.text}")
        print(f"Error: {response.text}")
except Exception as e:
    logger.log(f"API test failed: {str(e)}", level="error")
    print(f"API test failed: {str(e)}")

# 4. Direct API test
logger.log("\n[4/5] Direct API Request:")
print("\n[4/5] Direct API Request:")
try:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/WindsurfAI/HRM-Gemini",
        "X-Title": "HRM-Gemini AI"
    }
    data = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }
    
    print("Sending request to OpenRouter API...")
    start_time = time.time()
    response = requests.post(url, json=data, headers=headers, timeout=30)
    latency = time.time() - start_time
    
    logger.log(f"Status Code: {response.status_code}")
    print(f"Status Code: {response.status_code}")
    logger.log(f"Response Time: {latency:.2f} seconds")
    print(f"Response Time: {latency:.2f} seconds")
    if response.status_code == 200:
        logger.log(f"Response: {response.json()['choices'][0]['message']['content']}")
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        logger.log(f"Error: {response.text[:200]}")
        print(f"Error: {response.text[:200]}")
except Exception as e:
    logger.log(f"Direct API test failed: {str(e)}", level="error")
    print(f"Direct API test failed: {str(e)}")

# 5. Service integration test
logger.log("\n[5/5] Service Integration Test:")
print("\n[5/5] Service Integration Test:")
try:
    sys.path.append(os.getcwd())
    from services.openrouter_service import OpenRouterService
    
    print("Initializing OpenRouterService...")
    service = OpenRouterService()
    print("Service initialized successfully")
    
    print("Testing chat response...")
    response, success = service.generate_chat_response("Hello, how are you?")
    logger.log(f"Success: {success}")
    print(f"Success: {success}")
    logger.log(f"Response: {response}")
    print(f"Response: {response}")
except Exception as e:
    logger.log(f"Service test failed: {str(e)}", level="error")
    print(f"Service test failed: {str(e)}")

logger.log("\nDiagnostic complete! Review the output above for any errors.")
print("\nDiagnostic complete! Review the output above for any errors.")
