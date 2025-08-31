import os
import sys
import requests
from pprint import pprint

# Test 1: Environment check
print("=== Environment Check ===")
print(f"Python Path: {sys.executable}")
print(f"Working Directory: {os.getcwd()}")
print(f"OPENROUTER_API_KEY: {'set' if os.getenv('OPENROUTER_API_KEY') else 'not set'}")
print(f"HRM_PROVIDER: {os.getenv('HRM_PROVIDER', 'not set')}")

# Test 2: Network connectivity
try:
    print("\n=== Network Test ===")
    response = requests.get('https://openrouter.ai', timeout=5)
    print(f"OpenRouter website accessible: {response.status_code == 200}")
    response = requests.get('https://api.openrouter.ai', timeout=5)
    print(f"OpenRouter API accessible: {response.status_code == 200}")
except Exception as e:
    print(f"Network test failed: {str(e)}")
    sys.exit(1)

# Test 3: API key validation
print("\n=== API Key Test ===")
api_key = os.getenv('OPENROUTER_API_KEY', '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')
try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10)
    print(f"API Key Valid: {response.status_code == 200}")
    if response.status_code == 200:
        pprint(response.json())
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"API test failed: {str(e)}")
    sys.exit(1)

print("\nAll tests completed successfully!")
