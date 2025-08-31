import os
import sys
import json
import requests

def test_openrouter():
    print("🔍 Testing OpenRouter API directly...")
    
    # Use the hardcoded API key from the config
    api_key = "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"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Check auth
    print("\n🔐 Testing authentication...")
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers=headers,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Authentication successful!")
            print("Key info:", json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Authentication failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during authentication: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openrouter()
    sys.exit(0 if success else 1)
