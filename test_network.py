import os
import sys
import requests

def test_network():
    print("🔍 Testing network connectivity...")
    
    # Test basic internet connectivity
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print(f"✅ Internet connection: Working (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Internet connection failed: {str(e)}")
        return False
    
    # Test OpenRouter API endpoint
    try:
        response = requests.get("https://openrouter.ai/api/v1/auth/key", 
                             headers={"Authorization": "Bearer test"},
                             timeout=10)
        print(f"✅ OpenRouter API reachable (Status: {response.status_code})")
        return True
    except Exception as e:
        print(f"❌ OpenRouter API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_network()
    sys.exit(0 if success else 1)
