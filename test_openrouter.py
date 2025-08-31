import os
import sys
import json
import requests
from typing import Dict, Any, Optional, Tuple
from config.openrouter_config import (
    get_headers, 
    get_error_message, 
    DEFAULT_MODEL,
    get_model_config,
    exponential_backoff_retry
)

def test_connection() -> bool:
    """Test basic connection to OpenRouter API with retry logic.
    
    Returns:
        bool: True if connection was successful, False otherwise
    """
    def _make_request() -> requests.Response:
        url = "https://openrouter.ai/api/v1/auth/key"
        return requests.get(url, headers=get_headers())
    
    print("🔍 Testing OpenRouter connection...")
    response, success = exponential_backoff_retry(_make_request)
    
    if not success:
        print(f"❌ Connection failed after retries: {response}")
        return False
        
    if response.status_code == 200:
        print("✅ OpenRouter connection successful")
        print(f"🔑 Key info: {json.dumps(response.json(), indent=2)}")
        return True
    else:
        print(f"❌ Connection failed: {get_error_message(response.status_code)}")
        print(f"   Status code: {response.status_code}")
        if response.text:
            print(f"   Response: {response.text[:200]}...")
        return False

def test_chat() -> bool:
    """Test sending a chat message with retry logic.
    
    Returns:
        bool: True if chat test was successful, False otherwise
    """
    def _make_chat_request() -> requests.Response:
        url = "https://openrouter.ai/api/v1/chat/completions"
        data = get_model_config(
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )
        return requests.post(url, headers=get_headers(), json=data)
    
    print("\n💬 Testing chat functionality...")
    response, success = exponential_backoff_retry(_make_chat_request)
    
    if not success:
        print(f"❌ Chat test failed after retries: {response}")
        return False
        
    if response.status_code == 200:
        try:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                reply = result['choices'][0]['message']['content']
                print("✅ Chat test successful")
                print(f"🤖 Response: {reply}")
                return True
            else:
                print("❌ Unexpected response format from API")
                print(f"   Raw response: {response.text[:200]}...")
                return False
        except json.JSONDecodeError:
            print("❌ Failed to parse JSON response")
            print(f"   Response: {response.text[:200]}...")
            return False
    else:
        print(f"❌ Chat test failed: {get_error_message(response.status_code)}")
        print(f"   Status code: {response.status_code}")
        if response.text:
            print(f"   Response: {response.text[:200]}...")
        return False

def run_tests() -> bool:
    """Run all tests and return overall success status."""
    print("\n" + "="*50)
    print("🔍 Starting OpenRouter Integration Tests")
    print("="*50)
    
    # Test connection first
    if not test_connection():
        print("\n❌ OpenRouter connection test failed. Please check your API key and internet connection.")
        return False
    
    # If connection is good, test chat
    if not test_chat():
        print("\n❌ Chat test failed. Please check your configuration.")
        return False
    
    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
