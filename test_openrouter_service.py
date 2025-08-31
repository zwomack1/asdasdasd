import os
import sys
from config.openrouter_config import *
from services.openrouter_service import OpenRouterService

def test_service():
    print("=== OpenRouter Service Test ===")
    
    # Initialize service
    try:
        service = OpenRouterService()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {str(e)}")
        return False
    
    # Test chat completion
    test_prompt = "Hello, how are you today?"
    print(f"\nTesting with prompt: '{test_prompt}'")
    
    try:
        response, success = service.generate_chat_response(test_prompt)
        if success:
            print(f"✅ Success! Response: {response}")
            return True
        else:
            print(f"❌ Error response: {response}")
            return False
    except Exception as e:
        print(f"❌ Exception during test: {str(e)}")
        return False

if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set in environment")
        print(f"Using default key (may not work): {OPENROUTER_API_KEY}")
    
    success = test_service()
    sys.exit(0 if success else 1)
