import os
import sys
import json
from pprint import pprint

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from config.openrouter_config import *
from services.openrouter_service import OpenRouterService

def print_environment():
    """Print relevant environment variables"""
    print("\n=== Environment ===")
    for key, value in os.environ.items():
        if 'OPENROUTER' in key or 'HRM' in key or 'PYTHON' in key or 'PATH' in key:
            print(f"{key} = {value}")
    print("=================\n")

def test_openrouter():
    print("=== Starting OpenRouter Integration Test ===")
    
    # Print environment info
    print_environment()
    
    # Show config values
    print("\n=== Configuration ===")
    print(f"API Key: {'*' * 8 + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else 'Not set'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Base URL: {OPENROUTER_API_BASE}")
    print("==================\n")
    
    # Initialize the service
    print("Initializing OpenRouterService...")
    try:
        service = OpenRouterService()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OpenRouterService: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test a simple chat completion
    test_prompt = "Hello, how are you today?"
    print(f"\nSending test prompt: '{test_prompt}'")
    
    try:
        print("Calling generate_chat_response...")
        response, success = service.generate_chat_response(test_prompt)
        
        if success:
            print("\n✅ OpenRouter test successful!")
            print(f"Response type: {type(response)}")
            print(f"Response content: {response}")
        else:
            print(f"\n❌ OpenRouter returned an error:")
            if isinstance(response, dict):
                print(json.dumps(response, indent=2))
            else:
                print(response)
    except Exception as e:
        print(f"\n❌ Error testing OpenRouter: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if API key is set in environment
    env_key = os.getenv("OPENROUTER_API_KEY")
    if not env_key:
        print("⚠️  OPENROUTER_API_KEY environment variable is not set")
        print(f"Using default API key from config (may not work): {OPENROUTER_API_KEY}")
    else:
        print(f"✅ Found OPENROUTER_API_KEY in environment")
    
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    test_openrouter()
