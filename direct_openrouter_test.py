import os
import requests
import logging

# Setup logging - ensure file creation even with errors
try:
    logging.basicConfig(
        filename='openrouter_test.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite any existing file
    )
    logger = logging.getLogger()
except Exception as e:
    print(f"Logging setup failed: {e}")
    # Fallback to print statements
    def logger_info(msg): print(f"[INFO] {msg}")
    def logger_error(msg): print(f"[ERROR] {msg}")
else:
    logger_info = logger.info
    logger_error = logger.error

# Configuration
API_KEY = os.getenv('OPENROUTER_API_KEY', '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')
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Test 1: API key validation
logger_info("=== API Key Validation ===")
try:
    auth_response = requests.get("https://openrouter.ai/api/v1/auth/key", 
                                headers={"Authorization": f"Bearer {API_KEY}"})
    logger_info(f"Status Code: {auth_response.status_code}")
    logger_info(f"Response: {auth_response.text[:200]}")
except Exception as e:
    logger_error(f"API Key Validation Error: {str(e)}")

# Test 2: Simple chat completion
logger_info("\n=== Chat Completion Test ===")
try:
    payload = {
        "model": "kimi",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/WindsurfAI/HRM-Gemini",
        "X-Title": "HRM-Gemini AI"
    }

    chat_response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    logger_info(f"Status Code: {chat_response.status_code}")
    logger_info(f"Response: {chat_response.text[:500]}")
except Exception as e:
    logger_error(f"Chat Completion Error: {str(e)}")

logger_info("\nTest completed")
