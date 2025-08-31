import os
import requests

def test_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set in environment")
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        timeout=10
    )

    if response.status_code == 200:
        print("API key is valid!")
        print("Response:", response.json())
        return True
    else:
        print(f"API key validation failed with status {response.status_code}")
        print("Response:", response.text)
        return False

if __name__ == "__main__":
    test_key()
