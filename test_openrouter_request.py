import requests
import json
import os

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("OPENROUTER_API_KEY not set in environment")
    exit(1)

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost",  # Optional
    "X-Title": "HRM-Gemini AI"           # Optional
  },
  data=json.dumps({
    "model": "openai/gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

print(f"Status code: {response.status_code}")
print("Response body:")
print(response.text)
