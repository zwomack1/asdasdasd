@echo off
echo Testing OpenRouter Integration...
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

python -c "
import os
import sys
import json
import requests

print('Python Version:', sys.version)
print('Current Directory:', os.getcwd())

# Test basic requests
print('\nTesting requests installation...')
try:
    r = requests.get('https://httpbin.org/get')
    print(f'Requests test: Status {r.status_code}')
except Exception as e:
    print(f'Requests test failed: {str(e)}')
    sys.exit(1)

# Test OpenRouter
print('\nTesting OpenRouter API...')
api_key = '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'
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

try:
    # Test auth endpoint
    print('Testing authentication...')
    auth_response = requests.get('https://openrouter.ai/api/v1/auth/key', headers=headers, timeout=10)
    print(f'Auth status: {auth_response.status_code}')
    if auth_response.status_code == 200:
        print('✅ OpenRouter authentication successful!')
        print('Key info:', json.dumps(auth_response.json(), indent=2))
    else:
        print('❌ OpenRouter authentication failed')
        print('Response:', auth_response.text[:500])
        
    # Test chat completion
    print('\nTesting chat completion...')
    chat_data = {
        'model': 'kimi',
        'messages': [{'role': 'user', 'content': 'Hello, how are you?'}]
    }
    chat_response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=chat_data,
        timeout=30
    )
    print(f'Chat status: {chat_response.status_code}')
    if chat_response.status_code == 200:
        print('✅ Chat test successful!')
        result = chat_response.json()
        if 'choices' in result and result['choices']:
            print('🤖 Response:', result['choices'][0]['message']['content'])
    else:
        print('❌ Chat test failed')
        print('Response:', chat_response.text[:500])
        
except Exception as e:
    print(f'❌ Error: {str(e)}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

pause
