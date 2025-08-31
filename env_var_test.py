import os
print('=== Environment Variables ===')
print(f"OPENROUTER_API_KEY: {'set' if os.getenv('OPENROUTER_API_KEY') else 'not set'}")
print(f"HRM_PROVIDER: {os.getenv('HRM_PROVIDER', 'not set')}")
print('\n=== Current Directory Contents ===')
import os; print(os.listdir('.'))
