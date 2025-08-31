import sys
import os

print("=== Simple Python Test ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment variables:")
for key in ['PATH', 'PYTHONPATH', 'OPENROUTER_API_KEY', 'HRM_PROVIDER']:
    print(f"{key}: {os.environ.get(key, 'Not set')}")

input("Press Enter to exit...")
