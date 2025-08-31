import sys

def test_basic():
    print("✅ Basic Python test script is working!")
    print(f"Python version: {sys.version}")
    return True

if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)
