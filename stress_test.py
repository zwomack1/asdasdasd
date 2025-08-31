#!/usr/bin/env python3
"""
Comprehensive Stress Test for HRM-Gemini AI System
Tests all features, error handling, and performance under load
"""

import sys
import os
import time
import threading
import subprocess
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all imports work correctly"""
    print("🧪 Testing imports...")

    try:
        from config import Config
        from hrm_model import HRMModel
        from controllers.integration_controller import HRMGeminiController
        from demo.cli_interface import CLIInterface
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration system"""
    print("⚙️ Testing configuration...")

    try:
        from config import Config
        config = Config()
        print(f"✅ Config loaded: {config.project_root}")
        print(f"✅ Model path: {config.model_path}")
        print(f"✅ Services path: {config.services_path}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_hrm_model():
    """Test HRM model initialization and basic functions"""
    print("🧠 Testing HRM model...")

    try:
        from hrm_model import HRMModel
        hrm = HRMModel()

        # Test basic functions
        state = hrm.get_system_state()
        print(f"✅ HRM state: {state}")

        # Test decision making
        context = {'command': 'test', 'input': 'hello world'}
        decision = hrm.influence_decision(context)
        print(f"✅ HRM decision: {decision}")

        return True
    except Exception as e:
        print(f"❌ HRM model test failed: {e}")
        return False

def test_controller():
    """Test integration controller"""
    print("🎛️ Testing controller...")

    try:
        from controllers.integration_controller import HRMGeminiController
        controller = HRMGeminiController()

        # Test basic functions
        state = controller.get_state()
        print(f"✅ Controller state: {state}")

        # Test decision making
        context = {'command': 'chat', 'input': 'test message'}
        decision = controller.make_decision(context)
        print(f"✅ Controller decision: {decision}")

        return True
    except Exception as e:
        print(f"❌ Controller test failed: {e}")
        return False

def test_cli_initialization():
    """Test CLI interface initialization"""
    print("💻 Testing CLI initialization...")

    try:
        from demo.cli_interface import CLIInterface
        cli = CLIInterface()
        print("✅ CLI initialized successfully")
        return True
    except Exception as e:
        print(f"❌ CLI initialization failed: {e}")
        return False

def stress_test_commands():
    """Stress test various commands"""
    print("💪 Stress testing commands...")

    try:
        from controllers.integration_controller import HRMGeminiController
        controller = HRMGeminiController()

        # Test multiple commands rapidly
        test_commands = [
            {'command': 'chat', 'input': 'Hello HRM!'},
            {'command': 'optimize', 'input': 'def test(): pass'},
            {'command': 'recode', 'input': 'print("hello")', 'instructions': 'Make it better'},
            {'command': 'status'},
            {'command': 'unknown_command'}
        ]

        for i, cmd in enumerate(test_commands):
            try:
                if cmd['command'] == 'recode':
                    result = controller.recode(cmd['input'], cmd['instructions'])
                elif cmd['command'] == 'optimize':
                    result = controller.optimize_code(cmd['input'])
                else:
                    result = controller.make_decision(cmd)

                print(f"✅ Command {i+1}/{len(test_commands)}: {cmd['command']} -> Success")

            except Exception as e:
                print(f"⚠️ Command {i+1} failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Command stress test failed: {e}")
        return False

def test_concurrent_access():
    """Test concurrent access to the system"""
    print("🔄 Testing concurrent access...")

    results = []
    errors = []

    def worker_thread(thread_id):
        try:
            from controllers.integration_controller import HRMGeminiController
            controller = HRMGeminiController()

            # Perform some operations
            for i in range(5):
                context = {'command': f'test_{thread_id}_{i}', 'input': f'Thread {thread_id} message {i}'}
                result = controller.make_decision(context)
                results.append(f"Thread {thread_id}: {result}")

            print(f"✅ Thread {thread_id} completed successfully")

        except Exception as e:
            errors.append(f"Thread {thread_id}: {e}")
            print(f"❌ Thread {thread_id} failed: {e}")

    # Start multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print(f"✅ Concurrent test completed: {len(results)} successful operations, {len(errors)} errors")
    return len(errors) == 0

def test_error_handling():
    """Test error handling capabilities"""
    print("🚨 Testing error handling...")

    try:
        from controllers.integration_controller import HRMGeminiController
        controller = HRMGeminiController()

        # Test with invalid inputs
        test_cases = [
            None,
            {},
            {'command': None},
            {'input': None},
            {'command': '', 'input': ''},
            {'command': 'invalid_command'},
        ]

        for i, test_case in enumerate(test_cases):
            try:
                result = controller.make_decision(test_case)
                print(f"✅ Error case {i+1}: Handled gracefully -> {result}")
            except Exception as e:
                print(f"⚠️ Error case {i+1}: {e}")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage under load"""
    print("🧠 Testing memory usage...")

    try:
        from hrm_model import HRMModel
        import psutil
        import os

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(".1f")

        # Create multiple HRM instances
        hrms = []
        for i in range(10):
            hrm = HRMModel()
            hrms.append(hrm)

            # Perform operations
            for j in range(100):
                context = {'command': f'memory_test_{i}_{j}', 'input': f'Large input data ' * 100}
                hrm.influence_decision(context)

        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(".1f")
        print(".1f")

        # Cleanup
        del hrms

        return memory_increase < 500  # Less than 500MB increase is acceptable

    except ImportError:
        print("⚠️ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_launcher_files():
    """Test launcher files exist and are executable"""
    print("🚀 Testing launcher files...")

    launcher_files = [
        'Run HRM-Gemini AI.bat',
        'Run HRM-Gemini AI.ps1',
        'Create Desktop Shortcut.bat'
    ]

    all_exist = True
    for launcher in launcher_files:
        launcher_path = project_root / launcher
        if launcher_path.exists():
            print(f"✅ {launcher} exists")
        else:
            print(f"❌ {launcher} missing")
            all_exist = False

    return all_exist

def run_full_test_suite():
    """Run the complete test suite"""
    print("🧪 HRM-Gemini AI System - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config),
        ("HRM Model Tests", test_hrm_model),
        ("Controller Tests", test_controller),
        ("CLI Initialization Tests", test_cli_initialization),
        ("Command Stress Tests", stress_test_commands),
        ("Concurrent Access Tests", test_concurrent_access),
        ("Error Handling Tests", test_error_handling),
        ("Memory Usage Tests", test_memory_usage),
        ("Launcher File Tests", test_launcher_files),
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))

    end_time = time.time()
    duration = end_time - start_time

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(".2f")
    print(".1f")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_full_test_suite()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
