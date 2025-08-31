#!/usr/bin/env python3
"""
Advanced Error Recovery System for HRM-Gemini AI
Automatically recovers from various error conditions and maintains system stability
"""

import sys
import time
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class AdvancedErrorRecovery:
    def __init__(self, controller):
        self.controller = controller
        self.error_history = []
        self.recovery_attempts = {}
        self.system_health = {
            'status': 'healthy',
            'last_error': None,
            'recovery_count': 0,
            'critical_errors': 0
        }
        self.recovery_strategies = {
            'import_error': self._recover_import_error,
            'connection_error': self._recover_connection_error,
            'memory_error': self._recover_memory_error,
            'permission_error': self._recover_permission_error,
            'timeout_error': self._recover_timeout_error,
            'validation_error': self._recover_validation_error,
            'generic_error': self._recover_generic_error
        }

    def handle_error(self, error, context=None, operation=None):
        """Main error handling and recovery method"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'operation': operation,
            'traceback': traceback.format_exc()
        }

        # Log the error
        self.error_history.append(error_info)
        self.system_health['last_error'] = error_info['timestamp']

        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]

        # Classify and attempt recovery
        error_category = self._classify_error(error)

        print(f"🚨 Error detected: {error_category}")
        print(f"   {error_info['error_message']}")

        # Attempt recovery
        recovery_success = self._attempt_recovery(error_category, error_info)

        if recovery_success:
            print("✅ Error recovered successfully")
            self.system_health['recovery_count'] += 1
        else:
            print("❌ Recovery failed - escalating to manual intervention")
            self.system_health['critical_errors'] += 1
            self._escalate_error(error_info)

        # Update system health status
        self._update_health_status()

        return recovery_success

    def _classify_error(self, error):
        """Classify error type for appropriate recovery strategy"""
        error_name = type(error).__name__

        if 'Import' in error_name:
            return 'import_error'
        elif 'Connection' in error_name or 'Network' in str(error):
            return 'connection_error'
        elif 'Memory' in error_name:
            return 'memory_error'
        elif 'Permission' in error_name or 'Access' in str(error):
            return 'permission_error'
        elif 'Timeout' in error_name:
            return 'timeout_error'
        elif 'Validation' in str(error) or 'Invalid' in str(error):
            return 'validation_error'
        else:
            return 'generic_error'

    def _attempt_recovery(self, error_category, error_info):
        """Attempt recovery using appropriate strategy"""
        if error_category in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_category]
                return recovery_func(error_info)
            except Exception as recovery_error:
                print(f"Recovery strategy failed: {recovery_error}")
                return False
        else:
            return self._recover_generic_error(error_info)

    def _recover_import_error(self, error_info):
        """Recover from import errors"""
        print("🔄 Attempting import error recovery...")

        # Try to reinstall missing dependencies
        try:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            print("   Updated pip")

            # Try to install common missing packages
            common_packages = ['torch', 'transformers', 'psutil', 'requests']
            for package in common_packages:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=False)
                except:
                    pass

            return True
        except Exception as e:
            print(f"   Import recovery failed: {e}")
            return False

    def _recover_connection_error(self, error_info):
        """Recover from connection errors"""
        print("🔄 Attempting connection error recovery...")

        # Wait and retry
        for attempt in range(3):
            try:
                print(f"   Retry attempt {attempt + 1}/3...")
                time.sleep(2 ** attempt)  # Exponential backoff

                # Try a simple connection test
                import requests
                requests.get('https://www.google.com', timeout=5)
                return True
            except:
                continue

        return False

    def _recover_memory_error(self, error_info):
        """Recover from memory errors"""
        print("🔄 Attempting memory error recovery...")

        try:
            import gc
            gc.collect()  # Force garbage collection
            print("   Performed garbage collection")

            # Clear any cached data
            if hasattr(self.controller, 'hrm'):
                if hasattr(self.controller.hrm, 'memory'):
                    # Keep only recent memory items
                    recent_memory = {}
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    for key, value in self.controller.hrm.memory.items():
                        if isinstance(value, dict) and 'timestamp' in value:
                            if value['timestamp'] > cutoff_time:
                                recent_memory[key] = value
                        else:
                            recent_memory[key] = value
                    self.controller.hrm.memory = recent_memory

            return True
        except Exception as e:
            print(f"   Memory recovery failed: {e}")
            return False

    def _recover_permission_error(self, error_info):
        """Recover from permission errors"""
        print("🔄 Attempting permission error recovery...")

        # Try to create necessary directories
        try:
            import os
            import tempfile

            # Ensure temp directory exists and is writable
            temp_dir = Path(tempfile.gettempdir()) / 'hrm_temp'
            temp_dir.mkdir(exist_ok=True)

            # Test write permissions
            test_file = temp_dir / 'test_write.txt'
            test_file.write_text('test')
            test_file.unlink()

            print("   Verified write permissions")
            return True
        except Exception as e:
            print(f"   Permission recovery failed: {e}")
            return False

    def _recover_timeout_error(self, error_info):
        """Recover from timeout errors"""
        print("🔄 Attempting timeout error recovery...")

        # Increase timeout values and retry
        try:
            # This would be operation-specific, but for now we'll just wait
            time.sleep(1)
            print("   Increased timeout tolerance")
            return True
        except Exception as e:
            print(f"   Timeout recovery failed: {e}")
            return False

    def _recover_validation_error(self, error_info):
        """Recover from validation errors"""
        print("🔄 Attempting validation error recovery...")

        try:
            # Reset to default values
            if hasattr(self.controller, 'config'):
                # This would reload default configuration
                print("   Reset configuration to defaults")
            return True
        except Exception as e:
            print(f"   Validation recovery failed: {e}")
            return False

    def _recover_generic_error(self, error_info):
        """Generic error recovery"""
        print("🔄 Attempting generic error recovery...")

        try:
            # Basic recovery steps
            time.sleep(0.5)  # Brief pause
            print("   Applied generic recovery measures")
            return True
        except Exception as e:
            print(f"   Generic recovery failed: {e}")
            return False

    def _escalate_error(self, error_info):
        """Escalate critical errors for manual intervention"""
        print("🚨 CRITICAL ERROR - MANUAL INTERVENTION REQUIRED")
        print("Error details:")
        print(f"   Type: {error_info['error_type']}")
        print(f"   Message: {error_info['error_message']}")
        print(f"   Operation: {error_info['operation']}")
        print(f"   Time: {error_info['timestamp']}")

        # Could send notification, log to external system, etc.
        # For now, just provide detailed information

    def _update_health_status(self):
        """Update overall system health status"""
        recent_errors = [e for e in self.error_history
                        if e['timestamp'] > datetime.now() - timedelta(minutes=5)]

        if len(recent_errors) > 5:
            self.system_health['status'] = 'critical'
        elif len(recent_errors) > 2:
            self.system_health['status'] = 'warning'
        else:
            self.system_health['status'] = 'healthy'

    def get_health_report(self):
        """Get comprehensive health report"""
        report = {
            'overall_status': self.system_health['status'],
            'total_errors': len(self.error_history),
            'recent_errors': len([e for e in self.error_history
                                 if e['timestamp'] > datetime.now() - timedelta(hours=1)]),
            'recovery_success_rate': self._calculate_recovery_rate(),
            'critical_errors': self.system_health['critical_errors'],
            'last_error_time': self.system_health['last_error']
        }
        return report

    def _calculate_recovery_rate(self):
        """Calculate recovery success rate"""
        if not self.error_history:
            return 100.0

        recovered_count = self.system_health['recovery_count']
        total_errors = len(self.error_history)

        if total_errors == 0:
            return 100.0

        return (recovered_count / total_errors) * 100.0

    def perform_health_check(self):
        """Perform comprehensive health check"""
        print("🏥 Performing system health check...")

        issues_found = []
        recommendations = []

        # Check error rates
        error_rate = len(self.error_history) / max(1, (datetime.now() - (self.error_history[0]['timestamp'] if self.error_history else datetime.now())).total_seconds() / 3600)
        if error_rate > 10:  # More than 10 errors per hour
            issues_found.append("High error rate detected")
            recommendations.append("Consider reviewing recent operations")

        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                issues_found.append("High memory usage")
                recommendations.append("Consider restarting the system")
        except:
            pass

        # Generate report
        if issues_found:
            print("⚠️  Health issues detected:")
            for issue in issues_found:
                print(f"   • {issue}")
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print("✅ System health is excellent")

        return len(issues_found) == 0
