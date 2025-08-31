#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for HRM-Gemini AI System
Provides real-time insights into system performance and optimization metrics
"""

import sys
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class PerformanceMonitor:
    def __init__(self, controller):
        self.controller = controller
        self.start_time = datetime.now()
        self.metrics = {
            'commands_processed': 0,
            'optimizations_performed': 0,
            'errors_encountered': 0,
            'uptime_seconds': 0,
            'memory_usage': [],
            'cpu_usage': [],
            'response_times': []
        }
        self.is_monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("📊 Performance monitoring stopped")

    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_metrics()
                time.sleep(5)  # Collect metrics every 5 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)

    def _collect_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            self.metrics['cpu_usage'].append({
                'timestamp': datetime.now(),
                'percentage': cpu_percent
            })

            self.metrics['memory_usage'].append({
                'timestamp': datetime.now(),
                'percentage': memory_percent,
                'used_mb': memory.used / 1024 / 1024
            })

            # Keep only last 100 readings
            if len(self.metrics['cpu_usage']) > 100:
                self.metrics['cpu_usage'] = self.metrics['cpu_usage'][-100:]
            if len(self.metrics['memory_usage']) > 100:
                self.metrics['memory_usage'] = self.metrics['memory_usage'][-100:]

        except Exception as e:
            self.metrics['errors_encountered'] += 1

    def record_command(self, command, response_time=None):
        """Record command execution"""
        self.metrics['commands_processed'] += 1
        if response_time:
            self.metrics['response_times'].append({
                'timestamp': datetime.now(),
                'command': command,
                'response_time': response_time
            })

    def record_optimization(self):
        """Record optimization performed"""
        self.metrics['optimizations_performed'] += 1

    def record_error(self):
        """Record error encountered"""
        self.metrics['errors_encountered'] += 1

    def get_dashboard_data(self):
        """Get comprehensive dashboard data"""
        current_time = datetime.now()
        uptime = current_time - self.start_time
        self.metrics['uptime_seconds'] = uptime.total_seconds()

        # Calculate averages
        avg_cpu = sum([m['percentage'] for m in self.metrics['cpu_usage'][-10:]]) / max(len(self.metrics['cpu_usage'][-10:]), 1)
        avg_memory = sum([m['percentage'] for m in self.metrics['memory_usage'][-10:]]) / max(len(self.metrics['memory_usage'][-10:]), 1)

        # Calculate response time stats
        response_times = [r['response_time'] for r in self.metrics['response_times'][-20:]]
        avg_response_time = sum(response_times) / max(len(response_times), 1) if response_times else 0

        return {
            'uptime': str(uptime).split('.')[0],  # Remove microseconds
            'commands_processed': self.metrics['commands_processed'],
            'optimizations_performed': self.metrics['optimizations_performed'],
            'errors_encountered': self.metrics['errors_encountered'],
            'avg_cpu_usage': ".1f",
            'avg_memory_usage': ".1f",
            'avg_response_time': ".3f",
            'current_cpu': self.metrics['cpu_usage'][-1]['percentage'] if self.metrics['cpu_usage'] else 0,
            'current_memory': self.metrics['memory_usage'][-1]['percentage'] if self.metrics['memory_usage'] else 0,
            'memory_used_mb': self.metrics['memory_usage'][-1]['used_mb'] if self.metrics['memory_usage'] else 0,
            'system_status': 'Active' if self.is_monitoring else 'Inactive'
        }

    def show_dashboard(self):
        """Display the performance dashboard"""
        data = self.get_dashboard_data()

        print("\n" + "="*70)
        print("📊 HRM-GEMINI AI SYSTEM PERFORMANCE DASHBOARD")
        print("="*70)
        print("2d")
        print(f"🖥️  System Status: {data['system_status']}")
        print(f"⚡ CPU Usage: {data['current_cpu']:.1f}% (avg: {data['avg_cpu_usage']}%)")
        print(f"💾 Memory Usage: {data['current_memory']:.1f}% (avg: {data['avg_memory_usage']}%, {data['memory_used_mb']:.0f} MB)")
        print(f"🕐 Avg Response Time: {data['avg_response_time']:.3f}s")
        print()

        print("📈 ACTIVITY METRICS:")
        print("2d")
        print("2d")
        print("2d")
        print()

        print("🎯 PERFORMANCE INSIGHTS:")
        if data['avg_cpu_usage'] > 80:
            print("⚠️  High CPU usage detected - consider optimization")
        elif data['avg_memory_usage'] > 85:
            print("⚠️  High memory usage detected - consider cleanup")
        elif data['errors_encountered'] > data['commands_processed'] * 0.1:
            print("⚠️  High error rate detected - review system stability")
        else:
            print("✅ System performance is optimal")

        print("="*70 + "\n")

    def get_performance_report(self):
        """Generate detailed performance report"""
        data = self.get_dashboard_data()

        report = f"""
HRM-Gemini AI System Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM OVERVIEW:
- Uptime: {data['uptime']}
- Status: {data['system_status']}
- Commands Processed: {data['commands_processed']}
- Optimizations Performed: {data['optimizations_performed']}
- Errors Encountered: {data['errors_encountered']}

PERFORMANCE METRICS:
- Average CPU Usage: {data['avg_cpu_usage']}%
- Average Memory Usage: {data['avg_memory_usage']}%
- Average Response Time: {data['avg_response_time']}s
- Current CPU: {data['current_cpu']}%
- Current Memory: {data['current_memory']}%

SYSTEM HEALTH:
"""

        # Health assessment
        if data['avg_cpu_usage'] < 50 and data['avg_memory_usage'] < 70:
            report += "- Health Status: EXCELLENT\n"
        elif data['avg_cpu_usage'] < 70 and data['avg_memory_usage'] < 80:
            report += "- Health Status: GOOD\n"
        elif data['avg_cpu_usage'] < 85 and data['avg_memory_usage'] < 90:
            report += "- Health Status: FAIR\n"
        else:
            report += "- Health Status: POOR - Consider optimization\n"

        return report

    def export_report(self, filename=None):
        """Export performance report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hrM_performance_report_{timestamp}.txt"

        report = self.get_performance_report()

        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"✅ Performance report exported to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
            return False
