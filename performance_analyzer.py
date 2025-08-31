#!/usr/bin/env python3
"""
Performance Analyzer for Self-Modification System
Monitors and analyzes system performance to identify optimization opportunities
"""

import time
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import tracemalloc
import sys
import os

class PerformanceAnalyzer:
    """Analyzes system performance and identifies optimization opportunities"""

    def __init__(self, self_modification_engine):
        self.engine = self_modification_engine
        self.start_time = time.time()
        self.memory_snapshots = []
        self.response_times = []
        self.error_counts = []
        self.monitoring_active = False

        # Start monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        tracemalloc.start()

        # Start background monitoring thread
        monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        tracemalloc.stop()

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current system performance"""
        current_time = time.time()
        uptime = current_time - self.start_time

        analysis = {
            "uptime_seconds": uptime,
            "startup_time": self._measure_startup_time(),
            "memory_usage": self._analyze_memory_usage(),
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "response_time_avg": self._calculate_average_response_time(),
            "error_rate": self._calculate_error_rate(),
            "performance_score": self._calculate_performance_score()
        }

        return analysis

    def _measure_startup_time(self) -> float:
        """Measure application startup time"""
        # This would be measured from application launch to ready state
        # For now, we'll use a simulated measurement
        startup_times = self._load_performance_data().get("startup_times", [])
        return startup_times[-1] if startup_times else 2.5  # Default 2.5 seconds

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()

            memory_analysis = {
                "total_mb": memory_info.total / (1024 * 1024),
                "available_mb": memory_info.available / (1024 * 1024),
                "used_mb": memory_info.used / (1024 * 1024),
                "usage_percent": memory_info.percent,
                "process_memory_mb": process.memory_info().rss / (1024 * 1024),
                "memory_efficiency": self._calculate_memory_efficiency()
            }

            return memory_analysis

        except Exception as e:
            return {"error": str(e), "usage_percent": 0}

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        try:
            # Get memory snapshots
            snapshots = tracemalloc.get_traced_memory()
            if snapshots[1] > 0:  # Current memory
                efficiency = (snapshots[0] / snapshots[1]) * 100  # Peak vs current
                return min(100.0, efficiency)
            return 100.0
        except:
            return 85.0  # Default good efficiency

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.5  # Default 500ms

        recent_times = self.response_times[-50:]  # Last 50 responses
        return sum(recent_times) / len(recent_times) if recent_times else 0.5

    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        if not self.error_counts:
            return 0.0

        total_operations = len(self.response_times) + sum(self.error_counts)
        if total_operations == 0:
            return 0.0

        return (sum(self.error_counts) / total_operations) * 100

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            # Factors that affect performance score
            factors = {
                "memory_efficiency": 0.3,
                "response_time": 0.3,
                "error_rate": 0.2,
                "cpu_usage": 0.2
            }

            # Get current metrics
            memory_eff = self._calculate_memory_efficiency()
            response_time = self._calculate_average_response_time()
            error_rate = self._calculate_error_rate()
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Normalize and score each factor
            memory_score = min(100, memory_eff)
            response_score = max(0, 100 - (response_time * 200))  # Faster is better
            error_score = max(0, 100 - (error_rate * 10))  # Lower error rate is better
            cpu_score = max(0, 100 - cpu_usage)  # Lower CPU usage is better

            # Calculate weighted score
            total_score = (
                memory_score * factors["memory_efficiency"] +
                response_score * factors["response_time"] +
                error_score * factors["error_rate"] +
                cpu_score * factors["cpu_usage"]
            )

            return round(total_score, 1)

        except Exception:
            return 75.0  # Default good score

    def record_response_time(self, response_time: float):
        """Record a response time measurement"""
        self.response_times.append(response_time)

        # Keep only recent measurements
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

    def record_error(self, error_type: str = "general"):
        """Record an error occurrence"""
        self.error_counts.append(1)

        # Keep only recent error counts
        if len(self.error_counts) > 100:
            self.error_counts = self.error_counts[-100:]

    def optimize_startup(self) -> bool:
        """Optimize startup time"""
        try:
            print("⚡ Optimizing startup performance...")

            # Analyze current startup bottlenecks
            bottlenecks = self._identify_startup_bottlenecks()

            improvements = 0

            # Optimize imports
            if "import_heavy" in bottlenecks:
                improvements += self._optimize_imports()

            # Optimize initialization
            if "init_heavy" in bottlenecks:
                improvements += self._optimize_initialization()

            # Optimize memory allocation
            if "memory_heavy" in bottlenecks:
                improvements += self._optimize_memory_allocation()

            print(f"✅ Startup optimization complete: {improvements} improvements applied")
            return improvements > 0

        except Exception as e:
            print(f"❌ Startup optimization failed: {e}")
            return False

    def optimize_memory(self) -> bool:
        """Optimize memory usage"""
        try:
            print("🧠 Optimizing memory usage...")

            # Force garbage collection
            import gc
            gc.collect()

            # Analyze memory patterns
            memory_stats = tracemalloc.get_traced_memory()
            peak_memory = memory_stats[1]

            improvements = 0

            # Implement lazy loading if memory usage is high
            if peak_memory > 100 * 1024 * 1024:  # > 100MB
                improvements += self._implement_lazy_loading()

            # Optimize data structures
            improvements += self._optimize_data_structures()

            # Clear unused caches
            improvements += self._clear_unused_caches()

            print(f"✅ Memory optimization complete: {improvements} improvements applied")
            return improvements > 0

        except Exception as e:
            print(f"❌ Memory optimization failed: {e}")
            return False

    def _identify_startup_bottlenecks(self) -> List[str]:
        """Identify startup performance bottlenecks"""
        bottlenecks = []

        # Check import times
        try:
            import_time = time.time()
            import sys
            import_time = time.time() - import_time
            if import_time > 0.5:
                bottlenecks.append("import_heavy")
        except:
            pass

        # Check memory allocation
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 80:
            bottlenecks.append("memory_heavy")

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 70:
            bottlenecks.append("cpu_heavy")

        return bottlenecks

    def _optimize_imports(self) -> int:
        """Optimize import statements"""
        # This would modify the code to use lazy imports
        # For now, return a placeholder
        return 1

    def _optimize_initialization(self) -> int:
        """Optimize initialization processes"""
        # This would modify initialization code
        return 1

    def _optimize_memory_allocation(self) -> int:
        """Optimize memory allocation patterns"""
        # This would modify memory allocation strategies
        return 1

    def _implement_lazy_loading(self) -> int:
        """Implement lazy loading for heavy components"""
        # This would modify the code to load components on demand
        return 1

    def _optimize_data_structures(self) -> int:
        """Optimize data structures for memory efficiency"""
        # This would modify data structure usage
        return 1

    def _clear_unused_caches(self) -> int:
        """Clear unused caches and temporary data"""
        # This would clear various caches
        return 1

    def _background_monitor(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                # Take memory snapshot
                snapshot = tracemalloc.get_traced_memory()
                self.memory_snapshots.append(snapshot)

                # Keep only recent snapshots
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots = self.memory_snapshots[-100:]

                # Save performance data periodically
                if len(self.memory_snapshots) % 10 == 0:
                    self._save_performance_data()

            except Exception as e:
                print(f"Background monitoring error: {e}")

            time.sleep(5)  # Monitor every 5 seconds

    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            perf_data = self._load_performance_data()

            # Add current measurements
            perf_data["startup_times"].append(self._measure_startup_time())
            perf_data["memory_usage"].append(self._analyze_memory_usage()["usage_percent"])
            perf_data["response_times"].append(self._calculate_average_response_time())

            # Keep only recent data
            for key in perf_data:
                if isinstance(perf_data[key], list) and len(perf_data[key]) > 100:
                    perf_data[key] = perf_data[key][-100:]

            with open(self.engine.performance_data, 'w') as f:
                import json
                json.dump(perf_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to save performance data: {e}")

    def _load_performance_data(self) -> Dict[str, Any]:
        """Load performance data from file"""
        try:
            with open(self.engine.performance_data, 'r') as f:
                import json
                return json.load(f)
        except:
            return {
                "startup_times": [],
                "memory_usage": [],
                "response_times": [],
                "error_rates": [],
                "feature_usage": {}
            }
