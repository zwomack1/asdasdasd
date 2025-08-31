#!/usr/bin/env python3
"""
HRM-Gemini Self-Modification System
Advanced self-evolving AI system that analyzes and improves itself automatically
"""

import sys
import os
import json
import ast
import inspect
import time
from datetime import datetime, timedelta
from pathlib import Path
import threading
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SelfModificationEngine:
    """Core engine for HRM-Gemini self-modification and evolution"""

    def __init__(self, main_app=None):
        self.main_app = main_app
        self.project_root = Path("c:/Users/Shadow/CascadeProjects/windsurf-project")
        self.modification_log = self.project_root / "config" / "self_modifications.json"
        self.performance_data = self.project_root / "config" / "performance_data.json"
        self.user_interactions = self.project_root / "config" / "user_interactions.json"

        # Create config directory if it doesn't exist
        self.modification_log.parent.mkdir(exist_ok=True)

        # Initialize tracking systems
        self.initialize_tracking_systems()

        # Self-modification components
        self.performance_monitor = PerformanceAnalyzer(self)
        self.code_optimizer = CodeOptimizer(self)
        self.configuration_tuner = ConfigurationTuner(self)
        self.user_experience_learner = UserExperienceLearner(self)

        print("🔄 Self-Modification Engine initialized")

    def initialize_tracking_systems(self):
        """Initialize all tracking and logging systems"""
        # Initialize modification log
        if not self.modification_log.exists():
            initial_data = {
                "modifications": [],
                "performance_history": [],
                "user_feedback": [],
                "system_health": [],
                "last_self_improvement": None,
                "improvement_count": 0
            }
            with open(self.modification_log, 'w') as f:
                json.dump(initial_data, f, indent=2, default=str)

        # Initialize performance data
        if not self.performance_data.exists():
            with open(self.performance_data, 'w') as f:
                json.dump({
                    "startup_times": [],
                    "memory_usage": [],
                    "response_times": [],
                    "error_rates": [],
                    "feature_usage": {}
                }, f, indent=2, default=str)

        # Initialize user interactions
        if not self.user_interactions.exists():
            with open(self.user_interactions, 'w') as f:
                json.dump({
                    "command_patterns": {},
                    "feature_usage": {},
                    "error_patterns": {},
                    "satisfaction_scores": [],
                    "improvement_suggestions": []
                }, f, indent=2, default=str)

    def perform_self_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive self-analysis of the system"""
        print("🔍 Performing comprehensive self-analysis...")

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self.performance_monitor.analyze_performance(),
            "code_quality": self.code_optimizer.analyze_code_quality(),
            "configuration_efficiency": self.configuration_tuner.analyze_configuration(),
            "user_experience": self.user_experience_learner.analyze_user_experience(),
            "system_health": self._analyze_system_health(),
            "improvement_opportunities": []
        }

        # Identify improvement opportunities
        analysis["improvement_opportunities"] = self._identify_improvements(analysis)

        return analysis

    def apply_self_improvements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply identified improvements to the system"""
        print("🔧 Applying self-improvements...")

        improvements_applied = []
        improvements_failed = []

        for opportunity in analysis["improvement_opportunities"]:
            try:
                success = self._apply_improvement(opportunity)
                if success:
                    improvements_applied.append(opportunity)
                    print(f"✅ Applied improvement: {opportunity['description']}")
                else:
                    improvements_failed.append(opportunity)
                    print(f"❌ Failed to apply improvement: {opportunity['description']}")
            except Exception as e:
                improvements_failed.append(opportunity)
                print(f"❌ Error applying improvement: {e}")

        # Log the self-improvement session
        self._log_self_improvement(analysis, improvements_applied, improvements_failed)

        return {
            "success": True,
            "improvements_applied": len(improvements_applied),
            "improvements_failed": len(improvements_failed),
            "details": {
                "applied": improvements_applied,
                "failed": improvements_failed
            }
        }

    def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvements based on analysis"""
        improvements = []

        # Performance-based improvements
        perf = analysis["performance_metrics"]
        if perf["startup_time"] > 10:  # Slow startup
            improvements.append({
                "type": "performance",
                "category": "startup_optimization",
                "description": "Optimize startup time by reducing initialization overhead",
                "priority": "high",
                "estimated_impact": "high"
            })

        if perf["memory_usage"] > 500:  # High memory usage
            improvements.append({
                "type": "performance",
                "category": "memory_optimization",
                "description": "Optimize memory usage by implementing lazy loading",
                "priority": "medium",
                "estimated_impact": "medium"
            })

        # Code quality improvements
        code = analysis["code_quality"]
        if code["complexity_score"] > 7:
            improvements.append({
                "type": "code_quality",
                "category": "refactoring",
                "description": "Refactor complex functions to improve maintainability",
                "priority": "medium",
                "estimated_impact": "medium"
            })

        # Configuration improvements
        config = analysis["configuration_efficiency"]
        if config["inefficiency_score"] > 0.3:
            improvements.append({
                "type": "configuration",
                "category": "optimization",
                "description": "Optimize configuration for better performance",
                "priority": "low",
                "estimated_impact": "low"
            })

        # User experience improvements
        ux = analysis["user_experience"]
        if ux["satisfaction_score"] < 7:
            improvements.append({
                "type": "user_experience",
                "category": "interface_improvement",
                "description": "Improve user interface based on usage patterns",
                "priority": "high",
                "estimated_impact": "high"
            })

        return improvements

    def _apply_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a specific improvement to the system"""
        improvement_type = improvement["type"]
        category = improvement["category"]

        if improvement_type == "performance":
            if category == "startup_optimization":
                return self.performance_monitor.optimize_startup()
            elif category == "memory_optimization":
                return self.performance_monitor.optimize_memory()

        elif improvement_type == "code_quality":
            if category == "refactoring":
                return self.code_optimizer.refactor_complex_code()

        elif improvement_type == "configuration":
            if category == "optimization":
                return self.configuration_tuner.optimize_configuration()

        elif improvement_type == "user_experience":
            if category == "interface_improvement":
                return self.user_experience_learner.improve_interface()

        return False

    def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health"""
        try:
            health = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "system_uptime": time.time() - psutil.boot_time()
            }

            # Determine health status
            if health["memory_usage"] > 90 or health["cpu_usage"] > 95:
                health["status"] = "critical"
            elif health["memory_usage"] > 80 or health["cpu_usage"] > 85:
                health["status"] = "warning"
            else:
                health["status"] = "healthy"

            return health

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0
            }

    def _log_self_improvement(self, analysis: Dict[str, Any], applied: List[Dict], failed: List[Dict]):
        """Log self-improvement session"""
        try:
            with open(self.modification_log, 'r') as f:
                log_data = json.load(f)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "improvements_applied": applied,
                "improvements_failed": failed,
                "system_health": analysis["system_health"]
            }

            log_data["modifications"].append(log_entry)
            log_data["last_self_improvement"] = datetime.now().isoformat()
            log_data["improvement_count"] = len(log_data["modifications"])

            # Keep only last 50 entries
            if len(log_data["modifications"]) > 50:
                log_data["modifications"] = log_data["modifications"][-50:]

            with open(self.modification_log, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)

        except Exception as e:
            print(f"❌ Failed to log self-improvement: {e}")

    def get_self_improvement_report(self) -> Dict[str, Any]:
        """Get comprehensive report on self-improvements"""
        try:
            with open(self.modification_log, 'r') as f:
                log_data = json.load(f)

            recent_improvements = log_data["modifications"][-10:] if log_data["modifications"] else []

            return {
                "total_improvements": log_data["improvement_count"],
                "last_improvement": log_data["last_self_improvement"],
                "recent_improvements": recent_improvements,
                "success_rate": self._calculate_success_rate(recent_improvements),
                "performance_trends": self._analyze_performance_trends(),
                "user_satisfaction": self._analyze_user_satisfaction()
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_success_rate(self, improvements: List[Dict]) -> float:
        """Calculate success rate of improvements"""
        if not improvements:
            return 0.0

        total = len(improvements)
        successful = sum(1 for imp in improvements if imp["improvements_applied"] > 0)
        return (successful / total) * 100 if total > 0 else 0.0

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            with open(self.performance_data, 'r') as f:
                perf_data = json.load(f)

            trends = {}
            for metric, values in perf_data.items():
                if isinstance(values, list) and len(values) > 1:
                    recent = values[-10:] if len(values) > 10 else values
                    if recent:
                        avg_recent = sum(recent) / len(recent)
                        avg_overall = sum(values) / len(values)
                        improvement = ((avg_overall - avg_recent) / avg_overall) * 100 if avg_overall > 0 else 0
                        trends[metric] = {
                            "recent_average": avg_recent,
                            "overall_average": avg_overall,
                            "improvement_percent": improvement,
                            "trend": "improving" if improvement > 0 else "declining"
                        }

            return trends

        except Exception:
            return {}

    def _analyze_user_satisfaction(self) -> Dict[str, Any]:
        """Analyze user satisfaction trends"""
        try:
            with open(self.user_interactions, 'r') as f:
                user_data = json.load(f)

            satisfaction_scores = user_data.get("satisfaction_scores", [])
            if satisfaction_scores:
                avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
                recent_scores = satisfaction_scores[-20:] if len(satisfaction_scores) > 20 else satisfaction_scores
                recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0

                return {
                    "average_satisfaction": avg_satisfaction,
                    "recent_average": recent_avg,
                    "trend": "improving" if recent_avg > avg_satisfaction else "stable",
                    "sample_size": len(satisfaction_scores)
                }
            else:
                return {"status": "no_data"}

        except Exception:
            return {"error": "analysis_failed"}

    def emergency_rollback(self):
        """Emergency rollback system for failed improvements"""
        print("🚨 Initiating emergency rollback...")
        # Implementation for rolling back failed improvements
        # This would restore backup versions of modified files
        pass
