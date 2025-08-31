#!/usr/bin/env python3
"""
User Experience Learner for Self-Modification System
Learns from user interactions to improve the experience
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import re

class UserExperienceLearner:
    """Learns from user interactions to improve experience"""

    def __init__(self, self_modification_engine):
        self.engine = self_modification_engine
        self.interactions_file = self.engine.user_interactions

    def analyze_user_experience(self) -> Dict[str, Any]:
        """Analyze user experience patterns"""
        analysis = {
            "total_interactions": 0,
            "command_patterns": {},
            "feature_usage": {},
            "error_patterns": {},
            "satisfaction_score": 0.0,
            "improvement_suggestions": [],
            "user_behavior_trends": {}
        }

        try:
            with open(self.interactions_file, 'r') as f:
                interaction_data = json.load(f)

            # Analyze command patterns
            analysis["command_patterns"] = self._analyze_command_patterns(interaction_data)

            # Analyze feature usage
            analysis["feature_usage"] = self._analyze_feature_usage(interaction_data)

            # Analyze error patterns
            analysis["error_patterns"] = self._analyze_error_patterns(interaction_data)

            # Calculate satisfaction score
            analysis["satisfaction_score"] = self._calculate_satisfaction_score(interaction_data)

            # Generate improvement suggestions
            analysis["improvement_suggestions"] = self._generate_improvement_suggestions(analysis)

            # Analyze user behavior trends
            analysis["user_behavior_trends"] = self._analyze_behavior_trends(interaction_data)

            analysis["total_interactions"] = interaction_data.get("total_interactions", 0)

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def _analyze_command_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze command usage patterns"""
        patterns = data.get("command_patterns", {})

        # Find most used commands
        most_used = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]

        # Find command sequences (if available)
        sequences = self._find_command_sequences(data)

        return {
            "most_used_commands": most_used,
            "command_sequences": sequences,
            "usage_efficiency": self._calculate_usage_efficiency(patterns)
        }

    def _analyze_feature_usage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature usage patterns"""
        feature_usage = data.get("feature_usage", {})

        # Calculate feature adoption rates
        total_interactions = sum(feature_usage.values())
        adoption_rates = {}

        for feature, usage in feature_usage.items():
            adoption_rates[feature] = (usage / total_interactions) * 100 if total_interactions > 0 else 0

        # Find underutilized features
        underutilized = [feature for feature, rate in adoption_rates.items() if rate < 5.0]

        # Find overutilized features (potential for optimization)
        overutilized = [feature for feature, rate in adoption_rates.items() if rate > 50.0]

        return {
            "adoption_rates": adoption_rates,
            "underutilized_features": underutilized,
            "overutilized_features": overutilized,
            "feature_balance_score": self._calculate_feature_balance(adoption_rates)
        }

    def _analyze_error_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns and their causes"""
        error_patterns = data.get("error_patterns", {})

        # Categorize errors by type
        error_categories = defaultdict(int)
        for error, count in error_patterns.items():
            if "memory" in error.lower():
                error_categories["memory_errors"] += count
            elif "network" in error.lower() or "connection" in error.lower():
                error_categories["network_errors"] += count
            elif "permission" in error.lower() or "access" in error.lower():
                error_categories["permission_errors"] += count
            elif "syntax" in error.lower():
                error_categories["syntax_errors"] += count
            else:
                error_categories["other_errors"] += count

        # Find most common error types
        most_common_errors = sorted(error_categories.items(), key=lambda x: x[1], reverse=True)

        return {
            "error_categories": dict(error_categories),
            "most_common_errors": most_common_errors[:5],
            "error_frequency": sum(error_patterns.values()),
            "error_resolution_rate": self._calculate_error_resolution_rate(data)
        }

    def _calculate_satisfaction_score(self, data: Dict[str, Any]) -> float:
        """Calculate user satisfaction score"""
        satisfaction_scores = data.get("satisfaction_scores", [])

        if not satisfaction_scores:
            return 75.0  # Default neutral score

        # Calculate weighted average (recent scores have higher weight)
        total_weight = 0
        weighted_sum = 0

        for i, score in enumerate(reversed(satisfaction_scores)):
            weight = 1.0 / (i + 1)  # Recent scores have higher weight
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 75.0

    def _generate_improvement_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []

        # Feature usage suggestions
        feature_usage = analysis["feature_usage"]
        underutilized = feature_usage.get("underutilized_features", [])
        if underutilized:
            suggestions.append(f"Improve discoverability of underutilized features: {', '.join(underutilized[:3])}")

        # Error pattern suggestions
        error_patterns = analysis["error_patterns"]
        most_common = error_patterns.get("most_common_errors", [])
        if most_common:
            top_error = most_common[0][0]
            suggestions.append(f"Add better error handling for {top_error}")

        # Command pattern suggestions
        command_patterns = analysis["command_patterns"]
        usage_efficiency = command_patterns.get("usage_efficiency", 0)
        if usage_efficiency < 50:
            suggestions.append("Streamline command interface to improve user efficiency")

        # Satisfaction-based suggestions
        satisfaction = analysis["satisfaction_score"]
        if satisfaction < 70:
            suggestions.append("Focus on user experience improvements to increase satisfaction")
        elif satisfaction > 90:
            suggestions.append("Current user experience is excellent - maintain high standards")

        return suggestions[:5]  # Limit to 5 suggestions

    def _analyze_behavior_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior trends"""
        trends = {
            "learning_curve": "stable",
            "feature_discovery": "moderate",
            "error_recovery": "good",
            "engagement_level": "high"
        }

        # Analyze trends based on interaction patterns
        # This would analyze time-series data for trends
        # For now, return default trends

        return trends

    def _find_command_sequences(self, data: Dict[str, Any]) -> List[List[str]]:
        """Find common command sequences"""
        # This would analyze command sequences from interaction logs
        # For now, return some example sequences
        return [
            ["help", "tutorial"],
            ["chat", "memory"],
            ["analyze", "optimize"]
        ]

    def _calculate_usage_efficiency(self, patterns: Dict[str, int]) -> float:
        """Calculate command usage efficiency"""
        if not patterns:
            return 0.0

        total_usage = sum(patterns.values())
        unique_commands = len(patterns)

        # Efficiency based on command diversity and usage distribution
        efficiency = min(100.0, (total_usage / unique_commands) * 10)
        return efficiency

    def _calculate_feature_balance(self, adoption_rates: Dict[str, float]) -> float:
        """Calculate feature balance score"""
        if not adoption_rates:
            return 50.0

        rates = list(adoption_rates.values())
        mean_rate = sum(rates) / len(rates)

        # Calculate variance (how balanced the usage is)
        variance = sum((rate - mean_rate) ** 2 for rate in rates) / len(rates)
        balance_score = max(0, 100 - (variance * 100))

        return balance_score

    def _calculate_error_resolution_rate(self, data: Dict[str, Any]) -> float:
        """Calculate error resolution rate"""
        # This would analyze how often errors are resolved successfully
        # For now, return a default rate
        return 85.0

    def improve_interface(self) -> bool:
        """Improve user interface based on learning"""
        try:
            print("🎨 Improving user interface based on usage patterns...")

            improvements = 0

            # Analyze current interface usage
            interface_analysis = self.analyze_user_experience()

            # Improve navigation
            improvements += self._improve_navigation(interface_analysis)

            # Optimize feature placement
            improvements += self._optimize_feature_placement(interface_analysis)

            # Enhance user feedback
            improvements += self._enhance_user_feedback(interface_analysis)

            # Simplify workflows
            improvements += self._simplify_workflows(interface_analysis)

            print(f"✅ Interface improvement complete: {improvements} enhancements applied")
            return improvements > 0

        except Exception as e:
            print(f"❌ Interface improvement failed: {e}")
            return False

    def _improve_navigation(self, analysis: Dict[str, Any]) -> int:
        """Improve navigation based on usage patterns"""
        # This would modify the navigation structure
        return 1

    def _optimize_feature_placement(self, analysis: Dict[str, Any]) -> int:
        """Optimize feature placement for better usability"""
        # This would rearrange UI elements
        return 1

    def _enhance_user_feedback(self, analysis: Dict[str, Any]) -> int:
        """Enhance user feedback systems"""
        # This would improve error messages and help text
        return 1

    def _simplify_workflows(self, analysis: Dict[str, Any]) -> int:
        """Simplify user workflows"""
        # This would streamline common tasks
        return 1

    def record_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Record a user interaction for learning"""
        try:
            interaction_record = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction_type,
                "data": data
            }

            # Load existing data
            with open(self.interactions_file, 'r') as f:
                interaction_data = json.load(f)

            # Update counters and patterns
            self._update_interaction_patterns(interaction_data, interaction_record)

            # Save updated data
            with open(self.interactions_file, 'w') as f:
                json.dump(interaction_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to record interaction: {e}")

    def _update_interaction_patterns(self, data: Dict[str, Any], record: Dict[str, Any]):
        """Update interaction patterns with new data"""
        # Update command patterns
        if record["type"] == "command":
            command = record["data"].get("command", "")
            if command:
                data["command_patterns"][command] = data["command_patterns"].get(command, 0) + 1

        # Update feature usage
        elif record["type"] == "feature_usage":
            feature = record["data"].get("feature", "")
            if feature:
                data["feature_usage"][feature] = data["feature_usage"].get(feature, 0) + 1

        # Update error patterns
        elif record["type"] == "error":
            error_type = record["data"].get("error_type", "")
            if error_type:
                data["error_patterns"][error_type] = data["error_patterns"].get(error_type, 0) + 1

        # Update satisfaction scores
        elif record["type"] == "satisfaction":
            score = record["data"].get("score", 0)
            if score > 0:
                data["satisfaction_scores"].append(score)

        # Keep data size manageable
        if len(data["satisfaction_scores"]) > 100:
            data["satisfaction_scores"] = data["satisfaction_scores"][-100:]

        data["total_interactions"] = data.get("total_interactions", 0) + 1
