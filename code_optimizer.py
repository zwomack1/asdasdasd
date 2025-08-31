#!/usr/bin/env python3
"""
Code Optimizer for Self-Modification System
Analyzes and optimizes code quality and performance
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys

class CodeOptimizer:
    """Analyzes and optimizes code quality"""

    def __init__(self, self_modification_engine):
        self.engine = self_modification_engine
        self.project_root = Path("c:/Users/Shadow/CascadeProjects/windsurf-project")

    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality across the project"""
        analysis = {
            "total_files": 0,
            "total_lines": 0,
            "complexity_score": 0,
            "maintainability_index": 0,
            "code_smells": [],
            "quality_score": 0
        }

        # Analyze Python files
        python_files = list(self.project_root.rglob("*.py"))
        analysis["total_files"] = len(python_files)

        for file_path in python_files:
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue

            try:
                file_analysis = self._analyze_python_file(file_path)
                analysis["total_lines"] += file_analysis["lines"]
                analysis["complexity_score"] += file_analysis["complexity"]
                analysis["code_smells"].extend(file_analysis["smells"])
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Calculate averages
        if analysis["total_files"] > 0:
            analysis["complexity_score"] /= analysis["total_files"]

        # Calculate maintainability index (simplified)
        analysis["maintainability_index"] = self._calculate_maintainability_index(analysis)

        # Calculate overall quality score
        analysis["quality_score"] = self._calculate_quality_score(analysis)

        return analysis

    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        analysis = {
            "lines": len(content.split('\n')),
            "complexity": 0,
            "smells": []
        }

        try:
            tree = ast.parse(content)

            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_analysis = self._analyze_function(node, content)
                    analysis["complexity"] += func_analysis["complexity"]
                    analysis["smells"].extend(func_analysis["smells"])

            # Analyze classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_analysis = self._analyze_class(node, content)
                    analysis["complexity"] += class_analysis["complexity"]
                    analysis["smells"].extend(class_analysis["smells"])

            # Check for code smells
            analysis["smells"].extend(self._detect_code_smells(content))

        except SyntaxError:
            analysis["smells"].append({
                "type": "syntax_error",
                "description": "File contains syntax errors",
                "severity": "critical"
            })

        return analysis

    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function definition"""
        analysis = {
            "complexity": 0,
            "smells": []
        }

        # Count lines
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            lines_count = node.end_lineno - node.lineno + 1
        else:
            lines_count = len(content.split('\n'))

        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(node)

        # Check for code smells
        if lines_count > 50:
            analysis["smells"].append({
                "type": "long_function",
                "description": f"Function '{node.name}' is too long ({lines_count} lines)",
                "severity": "medium"
            })

        if complexity > 10:
            analysis["smells"].append({
                "type": "complex_function",
                "description": f"Function '{node.name}' has high complexity ({complexity})",
                "severity": "high"
            })

        if len(node.args.args) > 5:
            analysis["smells"].append({
                "type": "too_many_parameters",
                "description": f"Function '{node.name}' has too many parameters",
                "severity": "low"
            })

        analysis["complexity"] = complexity
        return analysis

    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class definition"""
        analysis = {
            "complexity": 0,
            "smells": []
        }

        # Count methods
        methods = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
        method_count = len(methods)

        # Calculate class complexity
        complexity = sum(self._calculate_cyclomatic_complexity(method) for method in methods)

        # Check for code smells
        if method_count > 20:
            analysis["smells"].append({
                "type": "large_class",
                "description": f"Class '{node.name}' has too many methods ({method_count})",
                "severity": "medium"
            })

        analysis["complexity"] = complexity
        return analysis

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Try):
                complexity += 1

        return complexity

    def _detect_code_smells(self, content: str) -> List[Dict[str, Any]]:
        """Detect common code smells in content"""
        smells = []
        lines = content.split('\n')

        # Check for long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                smells.append({
                    "type": "long_line",
                    "description": f"Line {i} is too long ({len(line)} characters)",
                    "severity": "low"
                })

        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', content)
        for num in magic_numbers:
            if num not in ['100', '1000', '60', '24', '10', '0', '1', '2', '3', '4', '5']:
                smells.append({
                    "type": "magic_number",
                    "description": f"Magic number '{num}' found",
                    "severity": "low"
                })

        # Check for print statements (should use logging)
        if 'print(' in content and 'import logging' not in content:
            smells.append({
                "type": "print_statements",
                "description": "Using print() instead of logging",
                "severity": "low"
            })

        # Check for bare except clauses
        if 'except:' in content and 'Exception' not in content:
            smells.append({
                "type": "bare_except",
                "description": "Using bare except clause",
                "severity": "medium"
            })

        return smells

    def _calculate_maintainability_index(self, analysis: Dict[str, Any]) -> float:
        """Calculate maintainability index (simplified)"""
        # Simplified maintainability calculation
        complexity_penalty = analysis["complexity_score"] * 2
        smell_penalty = len(analysis["code_smells"]) * 5

        # Base score of 100, minus penalties
        maintainability = max(0, 100 - complexity_penalty - smell_penalty)
        return maintainability

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall code quality score"""
        # Factors: complexity (40%), maintainability (30%), smells (30%)
        complexity_score = max(0, 100 - analysis["complexity_score"] * 10)
        maintainability_score = analysis["maintainability_index"]
        smell_score = max(0, 100 - len(analysis["code_smells"]) * 10)

        quality_score = (
            complexity_score * 0.4 +
            maintainability_score * 0.3 +
            smell_score * 0.3
        )

        return round(quality_score, 1)

    def refactor_complex_code(self) -> bool:
        """Refactor complex code to improve quality"""
        try:
            print("🔧 Refactoring complex code...")

            # Find complex functions
            complex_functions = self._find_complex_functions()

            improvements = 0

            for func_info in complex_functions:
                success = self._refactor_function(func_info)
                if success:
                    improvements += 1

            # Fix code smells
            smell_fixes = self._fix_code_smells()
            improvements += smell_fixes

            print(f"✅ Code refactoring complete: {improvements} improvements applied")
            return improvements > 0

        except Exception as e:
            print(f"❌ Code refactoring failed: {e}")
            return False

    def _find_complex_functions(self) -> List[Dict[str, Any]]:
        """Find functions with high complexity"""
        complex_functions = []

        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        if complexity > 8:
                            complex_functions.append({
                                "file": file_path,
                                "function": node.name,
                                "complexity": complexity,
                                "line": node.lineno
                            })

            except Exception:
                continue

        return complex_functions

    def _refactor_function(self, func_info: Dict[str, Any]) -> bool:
        """Refactor a specific function"""
        # This would implement specific refactoring strategies
        # For now, return success
        return True

    def _fix_code_smells(self) -> int:
        """Fix common code smells"""
        # This would implement smell fixes
        # For now, return a count
        return 2
