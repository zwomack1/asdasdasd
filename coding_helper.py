#!/usr/bin/env python3
"""
HRM-Gemini Coding Helper - Interactive Code Analysis and Improvement System
Provides real-time coding assistance, suggestions, and refactoring recommendations
"""

import sys
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import inspect

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class CodingHelper:
    """Interactive coding helper for HRM-Gemini AI system"""

    def __init__(self, gemini_model=None):
        self.gemini_model = gemini_model
        self.code_patterns = self._load_code_patterns()
        self.best_practices = self._load_best_practices()
        self.analysis_cache = {}

    def _load_code_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common code patterns and their improvements"""
        return {
            "long_function": {
                "description": "Functions longer than 50 lines",
                "suggestion": "Break down into smaller, focused functions",
                "severity": "medium"
            },
            "nested_loops": {
                "description": "Deeply nested loops (>3 levels)",
                "suggestion": "Extract inner loops to separate functions or use list comprehensions",
                "severity": "high"
            },
            "global_variables": {
                "description": "Excessive use of global variables",
                "suggestion": "Pass variables as parameters or use class attributes",
                "severity": "medium"
            },
            "magic_numbers": {
                "description": "Hard-coded numeric values",
                "suggestion": "Replace with named constants",
                "severity": "low"
            },
            "long_lines": {
                "description": "Lines longer than 100 characters",
                "suggestion": "Break long lines for better readability",
                "severity": "low"
            },
            "complex_conditionals": {
                "description": "Complex conditional statements",
                "suggestion": "Extract conditions to well-named variables or functions",
                "severity": "medium"
            },
            "duplicate_code": {
                "description": "Repeated code blocks",
                "suggestion": "Extract to reusable functions or methods",
                "severity": "high"
            }
        }

    def _load_best_practices(self) -> Dict[str, str]:
        """Load coding best practices"""
        return {
            "error_handling": "Always use try-except blocks for potential errors. Use specific exception types.",
            "docstrings": "Add docstrings to all functions, classes, and modules explaining purpose and parameters.",
            "type_hints": "Use type hints for function parameters and return values for better code clarity.",
            "naming": "Use descriptive variable and function names. Follow snake_case for functions/variables.",
            "comments": "Add comments for complex logic, not obvious code. Keep comments up-to-date.",
            "modularity": "Break large functions into smaller, focused functions with single responsibilities.",
            "testing": "Write unit tests for critical functions. Test edge cases and error conditions.",
            "logging": "Use logging instead of print statements for debugging and monitoring.",
            "constants": "Define magic numbers as named constants at the top of the file.",
            "imports": "Group imports: standard library, third-party, local. Remove unused imports.",
            "whitespace": "Use consistent indentation (4 spaces). Add blank lines between logical sections."
        }

    def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code and provide comprehensive feedback"""
        if language.lower() == "python":
            return self._analyze_python_code(code)
        else:
            return self._analyze_generic_code(code, language)

    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code specifically"""
        analysis = {
            "issues": [],
            "suggestions": [],
            "improvements": [],
            "metrics": {},
            "best_practices": []
        }

        try:
            # Parse the code
            tree = ast.parse(code)

            # Basic metrics
            lines = code.split('\n')
            analysis["metrics"] = {
                "total_lines": len(lines),
                "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                "functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "classes": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "imports": len([node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)])
            }

            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._analyze_function(node, analysis)

            # Check for code patterns
            self._check_code_patterns(code, analysis)

            # Generate improvement suggestions
            self._generate_improvements(code, analysis)

            # Add best practices suggestions
            self._suggest_best_practices(code, analysis)

        except SyntaxError as e:
            analysis["issues"].append({
                "type": "syntax_error",
                "description": f"Syntax error: {e.msg}",
                "line": e.lineno,
                "severity": "critical"
            })

        return analysis

    def _analyze_function(self, node: ast.FunctionDef, analysis: Dict[str, Any]):
        """Analyze a function definition"""
        # Check function length
        if len(node.body) > 50:
            analysis["issues"].append({
                "type": "long_function",
                "description": f"Function '{node.name}' is too long ({len(node.body)} lines)",
                "line": node.lineno,
                "severity": "medium",
                "suggestion": "Break down into smaller, focused functions"
            })

        # Check for too many parameters
        if len(node.args.args) > 5:
            analysis["issues"].append({
                "type": "too_many_parameters",
                "description": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                "line": node.lineno,
                "severity": "low",
                "suggestion": "Consider using a configuration object or class"
            })

        # Check for nested functions (generally not recommended)
        nested_functions = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n != node]
        if nested_functions:
            analysis["issues"].append({
                "type": "nested_functions",
                "description": f"Function '{node.name}' contains nested functions",
                "line": node.lineno,
                "severity": "low",
                "suggestion": "Move nested functions to module level"
            })

    def _check_code_patterns(self, code: str, analysis: Dict[str, Any]):
        """Check for common code patterns that need improvement"""
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 100:
                analysis["issues"].append({
                    "type": "long_line",
                    "description": f"Line {i} is too long ({len(line)} characters)",
                    "line": i,
                    "severity": "low",
                    "suggestion": "Break long lines for better readability"
                })

            # Check for magic numbers
            magic_numbers = re.findall(r'\b\d{2,}\b', line)
            for num in magic_numbers:
                if num not in ['100', '1000', '60', '24']:  # Common non-magic numbers
                    analysis["issues"].append({
                        "type": "magic_number",
                        "description": f"Magic number '{num}' found on line {i}",
                        "line": i,
                        "severity": "low",
                        "suggestion": "Replace with a named constant"
                    })

        # Check for deeply nested structures
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)

        max_indent = max(indent_levels) if indent_levels else 0
        if max_indent > 12:  # More than 3 levels of indentation
            analysis["issues"].append({
                "type": "deep_nesting",
                "description": f"Code has deep nesting (max indent: {max_indent} spaces)",
                "line": 1,
                "severity": "medium",
                "suggestion": "Reduce nesting by extracting functions or using early returns"
            })

    def _generate_improvements(self, code: str, analysis: Dict[str, Any]):
        """Generate improvement suggestions based on analysis"""
        improvements = []

        # Based on metrics
        metrics = analysis.get("metrics", {})

        if metrics.get("total_lines", 0) > 300:
            improvements.append("Consider breaking this file into multiple modules")

        if metrics.get("functions", 0) > 15:
            improvements.append("Consider grouping related functions into classes")

        if metrics.get("code_lines", 0) / max(metrics.get("total_lines", 1), 1) < 0.3:
            improvements.append("Add more comments and docstrings to improve readability")

        # Based on issues
        issues = analysis.get("issues", [])
        high_priority = [issue for issue in issues if issue.get("severity") == "high"]
        if len(high_priority) > 3:
            improvements.append("Address high-priority issues first - consider refactoring")

        analysis["improvements"] = improvements

    def _suggest_best_practices(self, code: str, analysis: Dict[str, Any]):
        """Suggest coding best practices"""
        suggestions = []

        # Check for missing docstrings
        if 'def ' in code and '"""' not in code:
            suggestions.append("docstrings")

        # Check for type hints
        if 'def ' in code and '->' not in code:
            suggestions.append("type_hints")

        # Check for error handling
        if 'try:' not in code and ('open(' in code or 'requests.' in code):
            suggestions.append("error_handling")

        # Check for naming conventions
        if re.search(r'\b[a-z]+_[A-Z]+\b', code):  # Mixed case variables
            suggestions.append("naming")

        # Check for imports organization
        if 'import' in code:
            suggestions.append("imports")

        analysis["best_practices"] = [
            {"practice": practice, "description": self.best_practices.get(practice, "")}
            for practice in suggestions[:5]  # Limit to 5 suggestions
        ]

    def _analyze_generic_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code in other languages generically"""
        lines = code.split('\n')

        analysis = {
            "issues": [],
            "suggestions": [],
            "improvements": [],
            "metrics": {
                "total_lines": len(lines),
                "code_lines": len([line for line in lines if line.strip()]),
                "language": language
            },
            "best_practices": []
        }

        # Generic analysis for any language
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # Generic long line check
                analysis["issues"].append({
                    "type": "long_line",
                    "description": f"Line {i} is too long",
                    "line": i,
                    "severity": "low"
                })

        analysis["improvements"].append(f"Consider language-specific best practices for {language}")
        analysis["best_practices"].append({
            "practice": "language_specific",
            "description": f"Follow {language} coding standards and conventions"
        })

        return analysis

    def generate_code_suggestion(self, code: str, request: str, language: str = "python") -> Dict[str, Any]:
        """Generate code improvement suggestions using AI"""
        if not self.gemini_model:
            return {"error": "Gemini model not available for AI suggestions"}

        prompt = f"""
You are an expert {language} developer and code reviewer. Analyze the following code and provide specific, actionable suggestions for improvement based on the user's request: "{request}"

Code to analyze:
```python
{code}
```

Please provide:
1. Specific code improvements with before/after examples
2. Best practices recommendations
3. Performance optimizations if applicable
4. Maintainability improvements
5. Any security considerations

Focus on practical, implementable suggestions that will actually improve the code quality and functionality.

Response format:
- Use clear headings for different types of suggestions
- Provide specific code examples where helpful
- Explain why each suggestion improves the code
- Prioritize the most impactful improvements first
"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return {
                "success": True,
                "suggestions": response.text,
                "request": request,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"AI suggestion generation failed: {e}",
                "request": request
            }

    def refactor_code_suggestion(self, code: str, refactor_type: str) -> Dict[str, Any]:
        """Generate refactoring suggestions"""
        refactoring_prompts = {
            "extract_function": "Extract repeated code into reusable functions",
            "simplify_conditionals": "Simplify complex conditional statements",
            "improve_naming": "Improve variable and function names for clarity",
            "reduce_complexity": "Reduce cyclomatic complexity",
            "optimize_performance": "Optimize for better performance",
            "improve_readability": "Improve code readability and structure"
        }

        if refactor_type not in refactoring_prompts:
            return {"error": f"Unknown refactoring type: {refactor_type}"}

        if not self.gemini_model:
            return {"error": "Gemini model not available for refactoring suggestions"}

        prompt = f"""
You are an expert code refactoring specialist. The user wants to {refactoring_prompts[refactor_type]}.

Analyze this code and provide a complete refactoring solution:

```python
{code}
```

Provide:
1. Analysis of current code issues
2. Step-by-step refactoring approach
3. Refactored code with explanations
4. Benefits of the refactoring
5. Any potential risks or considerations

Make the refactoring practical and maintainable.
"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return {
                "success": True,
                "refactoring": response.text,
                "type": refactor_type,
                "original_code_length": len(code)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Refactoring suggestion failed: {e}",
                "type": refactor_type
            }

    def get_quick_fixes(self, code: str) -> List[Dict[str, Any]]:
        """Get quick fix suggestions for common issues"""
        fixes = []

        # Check for common issues
        if 'print(' in code and 'import logging' not in code:
            fixes.append({
                "issue": "Using print() instead of logging",
                "fix": "Consider using logging module for better debugging",
                "code_example": "import logging\nlogging.info('message')"
            })

        if 'except:' in code and 'Exception' not in code:
            fixes.append({
                "issue": "Bare except clause",
                "fix": "Specify exception types for better error handling",
                "code_example": "except ValueError as e:"
            })

        if len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]) > 100:
            fixes.append({
                "issue": "Large function/file",
                "fix": "Consider breaking into smaller functions",
                "code_example": "def smaller_function():\n    # Extracted logic here\n    pass"
            })

        return fixes

    def analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze the overall project structure and provide recommendations"""
        project_path = Path(project_path)

        if not project_path.exists():
            return {"error": "Project path does not exist"}

        analysis = {
            "structure": {},
            "issues": [],
            "recommendations": []
        }

        # Analyze directory structure
        python_files = list(project_path.rglob("*.py"))
        analysis["structure"] = {
            "total_files": len(python_files),
            "directories": len([d for d in project_path.rglob("*") if d.is_dir()]),
            "python_files": len(python_files)
        }

        # Check for common structural issues
        if len(python_files) > 20 and len(list(project_path.glob("*.py"))) > 10:
            analysis["issues"].append("Too many Python files in root directory")
            analysis["recommendations"].append("Organize files into packages/modules")

        # Check for missing __init__.py files
        dirs_without_init = []
        for dir_path in project_path.rglob("*"):
            if dir_path.is_dir() and not (dir_path / "__init__.py").exists():
                py_files = list(dir_path.glob("*.py"))
                if py_files:
                    dirs_without_init.append(str(dir_path.relative_to(project_path)))

        if dirs_without_init:
            analysis["issues"].append("Missing __init__.py files in packages")
            analysis["recommendations"].append("Add __init__.py files to make directories proper Python packages")

        return analysis
