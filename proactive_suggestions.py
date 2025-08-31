#!/usr/bin/env python3
"""
Proactive Suggestion System for HRM-Gemini AI
Analyzes user behavior and provides intelligent suggestions
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ProactiveSuggestions:
    def __init__(self, cli_interface):
        self.cli = cli_interface
        self.user_behavior = {
            'commands_used': {},
            'command_sequences': [],
            'errors_encountered': [],
            'time_spent': {},
            'last_suggestion_time': None,
            'suggestion_count': 0
        }
        self.suggestion_templates = {
            'new_user': [
                "Welcome! Try the 'tutorial' command to learn about HRM-Gemini features!",
                "New to the system? Type 'help' to see all available commands.",
                "Tip: Use 'status' to check system health and performance."
            ],
            'frequent_optimizer': [
                "You optimize code often! Try 'autocode' for automatic optimization.",
                "Consider using 'evolution' to let HRM continuously improve your code.",
                "Pro tip: Combine 'optimize' with 'train' for better results."
            ],
            'error_prone': [
                "Experiencing errors? Try 'dashboard' to check system health.",
                "Consider using 'tutorial' to refresh your knowledge of commands.",
                "Tip: Use 'report' to export detailed performance data for troubleshooting."
            ],
            'power_user': [
                "Power user detected! Try 'evolution' for continuous system improvement.",
                "Advanced tip: Use 'train' with different sources to enhance AI capabilities.",
                "Consider 'dashboard' for real-time performance monitoring."
            ],
            'idle_user': [
                "Been inactive? HRM is always ready to help!",
                "Try 'chat' to have a conversation with the AI.",
                "Use 'status' to see what HRM has been up to."
            ]
        }

    def track_command(self, command):
        """Track user command usage"""
        if command not in self.user_behavior['commands_used']:
            self.user_behavior['commands_used'][command] = 0
        self.user_behavior['commands_used'][command] += 1

        # Track command sequences (last 5 commands)
        self.user_behavior['command_sequences'].append(command)
        if len(self.user_behavior['command_sequences']) > 5:
            self.user_behavior['command_sequences'] = self.user_behavior['command_sequences'][-5:]

    def track_error(self, error_type, command=None):
        """Track errors encountered"""
        self.user_behavior['errors_encountered'].append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'command': command
        })

        # Keep only last 10 errors
        if len(self.user_behavior['errors_encountered']) > 10:
            self.user_behavior['errors_encountered'] = self.user_behavior['errors_encountered'][-10:]

    def analyze_behavior(self):
        """Analyze user behavior and determine user type"""
        commands = self.user_behavior['commands_used']
        total_commands = sum(commands.values())

        if total_commands < 5:
            return 'new_user'
        elif commands.get('optimize', 0) > total_commands * 0.3:
            return 'frequent_optimizer'
        elif len(self.user_behavior['errors_encountered']) > total_commands * 0.2:
            return 'error_prone'
        elif total_commands > 20 and len(commands) > 8:
            return 'power_user'
        elif self._check_idle_time():
            return 'idle_user'
        else:
            return 'regular_user'

    def _check_idle_time(self):
        """Check if user has been idle"""
        if not self.user_behavior['command_sequences']:
            return True

        last_command_time = self.user_behavior.get('last_command_time')
        if last_command_time:
            idle_time = datetime.now() - last_command_time
            return idle_time > timedelta(minutes=5)
        return False

    def should_suggest(self):
        """Determine if a suggestion should be shown"""
        # Don't suggest too frequently
        if self.user_behavior['last_suggestion_time']:
            time_since_last = datetime.now() - self.user_behavior['last_suggestion_time']
            if time_since_last < timedelta(minutes=3):
                return False

        # Don't suggest if user has seen too many suggestions
        if self.user_behavior['suggestion_count'] > 10:
            return False

        # Suggest based on behavior patterns
        user_type = self.analyze_behavior()
        return user_type != 'regular_user' or random.random() < 0.1  # 10% chance for regular users

    def get_suggestion(self):
        """Get an appropriate suggestion based on user behavior"""
        user_type = self.analyze_behavior()

        if user_type in self.suggestion_templates:
            suggestions = self.suggestion_templates[user_type]
            suggestion = random.choice(suggestions)

            # Mark suggestion as shown
            self.user_behavior['last_suggestion_time'] = datetime.now()
            self.user_behavior['suggestion_count'] += 1

            return suggestion

        return None

    def get_contextual_help(self, current_command=None):
        """Provide contextual help based on current command or state"""
        if current_command:
            help_suggestions = {
                'optimize': [
                    "💡 Tip: After optimizing, try 'train file' to improve future optimizations.",
                    "💡 Pro tip: Use 'dashboard' to see optimization performance metrics.",
                    "💡 Advanced: Combine 'optimize' with 'evolution' for continuous improvement."
                ],
                'recode': [
                    "💡 Tip: Be specific in your recoding instructions for better results.",
                    "💡 Try: 'Make this function more efficient and add error handling'",
                    "💡 Use 'history' to see your previous recoding attempts."
                ],
                'train': [
                    "💡 Tip: Training from files gives the most consistent results.",
                    "💡 Try training from different sources to enhance versatility.",
                    "💡 Use 'status' to monitor training progress."
                ],
                'chat': [
                    "💡 Tip: HRM learns from conversations - ask technical questions!",
                    "💡 Try asking about code optimization or system architecture.",
                    "💡 HRM can help with debugging and problem-solving."
                ]
            }

            if current_command in help_suggestions:
                return random.choice(help_suggestions[current_command])

        # General contextual help
        general_tips = [
            "💡 Use 'dashboard' to monitor system performance in real-time.",
            "💡 Try 'tutorial' anytime to learn new features.",
            "💡 HRM continuously improves - check 'status' to see enhancements.",
            "💡 Use 'report' to export detailed performance data.",
            "💡 Combine multiple commands for powerful workflows."
        ]

        return random.choice(general_tips)

    def show_proactive_suggestion(self):
        """Show a proactive suggestion to the user"""
        if self.should_suggest():
            suggestion = self.get_suggestion()
            if suggestion:
                print(f"\n💡 {suggestion}")
                return True
        return False

    def show_contextual_help(self, command=None):
        """Show contextual help"""
        help_text = self.get_contextual_help(command)
        if help_text:
            print(f"\n{help_text}")
            return True
        return False
