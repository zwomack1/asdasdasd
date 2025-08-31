#!/usr/bin/env python3
"""
Interactive Tutorial System for HRM-Gemini AI
Guides users through system features and capabilities
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class InteractiveTutorial:
    def __init__(self, cli_interface):
        self.cli = cli_interface
        self.current_step = 0
        self.tutorial_steps = [
            {
                'title': 'Welcome to HRM-Gemini AI',
                'description': 'Learn how to use your intelligent AI assistant',
                'content': 'HRM-Gemini is powered by advanced Hierarchical Recurrent Memory (HRM) technology that makes intelligent decisions and continuously improves itself.',
                'action': 'show_welcome'
            },
            {
                'title': 'Basic Commands',
                'description': 'Explore the fundamental commands',
                'content': 'Try these essential commands: chat, optimize, recode, status',
                'action': 'demo_basic_commands'
            },
            {
                'title': 'Code Optimization',
                'description': 'Learn to optimize your code',
                'content': 'HRM can analyze and optimize your Python code for better performance and readability.',
                'action': 'demo_optimization'
            },
            {
                'title': 'Code Rewriting',
                'description': 'Master code rewriting capabilities',
                'content': 'Transform your code with specific instructions using Gemini AI integration.',
                'action': 'demo_rewriting'
            },
            {
                'title': 'AI Training',
                'description': 'Train the AI from various sources',
                'content': 'Enhance HRM intelligence by training it from files, websites, or human reasoning patterns.',
                'action': 'demo_training'
            },
            {
                'title': 'Advanced Features',
                'description': 'Explore advanced capabilities',
                'content': 'Discover autocoding, evolution, and proactive system monitoring.',
                'action': 'demo_advanced'
            },
            {
                'title': 'Customization',
                'description': 'Customize your experience',
                'content': 'Learn about system settings, performance monitoring, and personalization options.',
                'action': 'demo_customization'
            }
        ]

    def start_tutorial(self):
        """Begin the interactive tutorial"""
        print("\n" + "="*60)
        print("🎓 HRM-GEMINI AI INTERACTIVE TUTORIAL")
        print("="*60)
        print("This tutorial will guide you through all system features.")
        print("Type 'next' to continue, 'back' to go back, or 'quit' to exit.")
        print("="*60 + "\n")

        self.current_step = 0
        self.show_current_step()

    def show_current_step(self):
        """Display the current tutorial step"""
        if self.current_step >= len(self.tutorial_steps):
            self.end_tutorial()
            return

        step = self.tutorial_steps[self.current_step]

        print(f"\n📚 Step {self.current_step + 1}/{len(self.tutorial_steps)}: {step['title']}")
        print(f"📝 {step['description']}")
        print(f"💡 {step['content']}")
        print("\n" + "-"*50)

        # Execute step-specific action
        action_method = getattr(self, f"_{step['action']}", None)
        if action_method:
            action_method()

        self._get_user_input()

    def _get_user_input(self):
        """Get user input for tutorial navigation"""
        while True:
            try:
                choice = input("\nTutorial> ").strip().lower()

                if choice in ['next', 'n', '']:
                    self.current_step += 1
                    self.show_current_step()
                    break
                elif choice in ['back', 'b', 'prev', 'previous']:
                    if self.current_step > 0:
                        self.current_step -= 1
                        self.show_current_step()
                    else:
                        print("You're at the beginning of the tutorial.")
                    break
                elif choice in ['quit', 'exit', 'q']:
                    self.end_tutorial()
                    break
                elif choice in ['help', 'h', '?']:
                    self._show_tutorial_help()
                elif choice == 'jump':
                    self._jump_to_step()
                    break
                else:
                    print("Commands: next/back/quit/help/jump")

            except KeyboardInterrupt:
                print("\nTutorial interrupted.")
                self.end_tutorial()
                break
            except Exception as e:
                print(f"Error: {e}")

    def _show_tutorial_help(self):
        """Show tutorial help"""
        print("\n📖 Tutorial Commands:")
        print("  next/n     - Go to next step")
        print("  back/b     - Go to previous step")
        print("  quit/q     - Exit tutorial")
        print("  help/h     - Show this help")
        print("  jump       - Jump to specific step")
        print("  Enter      - Same as 'next'")

    def _jump_to_step(self):
        """Allow user to jump to a specific step"""
        try:
            step_num = int(input(f"Enter step number (1-{len(self.tutorial_steps)}): "))
            if 1 <= step_num <= len(self.tutorial_steps):
                self.current_step = step_num - 1
                self.show_current_step()
            else:
                print(f"Invalid step number. Please enter 1-{len(self.tutorial_steps)}")
        except ValueError:
            print("Please enter a valid number.")

    def _show_welcome(self):
        """Welcome step action"""
        print("🚀 Your AI assistant is ready to help!")
        print("💡 Tip: Type 'help' anytime to see available commands.")

    def _demo_basic_commands(self):
        """Demonstrate basic commands"""
        print("🛠️  Let's try some basic commands:")
        print("   • 'status'  - Check system status")
        print("   • 'chat'    - Start a conversation")
        print("   • 'help'    - Show all commands")

        demo_commands = ['status']
        for cmd in demo_commands:
            print(f"\n🔄 Demonstrating '{cmd}' command...")
            time.sleep(1)
            # Simulate command execution
            print(f"✅ Command '{cmd}' executed successfully!")

    def _demo_optimization(self):
        """Demonstrate code optimization"""
        print("🔧 Code Optimization Demo:")
        print("   HRM can analyze and improve your code automatically.")

        sample_code = "def calculate(x,y):return x+y if x>0 else 0"
        print(f"📝 Sample code: {sample_code}")
        print("   → HRM would optimize this for better performance and readability!")

    def _demo_rewriting(self):
        """Demonstrate code rewriting"""
        print("🔄 Code Rewriting Demo:")
        print("   Transform code with specific instructions using Gemini AI.")

        print("📝 Example: 'Make this function more efficient'")
        print("   → HRM + Gemini would rewrite the code accordingly!")

    def _demo_training(self):
        """Demonstrate AI training"""
        print("🎓 AI Training Capabilities:")
        print("   • Train from local files")
        print("   • Learn from websites")
        print("   • Absorb human reasoning patterns")
        print("   → Continuous intelligence improvement!")

    def _demo_advanced(self):
        """Demonstrate advanced features"""
        print("🚀 Advanced Features:")
        print("   • Auto-coding: Monitors and improves code automatically")
        print("   • Evolution: System continuously enhances itself")
        print("   • Proactive monitoring: Prevents issues before they occur")

    def _demo_customization(self):
        """Demonstrate customization options"""
        print("⚙️ Customization Options:")
        print("   • Performance settings")
        print("   • Interface preferences")
        print("   • Learning parameters")
        print("   • Integration options")

    def end_tutorial(self):
        """End the tutorial"""
        print("\n" + "="*60)
        print("🎉 TUTORIAL COMPLETED!")
        print("="*60)
        print("You now know how to use HRM-Gemini AI effectively!")
        print("\n📚 Quick Reference:")
        print("   • Type 'help' for command list")
        print("   • Use 'tutorial' to restart this tutorial")
        print("   • Explore features at your own pace")
        print("\n💡 Remember: HRM is always learning and improving!")
        print("="*60 + "\n")

        self.current_step = 0
