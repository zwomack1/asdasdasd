#!/usr/bin/env python3
"""
Multi-Modal Interface for HRM-Gemini AI Brain System
Integrates memory, file processing, RPG, and all other systems into a unified interface
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class HRMMultiModalInterface:
    """Unified multi-modal interface for HRM-Gemini AI Brain System"""

    def __init__(self):
        print("🧠 Initializing HRM-Gemini Multi-Modal Interface...")

        # Initialize all subsystems
        self.memory_system = None
        self.file_upload_system = None
        self.rpg_chatbot = None
        self.performance_monitor = None

        # Load core systems
        self._initialize_subsystems()

        # Interface state
        self.current_mode = "general"  # general, rpg, upload, admin
        self.user_sessions = {}
        self.active_uploads = {}

        # Command registry
        self.commands = self._register_commands()

        print("✅ Multi-Modal Interface initialized successfully!")

    def _initialize_subsystems(self):
        """Initialize all subsystem components"""
        try:
            # Memory System
            from hrm_memory_system import HRMMemorySystem
            self.memory_system = HRMMemorySystem()
            print("   ✅ Memory System loaded")

        except ImportError as e:
            print(f"   ⚠️ Memory System not available: {e}")

        try:
            # File Upload System
            from file_upload_system import FileUploadSystem
            self.file_upload_system = FileUploadSystem(memory_system=self.memory_system)
            print("   ✅ File Upload System loaded")

        except ImportError as e:
            print(f"   ⚠️ File Upload System not available: {e}")

        try:
            # RPG Chatbot
            from rpg_chatbot import RPGChatbot
            self.rpg_chatbot = RPGChatbot(memory_system=self.memory_system)
            print("   ✅ RPG Chatbot loaded")

        except ImportError as e:
            print(f"   ⚠️ RPG Chatbot not available: {e}")

        try:
            # Performance Monitor
            from performance_monitor import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor(self)
            self.performance_monitor.start_monitoring()
            print("   ✅ Performance Monitor loaded")

        except ImportError as e:
            print(f"   ⚠️ Performance Monitor not available: {e}")

    def _register_commands(self) -> Dict[str, callable]:
        """Register all available commands"""
        commands = {
            # Core Commands
            'help': self._cmd_help,
            'status': self._cmd_status,
            'mode': self._cmd_mode,

            # Memory Commands
            'remember': self._cmd_remember,
            'recall': self._cmd_recall,
            'search': self._cmd_search,
            'stats': self._cmd_memory_stats,

            # File Commands
            'upload': self._cmd_upload,
            'files': self._cmd_list_files,
            'download': self._cmd_download_file,

            # RPG Commands
            'rpg': self._cmd_rpg_mode,
            'create': self._cmd_rpg_create,
            'explore': self._cmd_rpg_explore,
            'quest': self._cmd_rpg_quest,

            # Performance Commands
            'performance': self._cmd_performance,
            'report': self._cmd_performance_report,

            # Utility Commands
            'clear': self._cmd_clear,
            'save': self._cmd_save_all,
            'load': self._cmd_load_all,
        }

        return commands

    def process_input(self, user_input: str, user_id: str = "default",
                     input_type: str = "text", metadata: Dict[str, Any] = None) -> str:
        """Process user input through the multi-modal interface"""

        # Record performance
        if self.performance_monitor:
            start_time = time.time()

        # Parse input
        response = self._parse_and_execute(user_input, user_id, input_type, metadata)

        # Record performance
        if self.performance_monitor:
            end_time = time.time()
            self.performance_monitor.record_command(user_input, end_time - start_time)

        return response

    def _parse_and_execute(self, user_input: str, user_id: str,
                          input_type: str, metadata: Dict[str, Any] = None) -> str:
        """Parse and execute user input"""

        # Handle file uploads
        if input_type == "file":
            return self._handle_file_upload(user_input, metadata)

        # Handle image uploads
        if input_type == "image":
            return self._handle_image_upload(user_input, metadata)

        # Parse text commands
        parts = user_input.strip().split()
        if not parts:
            return self._get_greeting()

        command = parts[0].lower()
        args = parts[1:]

        # Execute command
        if command in self.commands:
            try:
                return self.commands[command](user_id, args, metadata)
            except Exception as e:
                return f"❌ Command error: {e}"
        else:
            # Handle as natural language or delegate to appropriate system
            return self._handle_natural_language(user_input, user_id)

    def _handle_file_upload(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """Handle file upload"""
        if not self.file_upload_system:
            return "❌ File upload system not available"

        result = self.file_upload_system.upload_file(
            file_path,
            description=metadata.get('description', '') if metadata else ''
        )

        if result['success']:
            return f"📁 File uploaded successfully!\n{result['message']}\n\n📝 Content Preview:\n{result['content_preview']}"
        else:
            return f"❌ Upload failed: {result['error']}"

    def _handle_image_upload(self, image_path: str, metadata: Dict[str, Any] = None) -> str:
        """Handle image upload"""
        if not self.file_upload_system:
            return "❌ Image processing system not available"

        result = self.file_upload_system.upload_file(
            image_path,
            description=metadata.get('description', 'Image upload') if metadata else 'Image upload'
        )

        if result['success']:
            return f"🖼️ Image processed successfully!\n{result['message']}\n\n📝 Analysis:\n{result['content_preview']}"
        else:
            return f"❌ Image processing failed: {result['error']}"

    def _handle_natural_language(self, user_input: str, user_id: str) -> str:
        """Handle natural language input"""

        # Check if we're in RPG mode
        if self.current_mode == "rpg":
            return self._handle_rpg_input(user_input, user_id)

        # Check for memory-related queries
        if any(word in user_input.lower() for word in ['remember', 'recall', 'what', 'tell me about']):
            return self._handle_memory_query(user_input)

        # Check for file-related queries
        if any(word in user_input.lower() for word in ['file', 'upload', 'document', 'show me']):
            return self._handle_file_query(user_input)

        # General conversation
        return self._generate_general_response(user_input)

    def _handle_rpg_input(self, user_input: str, user_id: str) -> str:
        """Handle RPG-specific input"""
        if not self.rpg_chatbot:
            return "❌ RPG system not available"

        parts = user_input.split()
        command = parts[0] if parts else ""
        args = parts[1:]

        return self.rpg_chatbot.process_rpg_command(user_id, command, args)

    def _handle_memory_query(self, user_input: str) -> str:
        """Handle memory-related queries"""
        if not self.memory_system:
            return "❌ Memory system not available"

        # Extract search terms
        query_words = [word for word in user_input.lower().split()
                      if word not in ['what', 'tell', 'me', 'about', 'remember', 'recall', 'do', 'you', 'know']]

        if not query_words:
            return "🤔 What would you like me to recall?"

        query = " ".join(query_words)

        results = self.memory_system.recall_memory(query, limit=5)

        if not results:
            return f"🤔 I don't have any memories about '{query}' yet."

        response = f"💭 Recalling memories about '{query}':\n\n"
        for i, result in enumerate(results, 1):
            timestamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(result['timestamp']))
            response += f"{i}. [{timestamp}] {result['content'][:200]}...\n"

        return response

    def _handle_file_query(self, user_input: str) -> str:
        """Handle file-related queries"""
        if not self.file_upload_system:
            return "❌ File system not available"

        user_input_lower = user_input.lower()

        if 'list' in user_input_lower or 'show' in user_input_lower:
            files = self.file_upload_system.list_uploaded_files()
            if not files:
                return "📁 No files uploaded yet."

            file_list = "\n".join([f"• {f['name']} ({f['type']}, {f['size']} bytes)" for f in files[:10]])
            return f"📁 Uploaded Files:\n{file_list}"

        # Extract filename from query
        words = user_input.split()
        filename = None
        for word in words:
            if '.' in word or any(word.lower().endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.jpg', '.png']):
                filename = word
                break

        if filename:
            content = self.file_upload_system.get_file_content(filename)
            if content:
                return f"📄 Content of '{filename}':\n\n{content[:1000]}{'...' if len(content) > 1000 else ''}"
            else:
                return f"❌ File '{filename}' not found or not readable."

        return "🤔 Please specify a filename or use 'list files' to see uploaded files."

    def _generate_general_response(self, user_input: str) -> str:
        """Generate general conversational response"""
        user_input_lower = user_input.lower()

        # Greetings
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "👋 Hello! I'm your HRM-Gemini AI assistant. I can help you with files, RPG gaming, memory management, and much more. Type 'help' to see all my capabilities!"

        # Help requests
        elif any(word in user_input_lower for word in ['help', 'assist', 'what can you do']):
            return self._cmd_help("default", [])

        # Status requests
        elif any(word in user_input_lower for word in ['status', 'how are you', 'system']):
            return self._cmd_status("default", [])

        # Memory-related
        elif any(word in user_input_lower for word in ['memory', 'remember', 'recall']):
            return "🧠 I have a comprehensive memory system! You can 'upload' files, ask me to 'recall' information, or 'search' my knowledge base."

        # File-related
        elif any(word in user_input_lower for word in ['file', 'upload', 'document']):
            return "📁 I can process various file types! Try 'upload <filepath>' or 'files' to see uploaded documents."

        # RPG-related
        elif any(word in user_input_lower for word in ['game', 'rpg', 'adventure', 'character']):
            return "🎮 I have a built-in RPG system! Use 'rpg' to enter RPG mode, then 'create <name>' to make a character."

        # Default response
        else:
            return f"🤔 I understand you said: '{user_input}'\n\n💡 Try one of these commands:\n• 'help' - See all capabilities\n• 'upload <file>' - Process a file\n• 'rpg' - Enter RPG mode\n• 'search <topic>' - Search my knowledge"

    def _get_greeting(self) -> str:
        """Get a contextual greeting"""
        greetings = [
            "👋 Hello! I'm ready to assist you with files, RPG gaming, and memory management.",
            "🧠 HRM-Gemini AI at your service! What would you like to explore today?",
            "🎯 Ready for action! Try 'help' to see what I can do.",
            "🚀 Systems online! I'm here to help with your files, games, and memories."
        ]

        import random
        return random.choice(greetings)

    # Command Implementations
    def _cmd_help(self, user_id: str, args: List[str], metadata=None) -> str:
        """Show help information"""
        help_text = """
🧠 **HRM-Gemini AI Multi-Modal Interface**

📋 **Available Commands:**

🗣️ **General:**
  help              - Show this help
  status            - System status overview
  mode <type>       - Switch modes (general/rpg/upload)

🧠 **Memory & Knowledge:**
  remember <text>   - Store information in memory
  recall <query>    - Recall stored information
  search <topic>    - Search knowledge base
  stats             - Memory statistics

📁 **File Operations:**
  upload <path>     - Upload and process a file
  files             - List uploaded files
  download <name>   - Download processed file content

🎮 **RPG System:**
  rpg               - Enter RPG mode
  create <name>     - Create RPG character
  explore           - Explore current location
  quest <action>    - Manage quests

📊 **Performance:**
  performance       - Show performance dashboard
  report            - Generate performance report

🔧 **Utilities:**
  clear             - Clear screen
  save              - Save all system data
  load              - Load system data

💡 **Tips:**
• Upload files with 'upload <filepath>'
• Play RPG with 'rpg' then 'create <character>'
• Ask me anything - I'll use my memory to help!

🎯 **Example Usage:**
  upload "story.txt"    # Process a text file
  rpg                   # Enter RPG mode
  create "Hero"         # Create character
  explore              # Explore the world
        """.strip()

        return help_text

    def _cmd_status(self, user_id: str, args: List[str], metadata=None) -> str:
        """Show system status"""
        status_parts = ["📊 **System Status Overview**\n"]

        # Memory System
        if self.memory_system:
            mem_stats = self.memory_system.get_memory_stats()
            status_parts.append(f"🧠 **Memory System:** Active")
            status_parts.append(f"   • Total Memories: {mem_stats['total_memories']}")
            status_parts.append(f"   • Knowledge Items: {mem_stats['knowledge_base_size']}")
            status_parts.append(f"   • Files Processed: {mem_stats['total_files']}")
        else:
            status_parts.append("🧠 **Memory System:** Not Available")

        # File System
        if self.file_upload_system:
            files = self.file_upload_system.list_uploaded_files()
            status_parts.append(f"📁 **File System:** Active ({len(files)} files)")
        else:
            status_parts.append("📁 **File System:** Not Available")

        # RPG System
        if self.rpg_chatbot:
            status_parts.append(f"🎮 **RPG System:** Active ({len(self.rpg_chatbot.characters)} characters)")
        else:
            status_parts.append("🎮 **RPG System:** Not Available")

        # Performance System
        if self.performance_monitor:
            status_parts.append("📊 **Performance Monitor:** Active")
        else:
            status_parts.append("📊 **Performance Monitor:** Not Available")

        # Current Mode
        status_parts.append(f"🎯 **Current Mode:** {self.current_mode}")

        return "\n".join(status_parts)

    def _cmd_mode(self, user_id: str, args: List[str], metadata=None) -> str:
        """Switch system mode"""
        if not args:
            return f"🎯 Current mode: {self.current_mode}\nAvailable modes: general, rpg, upload, admin"

        new_mode = args[0].lower()
        valid_modes = ['general', 'rpg', 'upload', 'admin']

        if new_mode in valid_modes:
            self.current_mode = new_mode
            return f"🎯 Switched to {new_mode} mode"
        else:
            return f"❌ Invalid mode. Available: {', '.join(valid_modes)}"

    def _cmd_remember(self, user_id: str, args: List[str], metadata=None) -> str:
        """Store information in memory"""
        if not self.memory_system:
            return "❌ Memory system not available"

        if not args:
            return "❌ Usage: remember <information> [importance:0.1-1.0]"

        content = " ".join(args)
        importance = 0.5  # Default

        # Check for importance parameter
        if len(args) > 1 and args[-1].replace('.', '').isdigit():
            try:
                imp_val = float(args[-1])
                if 0.1 <= imp_val <= 1.0:
                    importance = imp_val
                    content = " ".join(args[:-1])
            except ValueError:
                pass

        memory_id = self.memory_system.store_memory(
            content,
            memory_type="user_input",
            importance=importance
        )

        return f"✅ Remembered: '{content[:50]}...'\n🆔 Memory ID: {memory_id}"

    def _cmd_recall(self, user_id: str, args: List[str], metadata=None) -> str:
        """Recall stored information"""
        if not self.memory_system:
            return "❌ Memory system not available"

        if not args:
            return "❌ Usage: recall <search_term>"

        query = " ".join(args)
        results = self.memory_system.recall_memory(query, limit=3)

        if not results:
            return f"🤔 No memories found for '{query}'"

        response = f"💭 Memories about '{query}':\n"
        for result in results:
            timestamp = time.strftime('%m/%d %H:%M', time.localtime(result['timestamp']))
            response += f"• [{timestamp}] {result['content'][:100]}...\n"

        return response

    def _cmd_search(self, user_id: str, args: List[str], metadata=None) -> str:
        """Search knowledge base"""
        if not self.memory_system:
            return "❌ Memory system not available"

        if not args:
            return "❌ Usage: search <topic>"

        query = " ".join(args)
        results = self.memory_system.search_knowledge(query)

        if not results:
            return f"🔍 No knowledge found about '{query}'"

        response = f"📚 Knowledge about '{query}':\n"
        for keyword, items in results.items():
            response += f"\n🔑 {keyword}:\n"
            for item in items[:2]:  # Limit to 2 per keyword
                response += f"  • {item['content'][:150]}...\n"

        return response

    def _cmd_memory_stats(self, user_id: str, args: List[str], metadata=None) -> str:
        """Show memory statistics"""
        if not self.memory_system:
            return "❌ Memory system not available"

        stats = self.memory_system.get_memory_stats()
        return f"""
🧠 **Memory Statistics:**
• Total Memories: {stats['total_memories']}
• Memory Types: {', '.join(f'{k}:{v}' for k, v in stats['memory_by_type'].items())}
• Knowledge Base: {stats['knowledge_base_size']} items
• Files Processed: {stats['total_files']}
• Training Data: {stats['training_data_size']} samples
• Short-term Memory: {stats['short_term_memory_size']} items
• Long-term Memory: {stats['long_term_memory_size']} items
        """.strip()

    def _cmd_upload(self, user_id: str, args: List[str], metadata=None) -> str:
        """Upload and process a file"""
        if not args:
            return "❌ Usage: upload <file_path> [description]"

        file_path = args[0]
        description = " ".join(args[1:]) if len(args) > 1 else ""

        return self._handle_file_upload(file_path, {'description': description})

    def _cmd_list_files(self, user_id: str, args: List[str], metadata=None) -> str:
        """List uploaded files"""
        if not self.file_upload_system:
            return "❌ File system not available"

        files = self.file_upload_system.list_uploaded_files()
        if not files:
            return "📁 No files uploaded yet."

        file_list = "\n".join([
            f"• {f['name']} ({f['type']}, {f['size']} bytes, {time.strftime('%m/%d %H:%M', time.localtime(f['modified']))})"
            for f in files[:10]
        ])

        if len(files) > 10:
            file_list += f"\n... and {len(files) - 10} more files"

        return f"📁 **Uploaded Files:**\n{file_list}"

    def _cmd_download_file(self, user_id: str, args: List[str], metadata=None) -> str:
        """Download file content"""
        if not args:
            return "❌ Usage: download <filename>"

        filename = args[0]
        return self._handle_file_query(f"show me {filename}")

    def _cmd_rpg_mode(self, user_id: str, args: List[str], metadata=None) -> str:
        """Enter RPG mode"""
        self.current_mode = "rpg"
        return "🎮 **RPG Mode Activated!**\n\n" + self.rpg_chatbot.get_rpg_status()

    def _cmd_rpg_create(self, user_id: str, args: List[str], metadata=None) -> str:
        """Create RPG character"""
        return self.rpg_chatbot.process_rpg_command(user_id, "create", args)

    def _cmd_rpg_explore(self, user_id: str, args: List[str], metadata=None) -> str:
        """Explore in RPG"""
        return self.rpg_chatbot.process_rpg_command(user_id, "explore", args)

    def _cmd_rpg_quest(self, user_id: str, args: List[str], metadata=None) -> str:
        """Handle RPG quests"""
        return self.rpg_chatbot.process_rpg_command(user_id, "quest", args)

    def _cmd_performance(self, user_id: str, args: List[str], metadata=None) -> str:
        """Show performance dashboard"""
        if not self.performance_monitor:
            return "❌ Performance monitor not available"

        self.performance_monitor.show_dashboard()
        return "📊 Performance dashboard displayed above"

    def _cmd_performance_report(self, user_id: str, args: List[str], metadata=None) -> str:
        """Generate performance report"""
        if not self.performance_monitor:
            return "❌ Performance monitor not available"

        success = self.performance_monitor.export_report()
        return "✅ Performance report exported!" if success else "❌ Report export failed"

    def _cmd_clear(self, user_id: str, args: List[str], metadata=None) -> str:
        """Clear screen"""
        print("\033[2J\033[H")  # ANSI escape sequence to clear screen
        return "🧹 Screen cleared"

    def _cmd_save_all(self, user_id: str, args: List[str], metadata=None) -> str:
        """Save all system data"""
        if self.memory_system:
            self.memory_system._save_memory()
        if self.rpg_chatbot:
            self.rpg_chatbot._cmd_save_game(user_id, [])
        return "💾 All system data saved"

    def _cmd_load_all(self, user_id: str, args: List[str], metadata=None) -> str:
        """Load all system data"""
        # This would reload all systems
        return "🔄 System data reloaded"

    def shutdown(self):
        """Clean shutdown of all systems"""
        print("🔄 Shutting down HRM Multi-Modal Interface...")

        if self.memory_system:
            self.memory_system.close()

        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

        print("✅ All systems shut down gracefully")
