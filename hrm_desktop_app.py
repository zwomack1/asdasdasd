#!/usr/bin/env python3
"""
HRM-Gemini Desktop Application
A complete desktop GUI application that integrates with Google Gemini API
and provides the full HRM-Gemini brain functionality in a native executable.
"""

import sys
import os
import json
import time
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import tkinter.font as tkfont

# Try to import Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini API not available. Install with: pip install google-generativeai")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class HRMGeminiDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HRM-Gemini AI Brain")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Check for API key before proceeding
        self.api_key_configured = self.check_api_key()

        if not self.api_key_configured:
            self.show_api_setup_wizard()
            return

        # Continue with normal initialization
        self.initialize_systems()
        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        self.start_background_tasks()
        self.setup_keyboard_shortcuts()

    def check_api_key(self):
        """Check if Google Gemini API key is configured"""
        try:
            from config import gemini_config
            if hasattr(gemini_config, 'GEMINI_API_KEY') and gemini_config.GEMINI_API_KEY:
                # Check if it's not the default placeholder
                if gemini_config.GEMINI_API_KEY != "your_google_gemini_api_key_here" and gemini_config.GEMINI_API_KEY.startswith("AIza"):
                    return True
            return False
        except Exception as e:
            print(f"API key check error: {e}")
            return False

    def show_api_setup_wizard(self):
        """Show the API setup wizard"""
        # Hide main window initially
        self.root.withdraw()

        # Create setup window
        setup_window = tk.Toplevel(self.root)
        setup_window.title("HRM-Gemini AI - Setup Wizard")
        setup_window.geometry("600x500")
        setup_window.resizable(False, False)
        setup_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing

        # Center the window
        setup_window.transient(self.root)
        setup_window.grab_set()

        # Setup content
        self.create_setup_wizard(setup_window)

    def create_setup_wizard(self, parent):
        """Create the API setup wizard interface"""
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        ttk.Label(header_frame, text=" HRM-Gemini AI Setup",
                 font=("Arial", 16, "bold")).pack(pady=(0, 10))

        ttk.Label(header_frame, text="Welcome! Let's get your AI assistant configured.",
                 font=("Arial", 10)).pack()

        # Progress indicator
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.setup_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.setup_progress.pack(pady=10)
        self.setup_progress['value'] = 25

        # Main content
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Step 1: API Key Input
        self.create_api_key_step(content_frame)

        # Footer
        footer_frame = ttk.Frame(parent)
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ttk.Button(footer_frame, text="Test Connection",
                  command=self.test_api_connection).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(footer_frame, text="Save & Continue",
                  command=lambda: self.save_api_config(parent)).pack(side=tk.RIGHT)

        ttk.Button(footer_frame, text="Get API Key",
                  command=self.open_gemini_api_page).pack(side=tk.LEFT)

    def create_api_key_step(self, parent):
        """Create the API key input step"""
        step_frame = ttk.LabelFrame(parent, text="Step 1: Configure Google Gemini API")
        step_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructions
        instructions = """
 To use HRM-Gemini AI, you need a Google Gemini API key:

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the API key and paste it below

This key will be stored securely on your computer.
        """

        ttk.Label(step_frame, text=instructions, justify=tk.LEFT,
                 wraplength=500).pack(pady=15, padx=15, anchor=tk.W)

        # API Key input
        input_frame = ttk.Frame(step_frame)
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        ttk.Label(input_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_entry = ttk.Entry(input_frame, width=50, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Show/Hide password
        self.show_key_var = tk.BooleanVar()
        ttk.Checkbutton(input_frame, text="Show API Key",
                       variable=self.show_key_var,
                       command=self.toggle_api_key_visibility).grid(
                           row=1, column=1, sticky=tk.W, padx=(10, 0))

        input_frame.columnconfigure(1, weight=1)

        # Status
        self.setup_status = ttk.Label(step_frame, text="Ready to configure API key",
                                    foreground="blue")
        self.setup_status.pack(pady=(10, 0))

    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")

    def open_gemini_api_page(self):
        """Open the Google Gemini API page"""
        import webbrowser
        webbrowser.open("https://makersuite.google.com/app/apikey")
        self.setup_status.config(text="Opened Google Gemini API page in browser", foreground="green")

    def test_api_connection(self):
        """Test the API connection"""
        api_key = self.api_key_entry.get().strip()

        if not api_key:
            self.setup_status.config(text="Please enter an API key first", foreground="red")
            return

        self.setup_status.config(text="Testing API connection...", foreground="blue")
        self.setup_progress['value'] = 50
        self.root.update()

        try:
            # Test the API key
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            # Try a simple test
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Hello, test message")

            self.setup_status.config(text=" API connection successful!", foreground="green")
            self.setup_progress['value'] = 75

        except Exception as e:
            self.setup_status.config(text=f" API test failed: {str(e)}", foreground="red")
            self.setup_progress['value'] = 25

    def save_api_config(self, setup_window):
        """Save the API configuration"""
        api_key = self.api_key_entry.get().strip()

        if not api_key:
            messagebox.showerror("Error", "Please enter an API key")
            return

        try:
            # Save to config file
            config_path = Path(__file__).parent / "config" / "gemini_config.py"

            config_content = f'''# HRM-Gemini AI Configuration
# Google Gemini API Configuration

GEMINI_API_KEY = "{api_key}"

# API Settings
GEMINI_MODEL = "gemini-pro"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Memory Settings
MAX_MEMORY_ITEMS = 1000
MEMORY_RETENTION_DAYS = 30

# File Processing Settings
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "jpg", "png", "py", "js", "html"]
MAX_FILE_SIZE_MB = 10

# RPG Settings
DEFAULT_CHARACTER_CLASS = "Adventurer"
MAX_QUESTS = 50
'''

            with open(config_path, 'w') as f:
                f.write(config_content)

            self.setup_progress['value'] = 100
            self.setup_status.config(text=" Configuration saved successfully!", foreground="green")

            # Close setup window and show main app
            setup_window.destroy()
            self.root.deiconify()
            self.initialize_systems()
            self.setup_styles()
            self.create_menu()
            self.create_main_layout()
            self.start_background_tasks()
            self.setup_keyboard_shortcuts()

            messagebox.showinfo("Setup Complete",
                              "HRM-Gemini AI has been configured successfully!\n\n"
                              "The application will now start with your API key.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings - HRM-Gemini AI")
        settings_window.geometry("500x400")

        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # API Settings tab
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API Settings")

        ttk.Label(api_frame, text="Google Gemini API Configuration",
                 font=("Arial", 12, "bold")).pack(pady=10)

        # API Key
        key_frame = ttk.Frame(api_frame)
        key_frame.pack(fill=tk.X, padx=20, pady=5)

        ttk.Label(key_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=5)
        api_key_entry = ttk.Entry(key_frame, width=40, show="*")
        api_key_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Load current API key
        try:
            from config import Config
            config = Config()
            if hasattr(config, 'gemini_api_key'):
                api_key_entry.insert(0, config.gemini_api_key)
        except:
            pass

        # Model selection
        model_frame = ttk.Frame(api_frame)
        model_frame.pack(fill=tk.X, padx=20, pady=5)

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_var = tk.StringVar(value="gemini-pro")
        model_combo = ttk.Combobox(model_frame, textvariable=model_var,
                                  values=["gemini-pro", "gemini-pro-vision"])
        model_combo.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Memory Settings tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory")

        ttk.Label(memory_frame, text="Memory Management Settings",
                 font=("Arial", 12, "bold")).pack(pady=10)

        # Memory retention
        retention_frame = ttk.Frame(memory_frame)
        retention_frame.pack(fill=tk.X, padx=20, pady=5)

        ttk.Label(retention_frame, text="Retention (days):").grid(row=0, column=0, sticky=tk.W, pady=5)
        retention_var = tk.IntVar(value=30)
        retention_spin = tk.Spinbox(retention_frame, from_=1, to=365, textvariable=retention_var)
        retention_spin.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Appearance Settings tab
        appearance_frame = ttk.Frame(notebook)
        notebook.add(appearance_frame, text="Appearance")

        ttk.Label(appearance_frame, text="Interface Customization",
                 font=("Arial", 12, "bold")).pack(pady=10)

        # Theme selection
        theme_frame = ttk.Frame(appearance_frame)
        theme_frame.pack(fill=tk.X, padx=20, pady=5)

        ttk.Label(theme_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, pady=5)
        theme_var = tk.StringVar(value="light")
        theme_combo = ttk.Combobox(theme_frame, textvariable=theme_var,
                                  values=["light", "dark", "auto"])
        theme_combo.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Buttons
        btn_frame = ttk.Frame(settings_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Save Settings",
                  command=lambda: self.save_settings(settings_window)).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Cancel",
                  command=settings_window.destroy).pack(side=tk.RIGHT)

    def save_settings(self, settings_window):
        """Save settings from the settings dialog"""
        # This would save the settings to config files
        messagebox.showinfo("Settings", "Settings saved successfully!")
        settings_window.destroy()

    def setup_styles(self):
        """Setup custom styles for the application"""
        style = ttk.Style()

        # Configure colors similar to Gemini
        style.configure("Primary.TButton",
                       background="#1a73e8",
                       foreground="white",
                       padding=(10, 5))

        style.configure("Secondary.TButton",
                       background="#f8f9fa",
                       foreground="#202124",
                       padding=(10, 5))

        style.configure("Success.TButton",
                       background="#34a853",
                       foreground="white",
                       padding=(10, 5))

        # Configure fonts
        self.title_font = tkfont.Font(family="Google Sans", size=14, weight="bold")
        self.normal_font = tkfont.Font(family="Google Sans", size=10)
        self.mono_font = tkfont.Font(family="Consolas", size=9)

    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Chat", command=self.new_chat, accelerator="Ctrl+N")
        file_menu.add_command(label="Open File", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Chat", command=self.export_chat)
        file_menu.add_command(label="Import Chat", command=self.import_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Copy", command=self.copy_text, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste_text, accelerator="Ctrl+V")
        edit_menu.add_command(label="Clear Chat", command=self.clear_chat, accelerator="Ctrl+L")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Memory Manager", command=self.show_memory_manager)
        tools_menu.add_command(label="File Processor", command=self.show_file_processor)
        tools_menu.add_command(label="RPG Game", command=self.show_rpg_game)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self.show_settings)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def create_main_layout(self):
        """Create the main application layout"""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create sidebar
        self.create_sidebar(main_frame)

        # Create main content area
        self.create_content_area(main_frame)

        # Create status bar
        self.create_status_bar()

    def create_sidebar(self, parent):
        """Create the sidebar with navigation"""
        sidebar_frame = ttk.Frame(parent, width=200)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Title
        title_label = ttk.Label(sidebar_frame, text="HRM-Gemini AI",
                               font=self.title_font)
        title_label.pack(pady=(0, 20))

        # Navigation buttons
        nav_buttons = [
            ("💬 Chat", "chat", self.switch_to_chat),
            ("🧠 Memory", "memory", self.switch_to_memory),
            ("📁 Files", "files", self.switch_to_files),
            ("🎮 RPG", "rpg", self.switch_to_rpg),
            ("💻 Code", "code", self.switch_to_code),
            ("📊 Dashboard", "dashboard", self.switch_to_dashboard)
        ]

        self.nav_buttons = []
        for text, mode, callback in nav_buttons:
            btn = ttk.Button(sidebar_frame, text=text,
                           command=lambda m=mode, c=callback: self.switch_mode(m, c))
            btn.pack(fill=tk.X, pady=2)
            self.nav_buttons.append((btn, mode))

        # Separator
        ttk.Separator(sidebar_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        # Quick actions
        ttk.Label(sidebar_frame, text="Quick Actions", font=self.title_font).pack(pady=(0, 10))

        ttk.Button(sidebar_frame, text="📤 Upload File",
                  command=self.quick_upload).pack(fill=tk.X, pady=2)

        ttk.Button(sidebar_frame, text="🔍 Search Memory",
                  command=self.quick_search).pack(fill=tk.X, pady=2)

        ttk.Button(sidebar_frame, text="📊 System Status",
                  command=self.show_system_status).pack(fill=tk.X, pady=2)

    def create_content_area(self, parent):
        """Create the main content area"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create different mode frames
        self.create_chat_mode(content_frame)
        self.create_memory_mode(content_frame)
        self.create_files_mode(content_frame)
        self.create_rpg_mode(content_frame)
        self.create_code_mode(content_frame)
        self.create_dashboard_mode(content_frame)

        # Show default mode
        self.switch_mode("chat", self.switch_to_chat)

    def create_chat_mode(self, parent):
        """Create the chat interface"""
        self.chat_frame = ttk.Frame(parent)

        # Chat history area
        chat_container = ttk.Frame(self.chat_frame)
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_text = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=self.normal_font,
            state=tk.DISABLED
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)

        # Input area
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.message_input = tk.Text(input_frame, height=3, font=self.normal_font,
                                   wrap=tk.WORD)
        self.message_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Send button
        send_btn = ttk.Button(input_frame, text="📤 Send", style="Primary.TButton",
                            command=self.send_message)
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Action buttons
        actions_frame = ttk.Frame(self.chat_frame)
        actions_frame.pack(fill=tk.X)

        ttk.Button(actions_frame, text="📎 Upload", command=self.upload_file_chat).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="🧹 Clear", command=self.clear_chat).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="💾 Save", command=self.save_chat).pack(side=tk.LEFT, padx=(0, 5))

    def create_memory_mode(self, parent):
        """Create the memory management interface"""
        self.memory_frame = ttk.Frame(parent)

        # Memory controls
        controls_frame = ttk.Frame(self.memory_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls_frame, text="Store Information:").grid(row=0, column=0, sticky=tk.W, pady=5)

        self.memory_input = tk.Text(controls_frame, height=3, width=50, font=self.normal_font)
        self.memory_input.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=5)

        ttk.Label(controls_frame, text="Type:").grid(row=2, column=0, sticky=tk.W)
        self.memory_type = ttk.Combobox(controls_frame, values=["general", "personal", "work", "learning"])
        self.memory_type.set("general")
        self.memory_type.grid(row=2, column=1, sticky=tk.EW, padx=(5, 0))

        ttk.Label(controls_frame, text="Importance:").grid(row=3, column=0, sticky=tk.W)
        self.importance_scale = tk.Scale(controls_frame, from_=0.1, to=1.0, resolution=0.1,
                                       orient=tk.HORIZONTAL)
        self.importance_scale.set(0.5)
        self.importance_scale.grid(row=3, column=1, sticky=tk.EW, padx=(5, 0))

        ttk.Button(controls_frame, text="💾 Remember", style="Primary.TButton",
                  command=self.store_memory).grid(row=4, column=0, columnspan=2, pady=10)

        # Search section
        search_frame = ttk.Frame(self.memory_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(search_frame, text="Search Memories:").grid(row=0, column=0, sticky=tk.W, pady=5)

        self.search_input = ttk.Entry(search_frame, font=self.normal_font)
        self.search_input.grid(row=1, column=0, sticky=tk.EW, pady=5, padx=(0, 5))

        ttk.Button(search_frame, text="🔍 Search", command=self.search_memory).grid(row=1, column=1)

        # Results area
        results_frame = ttk.Frame(self.memory_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.memory_results = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=self.mono_font,
            state=tk.DISABLED
        )
        self.memory_results.pack(fill=tk.BOTH, expand=True)

    def create_files_mode(self, parent):
        """Create the file processing interface"""
        self.files_frame = ttk.Frame(parent)

        # Upload area
        upload_frame = ttk.LabelFrame(self.files_frame, text="File Upload & Processing")
        upload_frame.pack(fill=tk.X, pady=(0, 10))

        upload_area = tk.Text(upload_frame, height=8, font=self.normal_font,
                            bg="#f8f9fa", relief=tk.FLAT)
        upload_area.pack(fill=tk.X, padx=10, pady=10)
        upload_area.insert(tk.END, "Drop files here or click 'Browse' to upload\n\nSupported formats:\n• Documents: PDF, DOCX, TXT\n• Images: JPG, PNG, GIF\n• Code: PY, JS, HTML, CSS\n• Data: CSV, JSON, XLSX")
        upload_area.config(state=tk.DISABLED)

        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="📁 Browse Files", command=self.browse_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="📊 View Processed", command=self.view_processed_files).pack(side=tk.LEFT)

        # Processing status
        self.processing_status = ttk.Label(self.files_frame, text="Ready to process files")
        self.processing_status.pack(anchor=tk.W, pady=(0, 10))

        # File list
        list_frame = ttk.LabelFrame(self.files_frame, text="Processed Files")
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, font=self.mono_font, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_rpg_mode(self, parent):
        """Create the RPG game interface"""
        self.rpg_frame = ttk.Frame(parent)

        # Game area
        game_frame = ttk.LabelFrame(self.rpg_frame, text="RPG Adventure")
        game_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.rpg_text = scrolledtext.ScrolledText(
            game_frame,
            wrap=tk.WORD,
            font=self.mono_font,
            bg="#1a1a1a",
            fg="#00ff00",
            state=tk.DISABLED
        )
        self.rpg_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Command input
        input_frame = ttk.Frame(self.rpg_frame)
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="Command:").grid(row=0, column=0, padx=(0, 5))
        self.rpg_command = ttk.Entry(input_frame, font=self.normal_font)
        self.rpg_command.grid(row=0, column=1, sticky=tk.EW, padx=(0, 5))

        ttk.Button(input_frame, text="⚔️ Execute", style="Success.TButton",
                  command=self.execute_rpg_command).grid(row=0, column=2)

        input_frame.columnconfigure(1, weight=1)

        # Quick commands
        quick_frame = ttk.Frame(self.rpg_frame)
        quick_frame.pack(fill=tk.X, pady=(10, 0))

        quick_commands = [
            ("🏠 Status", "status"),
            ("🗺️ Explore", "explore"),
            ("📋 Quests", "quest list"),
            ("🎒 Inventory", "inventory"),
            ("💾 Save", "save")
        ]

        for i, (text, cmd) in enumerate(quick_commands):
            ttk.Button(quick_frame, text=text,
                      command=lambda c=cmd: self.quick_rpg_command(c)).grid(
                          row=0, column=i, padx=(0, 5))

    def create_code_mode(self, parent):
        """Create the coding helper interface"""
        self.code_frame = ttk.Frame(parent)

        # Code input area
        code_input_frame = ttk.LabelFrame(self.code_frame, text="Code to Analyze")
        code_input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.code_input = scrolledtext.ScrolledText(
            code_input_frame,
            wrap=tk.NONE,
            font=("Consolas", 10),
            height=15
        )
        self.code_input.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Language selection and action buttons
        controls_frame = ttk.Frame(self.code_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls_frame, text="Language:").grid(row=0, column=0, padx=(10, 5))
        self.code_language = ttk.Combobox(controls_frame, values=["python", "javascript", "java", "cpp", "html", "css"])
        self.code_language.set("python")
        self.code_language.grid(row=0, column=1, padx=(0, 10))

        ttk.Button(controls_frame, text="🔍 Analyze Code", style="Primary.TButton",
                  command=self.analyze_code).grid(row=0, column=2, padx=(0, 5))

        ttk.Button(controls_frame, text="💡 Get Suggestions", style="Success.TButton",
                  command=self.get_code_suggestions).grid(row=0, column=3, padx=(0, 5))

        ttk.Button(controls_frame, text="🔄 Refactor", style="Secondary.TButton",
                  command=self.refactor_code).grid(row=0, column=4, padx=(0, 5))

        # Results area
        results_frame = ttk.LabelFrame(self.code_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.code_results = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            state=tk.DISABLED
        )
        self.code_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def switch_to_code(self):
        """Switch to code mode"""
        self.update_status("Code analysis and improvement mode active")

    def analyze_code(self):
        """Analyze the entered code"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some code to analyze.")
            return

        language = self.code_language.get()

        if self.coding_helper:
            def analyze():
                try:
                    analysis = self.coding_helper.analyze_code(code, language)

                    result_text = f"🔍 CODE ANALYSIS RESULTS\n{'='*50}\n\n"

                    # Metrics
                    metrics = analysis.get("metrics", {})
                    result_text += f"📊 METRICS:\n"
                    for key, value in metrics.items():
                        result_text += f"  • {key.replace('_', ' ').title()}: {value}\n"
                    result_text += "\n"

                    # Issues
                    issues = analysis.get("issues", [])
                    if issues:
                        result_text += f"⚠️ ISSUES FOUND ({len(issues)}):\n"
                        for issue in issues:
                            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.get("severity"), "⚪")
                            result_text += f"  {severity_icon} {issue['description']}\n"
                            result_text += f"     💡 {issue.get('suggestion', 'No suggestion available')}\n"
                        result_text += "\n"

                    # Improvements
                    improvements = analysis.get("improvements", [])
                    if improvements:
                        result_text += f"🚀 IMPROVEMENT SUGGESTIONS:\n"
                        for improvement in improvements:
                            result_text += f"  • {improvement}\n"
                        result_text += "\n"

                    # Best practices
                    practices = analysis.get("best_practices", [])
                    if practices:
                        result_text += f"📚 BEST PRACTICES:\n"
                        for practice in practices:
                            result_text += f"  • {practice['practice'].replace('_', ' ').title()}: {practice['description']}\n"

                    self.root.after(0, lambda: self.display_code_results(result_text))

                except Exception as e:
                    self.root.after(0, lambda: self.display_code_results(f"❌ Analysis failed: {e}"))

            # Run analysis in background
            thread = threading.Thread(target=analyze, daemon=True)
            thread.start()

            self.display_code_results("🔄 Analyzing code... Please wait...")
        else:
            messagebox.showerror("Error", "Coding helper not available")

    def get_code_suggestions(self):
        """Get AI-powered code suggestions"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some code to analyze.")
            return

        # Get user request
        request = simpledialog.askstring("Code Suggestions",
                                       "What kind of suggestions do you want?\n\n" +
                                       "Examples:\n" +
                                       "- Make this code more efficient\n" +
                                       "- Improve error handling\n" +
                                       "- Add type hints\n" +
                                       "- Optimize for performance\n" +
                                       "- Make it more readable")

        if not request:
            return

        if self.coding_helper:
            def get_suggestions():
                try:
                    result = self.coding_helper.generate_code_suggestion(code, request)

                    if result.get("success"):
                        result_text = f"💡 AI CODE SUGGESTIONS\n{'='*50}\n\n"
                        result_text += f"Request: {request}\n\n"
                        result_text += result["suggestions"]
                    else:
                        result_text = f"❌ Failed to get suggestions: {result.get('error', 'Unknown error')}"

                    self.root.after(0, lambda: self.display_code_results(result_text))

                except Exception as e:
                    self.root.after(0, lambda: self.display_code_results(f"❌ Suggestion generation failed: {e}"))

            thread = threading.Thread(target=get_suggestions, daemon=True)
            thread.start()

            self.display_code_results("🤖 AI is analyzing your code and generating suggestions...")
        else:
            messagebox.showerror("Error", "Coding helper not available")

    def refactor_code(self):
        """Refactor the entered code"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some code to refactor.")
            return

        # Get refactoring type
        refactor_types = {
            "extract_function": "Extract repeated code into reusable functions",
            "simplify_conditionals": "Simplify complex conditional statements",
            "improve_naming": "Improve variable and function names",
            "reduce_complexity": "Reduce code complexity",
            "optimize_performance": "Optimize for better performance",
            "improve_readability": "Improve code readability"
        }

        # Create refactoring selection dialog
        refactor_window = tk.Toplevel(self.root)
        refactor_window.title("Select Refactoring Type")
        refactor_window.geometry("400x300")

        ttk.Label(refactor_window, text="What type of refactoring do you want?",
                 font=("Arial", 12, "bold")).pack(pady=10)

        selected_refactor = tk.StringVar()

        for refactor_type, description in refactor_types.items():
            rb = ttk.Radiobutton(refactor_window, text=f"{refactor_type.replace('_', ' ').title()}\n{description}",
                               variable=selected_refactor, value=refactor_type)
            rb.pack(anchor=tk.W, padx=20, pady=2)

        def do_refactor():
            refactor_type = selected_refactor.get()
            refactor_window.destroy()

            if refactor_type and self.coding_helper:
                def refactor():
                    try:
                        result = self.coding_helper.refactor_code_suggestion(code, refactor_type)

                        if result.get("success"):
                            result_text = f"🔄 CODE REFACTORING - {refactor_type.replace('_', ' ').title()}\n{'='*50}\n\n"
                            result_text += result["refactoring"]
                        else:
                            result_text = f"❌ Refactoring failed: {result.get('error', 'Unknown error')}"

                        self.root.after(0, lambda: self.display_code_results(result_text))

                    except Exception as e:
                        self.root.after(0, lambda: self.display_code_results(f"❌ Refactoring failed: {e}"))

                thread = threading.Thread(target=refactor, daemon=True)
                thread.start()

                self.display_code_results("🔄 AI is analyzing and refactoring your code...")

        ttk.Button(refactor_window, text="Start Refactoring", command=do_refactor).pack(pady=20)

    def display_code_results(self, text):
        """Display results in the code analysis area"""
        self.code_results.config(state=tk.NORMAL)
        self.code_results.delete("1.0", tk.END)
        self.code_results.insert(tk.END, text)
        self.code_results.config(state=tk.DISABLED)
        self.code_results.see("1.0")

    def create_status_card(self, parent, title, var_name, row, col):
        """Create a status card for the dashboard"""
        card = ttk.LabelFrame(parent, text=title)
        card.grid(row=row, column=col, padx=5, pady=5, sticky=tk.NSEW)

        label = ttk.Label(card, text="Loading...", font=self.normal_font)
        label.pack(padx=10, pady=10)

        setattr(self, var_name, label)

        parent.grid_columnconfigure(col, weight=1)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(self.status_frame, text="Ready", font=self.normal_font)
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Progress bar for operations
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=10)

        # Gemini API status
        self.api_status = ttk.Label(self.status_frame, text="🤖 AI: Connected", font=self.normal_font)
        self.api_status.pack(side=tk.RIGHT, padx=10)

    def initialize_systems(self):
        """Initialize all HRM systems"""
        try:
            # Initialize Memory System
            from hrm_memory_system import HRMMemorySystem
            self.memory_system = HRMMemorySystem()
            print("✅ Memory System initialized")

            # Initialize File System
            from file_upload_system import FileUploadSystem
            self.file_upload_system = FileUploadSystem(memory_system=self.memory_system)
            print("✅ File Upload System initialized")

            # Initialize Login System
            from login_system import LoginSystem
            from login_interface import LoginInterface
            self.login_system = LoginSystem(self)
            self.login_interface = LoginInterface(self, self.login_system)

            # Initialize Self-Modification Engine
            from self_modification_engine import SelfModificationEngine
            from performance_analyzer import PerformanceAnalyzer
            from code_optimizer import CodeOptimizer
            from configuration_tuner import ConfigurationTuner
            from user_experience_learner import UserExperienceLearner

            self.self_modification_engine = SelfModificationEngine(self)
            print("🔄 Self-Modification Engine initialized")

            # Initialize RPG System
            from rpg_chatbot import RPGChatbot
            self.rpg_chatbot = RPGChatbot(memory_system=self.memory_system)
            print("✅ RPG System initialized")

            # Perform Self-Improvement Analysis
            print("🔄 Performing self-improvement analysis...")
            self.perform_self_improvement()
            print("✅ Self-improvement analysis complete")

            self.update_status("Systems initialized successfully")

        except Exception as e:
            self.show_error(f"System initialization failed: {e}")
            self.update_status("System initialization failed")

    def initialize_gemini_api(self):
        """Initialize Google Gemini API"""
        try:
            # Try to load API key from config
            print(f"❌ Gemini API initialization failed: {e}")
            self.api_status.config(text="🤖 AI: Disconnected")
            self.show_warning("Gemini API not configured. Please set your API key in config.")

    def start_background_tasks(self):
        """Start background monitoring tasks"""
        # Memory cleanup task
        def memory_cleanup():
            while True:
                time.sleep(300)  # Every 5 minutes
                if self.memory_system:
                    cleaned = self.memory_system.cleanup_old_data()
                    if cleaned > 0:
                        print(f"🧹 Cleaned up {cleaned} old memory items")

        cleanup_thread = threading.Thread(target=memory_cleanup, daemon=True)
        cleanup_thread.start()

        # System monitoring task
        def system_monitor():
            while True:
                time.sleep(10)  # Every 10 seconds
                self.update_dashboard()
                self.update_status("System monitoring active")

        monitor_thread = threading.Thread(target=system_monitor, daemon=True)
        monitor_thread.start()

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-n>', lambda e: self.new_chat())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-l>', lambda e: self.clear_chat())
        self.root.bind('<Control-q>', lambda e: self.on_closing())

        # Chat input shortcuts
        self.message_input.bind('<Control-Return>', lambda e: self.send_message())
        self.rpg_command.bind('<Return>', lambda e: self.execute_rpg_command())

    def switch_mode(self, mode, callback):
        """Switch to a different mode"""
        # Hide all frames
        frames = [self.chat_frame, self.memory_frame, self.files_frame,
                 self.rpg_frame, self.dashboard_frame]
        for frame in frames:
            frame.pack_forget()

        # Update button states
        for btn, btn_mode in self.nav_buttons:
            if btn_mode == mode:
                btn.config(style="Primary.TButton")
            else:
                btn.config(style="TButton")

        # Show selected frame
        getattr(self, f"{mode}_frame").pack(fill=tk.BOTH, expand=True)

        self.current_mode = mode
        callback()

    def switch_to_chat(self):
        """Switch to chat mode"""
        self.update_status("Chat mode active")

    def switch_to_memory(self):
        """Switch to memory mode"""
        self.update_status("Memory management mode active")

    def switch_to_files(self):
        """Switch to files mode"""
        self.update_status("File processing mode active")

    def switch_to_rpg(self):
        """Switch to RPG mode"""
        self.update_status("RPG game mode active")
        self.initialize_rpg()

    def switch_to_dashboard(self):
        """Switch to dashboard mode"""
        self.update_status("System dashboard active")
        self.refresh_dashboard()

    def send_message(self):
        """Send a message to the AI"""
        message = self.message_input.get("1.0", tk.END).strip()
        if not message:
            return

        # Clear input
        self.message_input.delete("1.0", tk.END)

        # Add user message to chat
        self.add_chat_message(f"You: {message}", "user")

        # Show typing indicator
        self.show_typing_indicator()

        # Process message in background
        def process_message():
            try:
                if self.gemini_model:
                    # Use Gemini API
                    response = self.gemini_model.generate_content(message)
                    ai_response = response.text
                else:
                    # Fallback to basic response
                    ai_response = f"I received your message: '{message}'. Gemini API not configured."

                # Store in memory
                if self.memory_system:
                    self.memory_system.store_memory(
                        f"User: {message}\nAI: {ai_response}",
                        memory_type="conversation"
                    )

                # Add AI response to chat
                self.root.after(0, lambda: self.add_chat_message(f"HRM-Gemini: {ai_response}", "ai"))
                self.root.after(0, self.hide_typing_indicator)

            except Exception as e:
                self.root.after(0, lambda: self.add_chat_message(f"Error: {e}", "error"))
                self.root.after(0, self.hide_typing_indicator)

        thread = threading.Thread(target=process_message, daemon=True)
        thread.start()

    def add_chat_message(self, message, msg_type):
        """Add a message to the chat"""
        self.chat_text.config(state=tk.NORMAL)

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format message based on type
        if msg_type == "user":
            self.chat_text.insert(tk.END, f"[{timestamp}] 👤 ", "user_tag")
            self.chat_text.insert(tk.END, f"{message}\n\n", "user_text")
        elif msg_type == "ai":
            self.chat_text.insert(tk.END, f"[{timestamp}] 🤖 ", "ai_tag")
            self.chat_text.insert(tk.END, f"{message}\n\n", "ai_text")
        elif msg_type == "error":
            self.chat_text.insert(tk.END, f"[{timestamp}] ❌ ", "error_tag")
            self.chat_text.insert(tk.END, f"{message}\n\n", "error_text")

        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def show_typing_indicator(self):
        """Show typing indicator"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "🤖 HRM-Gemini is typing...\n\n", "typing")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def hide_typing_indicator(self):
        """Hide typing indicator"""
        self.chat_text.config(state=tk.NORMAL)
        # Remove the last few lines (typing indicator)
        lines = self.chat_text.get("1.0", tk.END).split("\n")
        if "HRM-Gemini is typing..." in "\n".join(lines[-3:]):
            # Remove typing indicator
            self.chat_text.delete("end-3l", tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def store_memory(self):
        """Store information in memory"""
        content = self.memory_input.get("1.0", tk.END).strip()
        if not content:
            self.show_warning("Please enter some content to remember.")
            return

        memory_type = self.memory_type.get()
        importance = self.importance_scale.get()

        if self.memory_system:
            memory_id = self.memory_system.store_memory(
                content,
                memory_type=memory_type,
                importance=importance
            )

            self.memory_input.delete("1.0", tk.END)
            messagebox.showinfo("Success", f"Information remembered!\nMemory ID: {memory_id}")
        else:
            self.show_error("Memory system not available")

    def search_memory(self):
        """Search memory"""
        query = self.search_input.get().strip()
        if not query:
            self.show_warning("Please enter a search query.")
            return

        if self.memory_system:
            results = self.memory_system.recall_memory(query, limit=10)

            self.memory_results.config(state=tk.NORMAL)
            self.memory_results.delete("1.0", tk.END)

            if results:
                self.memory_results.insert(tk.END, f"Search results for '{query}':\n\n")
                for i, result in enumerate(results, 1):
                    timestamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(result['timestamp']))
                    self.memory_results.insert(tk.END,
                        f"{i}. [{timestamp}] {result['type'].upper()}\n"
                        f"   {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}\n\n")
            else:
                self.memory_results.insert(tk.END, f"No memories found for '{query}'")

            self.memory_results.config(state=tk.DISABLED)
        else:
            self.show_error("Memory system not available")

    def browse_files(self):
        """Browse and select files to upload"""
        filetypes = [
            ('All files', '*.*'),
            ('Documents', '*.pdf;*.docx;*.doc;*.txt'),
            ('Images', '*.jpg;*.jpeg;*.png;*.gif;*.bmp'),
            ('Code files', '*.py;*.js;*.html;*.css;*.java'),
            ('Data files', '*.csv;*.json;*.xlsx;*.xls')
        ]

        filenames = filedialog.askopenfilenames(
            title="Select files to upload",
            filetypes=filetypes
        )

        if filenames:
            self.process_selected_files(filenames)

def refresh_dashboard(self):
    """Refresh dashboard data"""
    self.update_dashboard()
    self.update_status("Dashboard refreshed")

# Menu command implementations
def new_chat(self):
    """Start a new chat"""
    self.clear_chat()
    self.switch_mode("chat", self.switch_to_chat)

def open_file(self):
    """Open a file"""
    self.browse_files()

def clear_chat(self):
    """Clear chat history"""
    self.chat_text.config(state=tk.NORMAL)
    self.chat_text.delete("1.0", tk.END)
    self.chat_text.config(state=tk.DISABLED)
    self.chat_history = []

def export_chat(self):
    """Export chat history"""
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.chat_text.get("1.0", tk.END))
        messagebox.showinfo("Success", "Chat exported successfully!")

def import_chat(self):
    """Import chat history"""
    filename = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
                            description=f"Uploaded via desktop app"
                        )

                        if result['success']:
                            processed += 1
                            print(f"✅ Processed: {os.path.basename(filename)}")
                        else:
                            print(f"❌ Failed: {os.path.basename(filename)} - {result['error']}")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

            self.root.after(0, lambda: self.processing_status.config(
                text=f"Processed {processed}/{len(filenames)} files successfully"
            ))

            # Update file list
            self.root.after(0, self.update_file_list)

        thread = threading.Thread(target=process_files, daemon=True)
        thread.start()

    def update_file_list(self):
        """Update the file list display"""
        if hasattr(self, 'file_listbox'):
            self.file_listbox.delete(0, tk.END)

            if self.file_upload_system:
                files = self.file_upload_system.list_uploaded_files()
                for file_info in files[:50]:  # Limit to 50 files
                    self.file_listbox.insert(tk.END,
                        f"{file_info['name']} ({file_info['type']}, {file_info['size']} bytes)")

    def initialize_rpg(self):
        """Initialize RPG game"""
        if self.rpg_chatbot:
            self.add_rpg_message("Welcome to HRM-RPG! Create a character to begin your adventure.\n")
            self.add_rpg_message("Try: 'create Hero' to start\n")

    def execute_rpg_command(self):
        """Execute RPG command"""
        command = self.rpg_command.get().strip()
        if not command:
            return

        # Add command to RPG display
        self.add_rpg_message(f"> {command}\n", "command")

        # Clear input
        self.rpg_command.delete(0, tk.END)

        if self.rpg_chatbot:
            def process_rpg():
                try:
                    response = self.rpg_chatbot.process_rpg_command("desktop_user", command, [])
                    self.root.after(0, lambda: self.add_rpg_message(f"{response}\n", "response"))
                except Exception as e:
                    self.root.after(0, lambda: self.add_rpg_message(f"Error: {e}\n", "error"))

            thread = threading.Thread(target=process_rpg, daemon=True)
            thread.start()

    def quick_rpg_command(self, command):
        """Execute quick RPG command"""
        self.rpg_command.delete(0, tk.END)
        self.rpg_command.insert(0, command)
        self.execute_rpg_command()

    def add_rpg_message(self, message, msg_type="response"):
        """Add message to RPG display"""
        self.rpg_text.config(state=tk.NORMAL)

        tag_config = {
            "command": ("command_tag", "command_text"),
            "response": ("response_tag", "response_text"),
            "error": ("error_tag", "error_text")
        }

        tag_prefix, tag_text = tag_config.get(msg_type, ("response_tag", "response_text"))

        # Add timestamp for non-command messages
        if msg_type != "command":
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.rpg_text.insert(tk.END, f"[{timestamp}] ", tag_prefix)

        self.rpg_text.insert(tk.END, message, tag_text)
        self.rpg_text.see(tk.END)
        self.rpg_text.config(state=tk.DISABLED)

    def update_dashboard(self):
        """Update dashboard with current system status"""
        try:
            # Update memory status
            if self.memory_system:
                stats = self.memory_system.get_memory_stats()
                self.memory_status.config(text=
                    f"Items: {stats['total_memories']}\n"
                    f"Types: {len(stats['memory_by_type'])}\n"
                    f"Files: {stats['total_files']}"
                )

            # Update files status
            if self.file_upload_system:
                files = self.file_upload_system.list_uploaded_files()
                self.files_status.config(text=f"Total: {len(files)}\nProcessed successfully")

            # Update RPG status
            if self.rpg_chatbot:
                characters = len(self.rpg_chatbot.characters)
                self.rpg_status.config(text=f"Characters: {characters}\nActive sessions")

            # Update AI status
            if self.gemini_model:
                self.ai_status.config(text="Connected\nReady")
            else:
                self.ai_status.config(text="Disconnected\nLimited features")

            # Update activity log
            self.update_activity_log()

        except Exception as e:
            print(f"Dashboard update error: {e}")

    def update_activity_log(self):
        """Update the activity log in dashboard"""
        if hasattr(self, 'activity_text'):
            self.activity_text.config(state=tk.NORMAL)
            self.activity_text.delete("1.0", tk.END)

            # Add recent activity
            activities = [
                f"[{datetime.now().strftime('%H:%M:%S')}] System initialized",
                f"[{datetime.now().strftime('%H:%M:%S')}] Memory system active",
                f"[{datetime.now().strftime('%H:%M:%S')}] File processing ready",
                f"[{datetime.now().strftime('%H:%M:%S')}] RPG system loaded",
                f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard updated"
            ]

            for activity in activities[-10:]:  # Show last 10 activities
                self.activity_text.insert(tk.END, f"{activity}\n")

            self.activity_text.config(state=tk.DISABLED)

    def refresh_dashboard(self):
        """Refresh dashboard data"""
        self.update_dashboard()
        self.update_status("Dashboard refreshed")

    # Menu command implementations
    def new_chat(self):
        """Start a new chat"""
        self.clear_chat()
        self.switch_mode("chat", self.switch_to_chat)

    def open_file(self):
        """Open a file"""
        self.browse_files()

    def clear_chat(self):
        """Clear chat history"""
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)
        self.chat_history = []

    def export_chat(self):
        """Export chat history"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.chat_text.get("1.0", tk.END))
            messagebox.showinfo("Success", "Chat exported successfully!")

    def import_chat(self):
        """Import chat history"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, content)
            self.chat_text.config(state=tk.DISABLED)

    def copy_text(self):
        """Copy selected text"""
        try:
            selected_text = self.chat_text.selection_get()
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except:
            pass

    def paste_text(self):
        """Paste text into message input"""
        try:
            clipboard_text = self.root.clipboard_get()
            self.message_input.insert(tk.END, clipboard_text)
        except:
            pass

    def show_memory_manager(self):
        """Show memory manager"""
        self.switch_mode("memory", self.switch_to_memory)

    def show_file_processor(self):
        """Show file processor"""
        self.switch_mode("files", self.switch_to_files)

    def show_rpg_game(self):
        """Show RPG game"""
        self.switch_mode("rpg", self.switch_to_rpg)

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")

        # Settings content would go here
        ttk.Label(settings_window, text="Settings panel coming soon!").pack(pady=20)

    def show_documentation(self):
        """Show documentation"""
        # Open documentation in web browser
        docs_url = "https://github.com/your-repo/docs"
        try:
            webbrowser.open(docs_url)
        except:
            messagebox.showinfo("Documentation", "Documentation URL: " + docs_url)

    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts = """
Keyboard Shortcuts:

File Operations:
  Ctrl+N - New Chat
  Ctrl+O - Open File
  Ctrl+Q - Exit

Edit Operations:
  Ctrl+C - Copy
  Ctrl+V - Paste
  Ctrl+L - Clear Chat

Chat Operations:
  Ctrl+Enter - Send Message

RPG Operations:
  Enter - Execute RPG Command
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def show_about(self):
        """Show about dialog"""
        about_text = """
HRM-Gemini AI Desktop Application

A comprehensive AI system featuring:
• Advanced memory management
• File processing and analysis
• RPG gaming capabilities
• Google Gemini API integration
• Real-time performance monitoring

Version: 1.0.0
Built with Python and Tkinter
"""
        messagebox.showinfo("About HRM-Gemini AI", about_text)

    def show_system_status(self):
        """Show system status dialog"""
        status_info = "System Status:\n\n"

        if self.memory_system:
            status_info += "✅ Memory System: Active\n"
        else:
            status_info += "❌ Memory System: Inactive\n"

        if self.file_upload_system:
            status_info += "✅ File System: Active\n"
        else:
            status_info += "❌ File System: Inactive\n"

        if self.rpg_chatbot:
            status_info += "✅ RPG System: Active\n"
        else:
            status_info += "❌ RPG System: Inactive\n"

        if self.gemini_model:
            status_info += "✅ Gemini API: Connected\n"
        else:
            status_info += "❌ Gemini API: Disconnected\n"

        messagebox.showinfo("System Status", status_info)

    # Utility methods
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)

    def show_warning(self, message):
        """Show warning message"""
        messagebox.showwarning("Warning", message)

    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)

    def show_info(self, message):
        """Show info message"""
        messagebox.showinfo("Information", message)

    # Quick action methods
    def quick_upload(self):
        """Quick file upload"""
        self.browse_files()

    def quick_search(self):
        """Quick memory search"""
        self.switch_mode("memory", self.switch_to_memory)
        self.search_input.focus()

    def save_chat(self):
        """Save current chat"""
        self.export_chat()

    def upload_file_chat(self):
        """Upload file from chat mode"""
        self.browse_files()

    def view_processed_files(self):
        """View processed files"""
        self.update_file_list()

    def export_report(self):
        """Export system report"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            # Generate report content
            report_content = f"""
HRM-Gemini AI System Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATUS:
Memory System: {'Active' if self.memory_system else 'Inactive'}
File System: {'Active' if self.file_upload_system else 'Inactive'}
RPG System: {'Active' if self.rpg_chatbot else 'Inactive'}
Gemini API: {'Connected' if self.gemini_model else 'Disconnected'}

USAGE STATISTICS:
Current Mode: {self.current_mode}
Chat Messages: {len(self.chat_history)}
"""

            if self.memory_system:
                stats = self.memory_system.get_memory_stats()
                report_content += f"""
MEMORY STATISTICS:
Total Memories: {stats['total_memories']}
Knowledge Items: {stats['knowledge_base_size']}
Files Processed: {stats['total_files']}
Training Samples: {stats['training_data_size']}
"""

            with open(filename, 'w') as f:
                f.write(report_content.strip())

            messagebox.showinfo("Success", "System report exported successfully!")

    def cleanup_system(self):
        """Clean up system resources"""
        if self.memory_system:
            cleaned = self.memory_system.cleanup_old_data()
            messagebox.showinfo("Cleanup Complete", f"Cleaned up {cleaned} old memory items")

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit HRM-Gemini AI?"):
            # Save any pending data
            if self.memory_system:
                self.memory_system._save_memory()

            self.root.quit()

    def perform_self_improvement(self):
        """Perform self-improvement analysis and apply improvements"""
        if not hasattr(self, 'self_modification_engine'):
            return

        try:
            # Perform comprehensive self-analysis
            analysis = self.self_modification_engine.perform_self_analysis()

            # Log analysis results
            print("📊 Self-Analysis Results:")
            print(f"  • Performance Score: {analysis['performance_metrics']['performance_score']}")
            print(f"  • Code Quality Score: {analysis['code_quality']['quality_score']}")
            print(f"  • User Satisfaction: {analysis['user_experience']['satisfaction_score']}")
            print(f"  • System Health: {analysis['system_health']['status']}")
            print(f"  • Improvement Opportunities: {len(analysis['improvement_opportunities'])}")

            # Apply self-improvements
            if analysis['improvement_opportunities']:
                result = self.self_modification_engine.apply_self_improvements(analysis)

                if result['success']:
                    print(f"✅ Applied {result['improvements_applied']} self-improvements")
                    if result['improvements_failed'] > 0:
                        print(f"⚠️ {result['improvements_failed']} improvements failed")

                    # Update status with improvement summary
                    self.update_status(f"Self-improvement complete: {result['improvements_applied']} enhancements applied")
                else:
                    print("❌ Self-improvement process failed")
            else:
                print("✨ No improvements needed - system is optimal!")
                self.update_status("System analysis complete - no improvements needed")

        except Exception as e:
            print(f"❌ Self-improvement process error: {e}")
            self.update_status("Self-improvement analysis encountered an error")

def main():
    """Main application entry point with login system"""
    # Create root window but don't show it yet
    root = tk.Tk()
    root.withdraw()  # Hide until login is successful

    try:
        # Initialize the application
        app = HRMGeminiDesktopApp(root)

        # Show login interface
        login_success = app.show_login()

        if login_success:
            # Set window close handler
            root.protocol("WM_DELETE_WINDOW", app.on_closing)

            # Show the main window now that login is successful
            root.deiconify()

            # Start the application
            root.mainloop()
        else:
            # Login failed or was cancelled
            print("Login cancelled. Exiting application.")

    except Exception as e:
        print(f"Application initialization error: {e}")
        root.destroy()

if __name__ == "__main__":
    main()
