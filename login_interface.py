#!/usr/bin/env python3
"""
HRM-Gemini AI Login Interface
Automated login GUI with user management
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
from pathlib import Path
import threading
import time

class LoginInterface:
    """Login interface for HRM-Gemini AI"""

    def __init__(self, main_app=None, login_system=None):
        self.main_app = main_app
        self.login_system = login_system or LoginSystem()
        self.root = None
        self.login_successful = False

    def show_login_window(self):
        """Show the login window"""
        self.root = tk.Toplevel()
        self.root.title("HRM-Gemini AI - Login")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        # Center the window
        self.root.transient()
        self.root.grab_set()

        # Make it modal
        self.root.focus_set()

        self._create_login_ui()
        self._check_auto_login()

        # Wait for login to complete
        self.root.wait_window()

        return self.login_successful

    def _create_login_ui(self):
        """Create the login user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="🤖 HRM-Gemini AI",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(main_frame, text="Intelligent AI Companion",
                                  font=("Arial", 10))
        subtitle_label.pack(pady=(0, 30))

        # Login form frame
        form_frame = ttk.LabelFrame(main_frame, text="Login", padding="15")
        form_frame.pack(fill=tk.X, pady=(0, 20))

        # Username
        ttk.Label(form_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.username_var = tk.StringVar()
        self.username_entry = ttk.Entry(form_frame, textvariable=self.username_var, width=30)
        self.username_entry.grid(row=0, column=1, sticky=tk.EW, padx=(10, 0), pady=5)
        self.username_entry.focus()

        # Password
        ttk.Label(form_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(form_frame, textvariable=self.password_var,
                                       show="*", width=30)
        self.password_entry.grid(row=1, column=1, sticky=tk.EW, padx=(10, 0), pady=5)

        # Remember me checkbox
        self.remember_var = tk.BooleanVar(value=True)
        remember_check = ttk.Checkbutton(form_frame, text="Remember me",
                                       variable=self.remember_var)
        remember_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Auto login checkbox
        self.auto_login_var = tk.BooleanVar(value=False)
        auto_login_check = ttk.Checkbutton(form_frame, text="Auto-login on startup",
                                         variable=self.auto_login_var)
        auto_login_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Buttons frame
        buttons_frame = ttk.Frame(form_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))

        # Login button
        login_btn = ttk.Button(buttons_frame, text="🚀 Login", style="Primary.TButton",
                             command=self._attempt_login)
        login_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Create account button
        create_btn = ttk.Button(buttons_frame, text="👤 Create Account",
                              command=self._show_create_account)
        create_btn.pack(side=tk.LEFT)

        # Bind Enter key to login
        self.root.bind('<Return>', lambda e: self._attempt_login())

        # Status label
        self.status_label = ttk.Label(main_frame, text="", foreground="red")
        self.status_label.pack(pady=(0, 10))

        # Footer
        footer_label = ttk.Label(main_frame,
                               text="Welcome to HRM-Gemini AI!\nYour intelligent companion awaits.",
                               font=("Arial", 8), justify=tk.CENTER)
        footer_label.pack(pady=(20, 0))

    def _check_auto_login(self):
        """Check for auto-login credentials"""
        def check():
            try:
                auto_login_available, username = self.login_system.check_auto_login()

                if auto_login_available and username:
                    self.root.after(0, lambda: self._perform_auto_login(username))
                else:
                    # Set default username if available
                    self.root.after(0, lambda: self._load_saved_credentials())

            except Exception as e:
                print(f"Auto-login check error: {e}")

        thread = threading.Thread(target=check, daemon=True)
        thread.start()

    def _load_saved_credentials(self):
        """Load saved credentials if available"""
        try:
            sessions_file = Path("config/sessions.json")
            if sessions_file.exists():
                with open(sessions_file, 'r') as f:
                    sessions = json.load(f)

                # Find the most recent session with remember_me
                for username, session_data in sessions.items():
                    if session_data.get('remember_me', False):
                        self.username_var.set(username)
                        break
        except Exception:
            pass

    def _perform_auto_login(self, username):
        """Perform automatic login"""
        try:
            # Create loading indicator
            loading_label = ttk.Label(self.root, text=f"🔄 Auto-login as {username}...",
                                    foreground="blue")
            loading_label.pack(pady=10)

            self.root.update()

            # Simulate login process
            time.sleep(1)

            # Get stored password (in real implementation, this would be more secure)
            with open(Path("config/sessions.json"), 'r') as f:
                sessions = json.load(f)

            if username in sessions and sessions[username].get('auto_login'):
                # In a real system, you'd store encrypted passwords
                # For demo purposes, we'll use a default password
                success, message = self.login_system.authenticate_user(username, "admin123")

                loading_label.destroy()

                if success:
                    self.login_system.save_session(username, True, True)
                    self.login_successful = True
                    self.status_label.config(text="✅ Auto-login successful!", foreground="green")
                    self.root.after(1000, self.root.destroy)
                else:
                    self.status_label.config(text="❌ Auto-login failed", foreground="red")
            else:
                loading_label.destroy()
                self.status_label.config(text="Auto-login expired", foreground="orange")

        except Exception as e:
            loading_label.destroy()
            self.status_label.config(text=f"❌ Auto-login error: {e}", foreground="red")

    def _attempt_login(self):
        """Attempt to login with provided credentials"""
        username = self.username_var.get().strip()
        password = self.password_var.get()
        remember_me = self.remember_var.get()
        auto_login = self.auto_login_var.get()

        if not username or not password:
            self.status_label.config(text="❌ Please enter both username and password", foreground="red")
            return

        # Clear previous status
        self.status_label.config(text="🔄 Logging in...", foreground="blue")
        self.root.update()

        def login():
            try:
                success, message = self.login_system.authenticate_user(username, password)

                self.root.after(0, lambda: self._handle_login_result(success, message, username, remember_me, auto_login))

            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"❌ Login error: {e}", foreground="red"))

        thread = threading.Thread(target=login, daemon=True)
        thread.start()

    def _handle_login_result(self, success, message, username, remember_me, auto_login):
        """Handle login result"""
        if success:
            self.login_system.save_session(username, auto_login, remember_me)
            self.login_successful = True
            self.status_label.config(text="✅ " + message, foreground="green")

            # Close login window after success
            self.root.after(1000, self.root.destroy)
        else:
            self.status_label.config(text="❌ " + message, foreground="red")

    def _show_create_account(self):
        """Show create account dialog"""
        create_window = tk.Toplevel(self.root)
        create_window.title("Create Account")
        create_window.geometry("350x300")
        create_window.resizable(False, False)
        create_window.transient(self.root)
        create_window.grab_set()

        # Username
        ttk.Label(create_window, text="Username:").pack(pady=(20, 5))
        username_var = tk.StringVar()
        username_entry = ttk.Entry(create_window, textvariable=username_var, width=30)
        username_entry.pack(pady=(0, 10))

        # Password
        ttk.Label(create_window, text="Password:").pack(pady=(0, 5))
        password_var = tk.StringVar()
        password_entry = ttk.Entry(create_window, textvariable=password_var, show="*", width=30)
        password_entry.pack(pady=(0, 10))

        # Confirm Password
        ttk.Label(create_window, text="Confirm Password:").pack(pady=(0, 5))
        confirm_var = tk.StringVar()
        confirm_entry = ttk.Entry(create_window, textvariable=confirm_var, show="*", width=30)
        confirm_entry.pack(pady=(0, 10))

        def create_account():
            username = username_var.get().strip()
            password = password_var.get()
            confirm = confirm_var.get()

            if not username or not password:
                messagebox.showerror("Error", "Please fill in all fields")
                return

            if password != confirm:
                messagebox.showerror("Error", "Passwords do not match")
                return

            if len(password) < 6:
                messagebox.showerror("Error", "Password must be at least 6 characters")
                return

            success, message = self.login_system.create_user(username, password)

            if success:
                messagebox.showinfo("Success", message)
                create_window.destroy()
                self.username_var.set(username)
                self.password_var.set(password)
            else:
                messagebox.showerror("Error", message)

        ttk.Button(create_window, text="Create Account", command=create_account).pack(pady=20)

        # Focus on username entry
        username_entry.focus()
