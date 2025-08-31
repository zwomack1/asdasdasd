#!/usr/bin/env python3
"""
HRM-Gemini AI Login System
Automated user authentication and session management
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
import threading

class LoginSystem:
    """Automated login system for HRM-Gemini AI"""

    def __init__(self, main_app=None):
        self.main_app = main_app
        self.users_file = Path("config/users.json")
        self.sessions_file = Path("config/sessions.json")
        self.current_user = None
        self.auto_login_enabled = True
        self.remember_me = True

        # Create config directory if it doesn't exist
        self.users_file.parent.mkdir(exist_ok=True)

        # Initialize user database
        self._initialize_user_database()

    def _initialize_user_database(self):
        """Initialize the user database with default admin user"""
        if not self.users_file.exists():
            default_users = {
                "admin": {
                    "password_hash": self._hash_password("admin123"),
                    "role": "administrator",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "preferences": {
                        "theme": "dark",
                        "auto_login": True,
                        "remember_me": True
                    }
                }
            }
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)

        if not self.sessions_file.exists():
            with open(self.sessions_file, 'w') as f:
                json.dump({}, f)

    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username, password):
        """Authenticate a user"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)

            if username in users:
                user_data = users[username]
                if user_data['password_hash'] == self._hash_password(password):
                    # Update last login
                    user_data['last_login'] = datetime.now().isoformat()
                    with open(self.users_file, 'w') as f:
                        json.dump(users, f, indent=2)

                    self.current_user = {
                        'username': username,
                        'role': user_data['role'],
                        'preferences': user_data['preferences']
                    }
                    return True, "Login successful!"

            return False, "Invalid username or password"

        except Exception as e:
            return False, f"Authentication error: {e}"

    def create_user(self, username, password, role="user"):
        """Create a new user account"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)

            if username in users:
                return False, "Username already exists"

            users[username] = {
                "password_hash": self._hash_password(password),
                "role": role,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "preferences": {
                    "theme": "light",
                    "auto_login": False,
                    "remember_me": False
                }
            }

            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)

            return True, f"User '{username}' created successfully"

        except Exception as e:
            return False, f"User creation error: {e}"

    def check_auto_login(self):
        """Check if auto-login is available"""
        try:
            if not self.sessions_file.exists():
                return False, None

            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)

            # Look for valid session
            for username, session_data in sessions.items():
                if session_data.get('auto_login', False):
                    expires_at = datetime.fromisoformat(session_data['expires_at'])
                    if datetime.now() < expires_at:
                        return True, username

            return False, None

        except Exception:
            return False, None

    def save_session(self, username, auto_login=False, remember_me=False):
        """Save user session"""
        try:
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)

            sessions[username] = {
                'auto_login': auto_login,
                'remember_me': remember_me,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=30)).isoformat()
            }

            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)

        except Exception as e:
            print(f"Session save error: {e}")

    def logout(self):
        """Logout current user"""
        if self.current_user:
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)

                if self.current_user['username'] in sessions:
                    del sessions[self.current_user['username']]

                    with open(self.sessions_file, 'w') as f:
                        json.dump(sessions, f)

            except Exception as e:
                print(f"Logout error: {e}")

        self.current_user = None

    def get_current_user(self):
        """Get current user information"""
        return self.current_user

    def update_user_preferences(self, preferences):
        """Update user preferences"""
        if not self.current_user:
            return False, "No user logged in"

        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)

            if self.current_user['username'] in users:
                users[self.current_user['username']]['preferences'].update(preferences)

                with open(self.users_file, 'w') as f:
                    json.dump(users, f, indent=2)

                self.current_user['preferences'].update(preferences)
                return True, "Preferences updated successfully"

            return False, "User not found"

        except Exception as e:
            return False, f"Preference update error: {e}"
