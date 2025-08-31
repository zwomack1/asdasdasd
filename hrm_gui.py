import tkinter as tk
from tkinter import ttk, scrolledtext
from services.integration_service import IntegrationService
import traceback

class HRMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HRM-Gemini AI")
        self.root.geometry("1200x800")
        
        # Create main panels
        self.create_chat_panel()
        self.create_device_panel()
        self.create_control_panel()
        
    def create_chat_panel(self):
        """Create the chat interface."""
        chat_frame = ttk.LabelFrame(self.root, text="AI Chat")
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(chat_frame, state='disabled')
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.user_input = tk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)
        
        send_btn = tk.Button(input_frame, text="Send", command=self.send_message)
        send_btn.pack(side=tk.RIGHT)
        
    def create_device_panel(self):
        """Create device and application detection panel."""
        device_frame = ttk.LabelFrame(self.root, text="Integration Targets")
        device_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Treeview for targets
        columns = ("Type", "Name")
        self.target_tree = ttk.Treeview(device_frame, columns=columns, show='headings')
        for col in columns:
            self.target_tree.heading(col, text=col)
        self.target_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Refresh button
        refresh_btn = tk.Button(device_frame, text="Refresh", command=self.refresh_targets)
        refresh_btn.pack(pady=5)
        
        # Inject button
        inject_btn = tk.Button(device_frame, text="Inject AI", command=self.inject_ai)
        inject_btn.pack(pady=5)
        
    def create_control_panel(self):
        """Create system control panel."""
        control_frame = ttk.LabelFrame(self.root, text="System Controls")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Control buttons
        start_btn = tk.Button(control_frame, text="Start AI", command=self.start_ai)
        start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        stop_btn = tk.Button(control_frame, text="Stop AI", command=self.stop_ai)
        stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
    def send_message(self, event=None):
        """Send user message to AI and display response."""
        message = self.user_input.get()
        if not message:
            return
        
        # Display user message
        self.display_message("user", message)
        
        # TODO: Get AI response
        # response = ai_model.process_input(message)
        response = "This is a placeholder response"
        
        # Display AI response
        self.display_message("ai", response)
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
    def display_message(self, sender, message):
        """Display message in chat history."""
        self.chat_history.config(state='normal')
        tag = "user" if sender == "user" else "ai"
        self.chat_history.insert(tk.END, f"{sender}: {message}\n", tag)
        self.chat_history.config(state='disabled')
        self.chat_history.see(tk.END)
        
    def refresh_targets(self):
        """Refresh list of integration targets."""
        # Clear existing
        for item in self.target_tree.get_children():
            self.target_tree.delete(item)
        
        # Get new targets
        integration_service = IntegrationService()
        targets = integration_service.find_integration_targets()
        
        # Add to treeview
        for target in targets:
            target_type = "Application" if 'pid' in target else "Device"
            name = target['name']
            self.target_tree.insert("", tk.END, values=(target_type, name))
        
    def inject_ai(self):
        """Inject AI into selected target."""
        selected = self.target_tree.selection()
        if not selected:
            return
        
        item = self.target_tree.item(selected)
        target_type, name = item['values']
        # TODO: Implement actual injection
        print(f"Injecting AI into {name} ({target_type})")
        
    def start_ai(self):
        """Start the AI system."""
        # TODO: Implement
        print("Starting AI system")
        
    def stop_ai(self):
        """Stop the AI system."""
        # TODO: Implement
        print("Stopping AI system")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = HRMGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
