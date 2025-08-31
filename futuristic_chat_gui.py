import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QLinearGradient, QPalette, QTextCursor, QFont

class FuturisticChatGUI(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("HRM-Gemini Futuristic Chat")
        self.setGeometry(100, 100, 800, 600)
        
        # Set dark theme with gradient
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f1e;
            }
            QTextEdit {
                background-color: #1a1a2e;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Arial';
                font-size: 14px;
            }
            QLineEdit {
                background-color: #1a1a2e;
                color: #ffffff;
                border: 1px solid #4a4a8a;
                border-radius: 15px;
                padding: 10px;
                font-family: 'Arial';
                font-size: 14px;
            }
            QPushButton {
                background-color: #4a4a8a;
                color: #ffffff;
                border: none;
                border-radius: 15px;
                padding: 10px 20px;
                font-family: 'Arial';
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a5aaa;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input area
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)
        
        # Initialize chat
        self.add_message("HRM", "Welcome to HRM-Gemini Futuristic Chat! How can I assist you today?")
    
    def add_message(self, sender, message):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Format sender
        cursor.insertText(f"{sender}: ", self.sender_format(sender))
        
        # Format message
        cursor.insertText(f"{message}\n\n", self.message_format(sender))
        
        self.chat_display.ensureCursorVisible()
    
    def sender_format(self, sender):
        if sender == "HRM":
            return "color: #00ffff; font-weight: bold;"
        else:
            return "color: #ffaa00; font-weight: bold;"
    
    def message_format(self, sender):
        return ""  # Use default
    
    def send_message(self):
        user_input = self.input_field.text().strip()
        if user_input:
            self.add_message("You", user_input)
            self.input_field.clear()
            
            # Get response from controller
            response = self.controller.get_chat_response(user_input)
            self.add_message("HRM", response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # For testing, we'll use a dummy controller
    class DummyController:
        def get_chat_response(self, user_input):
            return "This is a sample response from HRM-Gemini."
    
    window = FuturisticChatGUI(DummyController())
    window.show()
    sys.exit(app.exec_())
