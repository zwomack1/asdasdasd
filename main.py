import sys
import os
import json
import psutil
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QListWidget, QLabel, 
                            QFileDialog, QMessageBox, QTabWidget, QStatusBar,
                            QListWidgetItem, QProgressBar, QInputDialog, QMenu,
                            QGroupBox, QLineEdit, QCheckBox, QAction, QDialog)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QPalette, QFont, QColor

from mod_manager import ModManager
from config import config
from folder_scanner import FirstTimeSetupDialog, FolderSearchDialog

class ModScanThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    
    def __init__(self, mod_manager):
        super().__init__()
        self.mod_manager = mod_manager
        
    def run(self):
        try:
            mods = self.mod_manager.scan_mods()
            self.finished.emit(mods)
        except Exception as e:
            print(f"Error scanning mods: {e}")
            self.finished.emit([])

class BeamNGModManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeamNG.drive Mod Manager")
        self.setMinimumSize(1000, 700)
        
        # Check for first-time setup
        self.check_first_time_setup()
        
        # Initialize mod manager
        mods_dir = config.get('paths.beamng_mods_dir', os.path.join(os.path.expanduser("~"), "Documents", "BeamNG.drive", "mods"))
        self.mod_manager = ModManager(mods_dir)
        self.current_mods = []
        
        # Game status tracking
        self.game_running = False
        self.game_status_timer = QTimer()
        self.game_status_timer.timeout.connect(self.check_game_status)
        self.game_status_timer.start(5000)  # Check every 5 seconds
        
        # Initialize UI
        self.init_ui()
        
        # Start scanning for mods and check game status
        self.scan_mods()
        self.check_game_status()
    
    def check_first_time_setup(self):
        """Check if this is the first time running and show setup dialog"""
        # Check if mod folder path is configured
        current_path = config.get('paths.beamng_mods_dir', '')
        first_run = config.get('app.first_run', True)
        
        if first_run or not current_path or not os.path.exists(current_path):
            setup_dialog = FirstTimeSetupDialog(self)
            if setup_dialog.exec_() == QDialog.Accepted:
                selected_folder = setup_dialog.get_selected_folder()
                if selected_folder:
                    config.set('paths.beamng_mods_dir', selected_folder)
                    QMessageBox.information(
                        self,
                        "Setup Complete",
                        f"BeamNG mod folder set to:\n{selected_folder}"
                    )
            
            # Mark first run as complete
            config.set('app.first_run', False)
        
    def init_ui(self):
        # Apply modern styling
        self.apply_modern_styling()
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create tab widget with modern styling
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
        """)
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_installed_tab()
        self.create_available_tab()
        self.create_settings_tab()
        
        # Create enhanced status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #2b2b2b;
                color: white;
                border-top: 1px solid #3a3a3a;
                font-weight: bold;
            }
        """)
        self.setStatusBar(self.status_bar)
        
        # Game status with visual indicator
        self.game_status_widget = QWidget()
        status_layout = QHBoxLayout(self.game_status_widget)
        status_layout.setContentsMargins(5, 2, 5, 2)
        
        self.game_status_indicator = QLabel("●")
        self.game_status_indicator.setStyleSheet("""
            QLabel {
                color: #ff4444;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        
        self.game_status_label = QLabel("BeamNG.drive: Not Running")
        self.game_status_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                margin-left: 5px;
            }
        """)
        
        status_layout.addWidget(self.game_status_indicator)
        status_layout.addWidget(self.game_status_label)
        status_layout.addStretch()
        
        self.status_bar.addPermanentWidget(self.game_status_widget)
        self.status_bar.showMessage("Ready")
        
        # Menu bar
        self.create_menu_bar()
    
    def apply_modern_styling(self):
        """Apply modern dark theme styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #0078d4;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #404040;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2b2b2b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #0078d4;
                font-size: 12px;
                font-weight: bold;
            }
            QLineEdit {
                background-color: #2b2b2b;
                border: 2px solid #3a3a3a;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #3a3a3a;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QProgressBar {
                border: 2px solid #3a3a3a;
                border-radius: 6px;
                background-color: #2b2b2b;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
            QLabel {
                color: white;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                border-bottom: 1px solid #3a3a3a;
            }
            QMenuBar::item {
                padding: 8px 12px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            QMenu {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #3a3a3a;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)
        
    def create_installed_tab(self):
        """Create the Installed Mods tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.scan_mods)
        
        self.install_btn = QPushButton("Install Mod")
        self.install_btn.clicked.connect(self.install_mod)
        
        self.uninstall_btn = QPushButton("Uninstall Selected")
        self.uninstall_btn.clicked.connect(self.uninstall_selected_mods)
        self.uninstall_btn.setEnabled(False)
        
        self.backup_btn = QPushButton("Backup Mods")
        self.backup_btn.clicked.connect(self.backup_mods)
        
        self.restore_btn = QPushButton("Restore Backup")
        self.restore_btn.clicked.connect(self.restore_backup)
        
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.install_btn)
        toolbar.addWidget(self.uninstall_btn)
        toolbar.addWidget(self.backup_btn)
        toolbar.addWidget(self.restore_btn)
        toolbar.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Mod list
        self.installed_mods_list = QListWidget()
        self.installed_mods_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.installed_mods_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.installed_mods_list.customContextMenuRequested.connect(self.show_context_menu)
        self.installed_mods_list.itemSelectionChanged.connect(self.update_ui_state)
        
        # Add widgets to layout
        layout.addLayout(toolbar)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.installed_mods_list)
        
        self.tabs.addTab(tab, "Installed Mods")
        
    def create_available_tab(self):
        """Create the Available Mods tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search mods...")
        search_layout.addWidget(self.search_input)
        
        # Available mods list
        self.available_mods_list = QListWidget()
        
        # Add widgets to layout
        layout.addLayout(search_layout)
        layout.addWidget(self.available_mods_list)
        
        self.tabs.addTab(tab, "Available Mods")
        
    def create_settings_tab(self):
        """Create the Settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # BeamNG Path settings
        path_group = QGroupBox("BeamNG.drive Path")
        path_layout = QVBoxLayout()
        
        # Game directory
        game_dir_layout = QHBoxLayout()
        self.path_edit = QLineEdit(config.get('paths.beamng_mods_dir', ''))
        self.path_edit.setReadOnly(True)
        
        browse_game_btn = QPushButton("Browse Game Directory...")
        browse_game_btn.clicked.connect(self.browse_beamng_path)
        
        game_dir_layout.addWidget(QLabel("Game Directory:"))
        game_dir_layout.addWidget(self.path_edit)
        game_dir_layout.addWidget(browse_game_btn)
        
        # Mod folder finder
        mod_folder_layout = QHBoxLayout()
        self.mod_folder_edit = QLineEdit(config.get('paths.beamng_mods_dir', ''))
        self.mod_folder_edit.setReadOnly(True)
        
        browse_mod_btn = QPushButton("Browse Mod Folder...")
        browse_mod_btn.clicked.connect(self.browse_mod_folder)
        
        auto_find_btn = QPushButton("Auto-Find Mod Folder")
        auto_find_btn.clicked.connect(self.auto_find_mod_folder)
        
        search_btn = QPushButton("Search Computer")
        search_btn.clicked.connect(self.search_computer_for_mods)
        
        mod_folder_layout.addWidget(QLabel("Mod Folder:"))
        mod_folder_layout.addWidget(self.mod_folder_edit)
        mod_folder_layout.addWidget(browse_mod_btn)
        mod_folder_layout.addWidget(auto_find_btn)
        mod_folder_layout.addWidget(search_btn)
        
        # Folder status
        self.folder_status_label = QLabel()
        self.update_folder_status()
        
        path_layout.addLayout(game_dir_layout)
        path_layout.addLayout(mod_folder_layout)
        path_layout.addWidget(self.folder_status_label)
        path_group.setLayout(path_layout)
        
        # Game Status settings
        game_status_group = QGroupBox("Game Status")
        game_status_layout = QVBoxLayout()
        
        self.game_status_info = QLabel()
        self.update_game_status_info()
        
        refresh_status_btn = QPushButton("Refresh Game Status")
        refresh_status_btn.clicked.connect(self.check_game_status)
        
        game_status_layout.addWidget(self.game_status_info)
        game_status_layout.addWidget(refresh_status_btn)
        game_status_group.setLayout(game_status_layout)
        
        # Appearance settings
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout()
        
        self.dark_mode = QCheckBox("Dark Mode")
        appearance_layout.addWidget(self.dark_mode)
        appearance_group.setLayout(appearance_layout)
        
        # Add widgets to main layout
        layout.addWidget(path_group)
        layout.addWidget(game_status_group)
        layout.addWidget(appearance_group)
        layout.addStretch()
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        self.tabs.addTab(tab, "Settings")
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        scan_mods_action = QAction("Scan for Mods", self)
        scan_mods_action.triggered.connect(self.scan_mods)
        file_menu.addAction(scan_mods_action)
        
        file_menu.addSeparator()
        
        backup_action = QAction("Backup Mods", self)
        backup_action.triggered.connect(self.backup_mods)
        file_menu.addAction(backup_action)
        
        restore_action = QAction("Restore Backup", self)
        restore_action.triggered.connect(self.restore_backup)
        file_menu.addAction(restore_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        check_updates_action = QAction("Check for Updates", self)
        check_updates_action.triggered.connect(self.check_for_updates)
        tools_menu.addAction(check_updates_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    # Game Status Methods
    def check_game_status(self):
        """Check if BeamNG.drive is currently running"""
        beamng_processes = ['BeamNG.drive.x64.exe', 'BeamNG.drive.exe', 'BeamNG.x64.exe']
        
        self.game_running = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] in beamng_processes:
                    self.game_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Update status bar with fancy indicators
        if self.game_running:
            self.game_status_indicator.setStyleSheet("""
                QLabel {
                    color: #00ff88;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            self.game_status_label.setText("BeamNG.drive: Running")
            self.game_status_label.setStyleSheet("""
                QLabel {
                    color: #00ff88;
                    font-weight: bold;
                    margin-left: 5px;
                }
            """)
        else:
            self.game_status_indicator.setStyleSheet("""
                QLabel {
                    color: #ff4444;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)
            self.game_status_label.setText("BeamNG.drive: Not Running")
            self.game_status_label.setStyleSheet("""
                QLabel {
                    color: #ff4444;
                    font-weight: bold;
                    margin-left: 5px;
                }
            """)
        
        # Update detailed info if settings tab is visible
        if hasattr(self, 'game_status_info'):
            self.update_game_status_info()
    
    def update_game_status_info(self):
        """Update detailed game status information"""
        if hasattr(self, 'game_status_info'):
            if self.game_running:
                self.game_status_info.setText(
                    "⚠️ <b>BeamNG.drive is currently running</b><br>"
                    "Some mod operations may require restarting the game to take effect."
                )
                self.game_status_info.setStyleSheet("""
                    QLabel {
                        color: #ffaa00;
                        font-weight: bold;
                        background-color: #2b2b2b;
                        border: 2px solid #ffaa00;
                        border-radius: 6px;
                        padding: 10px;
                    }
                """)
            else:
                self.game_status_info.setText(
                    "✅ <b>BeamNG.drive is not running</b><br>"
                    "Safe to install, uninstall, or modify mods."
                )
                self.game_status_info.setStyleSheet("""
                    QLabel {
                        color: #00ff88;
                        font-weight: bold;
                        background-color: #2b2b2b;
                        border: 2px solid #00ff88;
                        border-radius: 6px;
                        padding: 10px;
                    }
                """)
    
    # Mod Folder Methods
    def browse_mod_folder(self):
        """Browse for mod folder"""
        current_path = self.mod_folder_edit.text()
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Mod Folder",
            current_path if os.path.exists(current_path) else os.path.expanduser("~")
        )
        
        if path:
            self.mod_folder_edit.setText(path)
            self.update_folder_status()
    
    def auto_find_mod_folder(self):
        """Automatically find the mod folder using comprehensive search"""
        # Show progress dialog
        progress = QProgressBar()
        progress.setRange(0, 0)  # Indeterminate
        progress.setVisible(True)
        
        # Search common locations more thoroughly
        search_locations = [
            # Current config
            config.get('paths.beamng_mods_dir', ''),
            
            # Standard user locations
            os.path.join(os.path.expanduser("~"), "Documents", "BeamNG.drive", "mods"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "BeamNG.drive", "mods"),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "BeamNG.drive", "mods"),
            
            # Public locations
            os.path.join("C:", "Users", "Public", "Documents", "BeamNG.drive", "mods"),
            
            # Steam locations
            os.path.join("C:", "Program Files (x86)", "Steam", "steamapps", "common", "BeamNG.drive", "mods"),
            os.path.join("C:", "Program Files", "Steam", "steamapps", "common", "BeamNG.drive", "mods"),
            
            # Alternative drive locations
            os.path.join("D:", "Games", "BeamNG.drive", "mods"),
            os.path.join("D:", "Steam", "steamapps", "common", "BeamNG.drive", "mods"),
            os.path.join("E:", "Games", "BeamNG.drive", "mods"),
            os.path.join("E:", "Steam", "steamapps", "common", "BeamNG.drive", "mods"),
        ]
        
        found_paths = []
        
        for path in search_locations:
            if path and os.path.exists(path) and os.path.isdir(path):
                # Check if it actually contains mods or looks like a mod folder
                try:
                    items = os.listdir(path)
                    mod_count = 0
                    
                    for item in items:
                        item_path = os.path.join(path, item)
                        if (os.path.isdir(item_path) and not item.startswith('.')) or item.lower().endswith('.zip'):
                            mod_count += 1
                    
                    if mod_count > 0:
                        found_paths.append((path, mod_count))
                except (PermissionError, OSError):
                    continue
        
        progress.setVisible(False)
        
        if found_paths:
            # Sort by mod count (descending) and take the best one
            found_paths.sort(key=lambda x: x[1], reverse=True)
            best_path, mod_count = found_paths[0]
            
            self.mod_folder_edit.setText(best_path)
            self.update_folder_status()
            
            # Show results dialog if multiple found
            if len(found_paths) > 1:
                message = f"Found mod folder at:\n{best_path}\n\n"
                message += f"Contains {mod_count} mods/archives.\n\n"
                message += f"Other locations found: {len(found_paths) - 1}"
                QMessageBox.information(self, "Mod Folder Found", message)
            else:
                QMessageBox.information(self, "Mod Folder Found", 
                    f"Found mod folder at:\n{best_path}\n\nContains {mod_count} mods/archives.")
        else:
            QMessageBox.warning(self, "Mod Folder Not Found", 
                "Could not automatically find any BeamNG mod folders.\n"
                "Please use 'Search Computer' for a more thorough search or browse manually.")
    
    def update_folder_status(self):
        """Update folder status display"""
        mod_path = self.mod_folder_edit.text()
        
        if os.path.exists(mod_path) and os.path.isdir(mod_path):
            try:
                items = os.listdir(mod_path)
                folder_count = len([f for f in items if os.path.isdir(os.path.join(mod_path, f)) and not f.startswith('.')])
                zip_count = len([f for f in items if f.lower().endswith('.zip')])
                total_count = folder_count + zip_count
                
                status_text = f"✅ Folder exists - {folder_count} mod folders, {zip_count} zip files ({total_count} total)"
                self.folder_status_label.setStyleSheet("color: #00ff88; font-weight: bold;")
            except (PermissionError, OSError):
                status_text = "⚠️ Folder exists but cannot be accessed"
                self.folder_status_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        else:
            status_text = "❌ Folder does not exist"
            self.folder_status_label.setStyleSheet("color: red;")
        
        self.folder_status_label.setText(status_text)
    
    # Mod Management Methods
    def scan_mods(self):
        """Scan for installed mods"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.scan_thread = ModScanThread(self.mod_manager)
        self.scan_thread.finished.connect(self.on_mod_scan_complete)
        self.scan_thread.start()
    
    def on_mod_scan_complete(self, mods):
        """Handle completion of mod scanning"""
        self.progress_bar.setVisible(False)
        self.current_mods = mods
        self.update_mods_list()
    
    def update_mods_list(self):
        """Update the installed mods list widget"""
        self.installed_mods_list.clear()
        
        for mod in self.current_mods:
            item = QListWidgetItem()
            
            # Create display text
            name = mod.get('name', 'Unknown Mod')
            size = self.format_size(mod.get('size', 0))
            enabled_text = "✓" if mod.get('enabled', True) else "✗"
            mod_type = mod.get('type', 'folder')
            
            # Add type indicator
            type_indicator = "📦" if mod_type == 'archive' else "📁"
            
            item.setText(f"{enabled_text} {type_indicator} {name} ({size})")
            item.setData(Qt.UserRole, mod)
            
            # Color coding
            if mod.get('enabled', True):
                if mod_type == 'archive':
                    item.setForeground(QColor("#ffaa00"))  # Orange for zip files
                else:
                    item.setForeground(QColor("white"))    # White for folders
            else:
                item.setForeground(QColor("gray"))
            
            self.installed_mods_list.addItem(item)
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def install_mod(self):
        """Install a new mod from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Mod File", 
            "", 
            "Mod Files (*.zip *.7z *.rar *.zipmod)"
        )
        
        if file_path:
            # TODO: Implement mod installation
            QMessageBox.information(self, "Install Mod", f"Would install mod from: {file_path}")
    
    def uninstall_selected_mods(self):
        """Uninstall selected mods"""
        selected_items = self.installed_mods_list.selectedItems()
        if not selected_items:
            return
            
        mod_names = [item.text() for item in selected_items]
        reply = QMessageBox.question(
            self,
            'Confirm Uninstall',
            f'Are you sure you want to uninstall the following mods?\n\n' + '\n'.join(f"- {name}" for name in mod_names),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for item in selected_items:
                mod_name = item.text()
                if self.mod_manager.uninstall_mod(mod_name):
                    self.installed_mods_list.takeItem(self.installed_mods_list.row(item))
    
    def toggle_selected_mods(self, enable=True):
        """Enable or disable selected mods"""
        selected_items = self.installed_mods_list.selectedItems()
        for item in selected_items:
            mod_name = item.text()
            if self.mod_manager.toggle_mod(mod_name, enable):
                # Update item appearance
                if enable:
                    item.setForeground(Qt.black)
                else:
                    item.setForeground(Qt.gray)
    
    def backup_mods(self):
        """Create a backup of all mods"""
        backup_path = self.mod_manager.backup_mods()
        if backup_path:
            QMessageBox.information(self, "Backup Created", f"Mods backed up to:\n{backup_path}")
        else:
            QMessageBox.warning(self, "Backup Failed", "Failed to create backup.")
    
    def restore_backup(self):
        """Restore mods from a backup"""
        backup_dir = self.mod_manager.backup_dir
        if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
            QMessageBox.information(self, "No Backups", "No backup files found.")
            return
            
        # Show a dialog to select a backup
        backup_dirs = sorted(os.listdir(backup_dir), reverse=True)
        backup, ok = QInputDialog.getItem(
            self,
            "Select Backup",
            "Choose a backup to restore:",
            backup_dirs,
            0,
            False
        )
        
        if ok and backup:
            reply = QMessageBox.question(
                self,
                'Confirm Restore',
                'This will replace all current mods with the selected backup. Continue?',
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                backup_path = os.path.join(backup_dir, backup)
                if self.mod_manager.restore_backup(backup_path):
                    QMessageBox.information(self, "Restore Complete", "Mods have been restored from backup.")
                    self.scan_mods()
                else:
                    QMessageBox.warning(self, "Restore Failed", "Failed to restore from backup.")
    
    def show_context_menu(self, position):
        """Show context menu for mod items"""
        selected_items = self.installed_mods_list.selectedItems()
        if not selected_items:
            return
            
        menu = QMenu()
        
        # Check if any selected mods are disabled
        any_disabled = any(
            not item.data(Qt.UserRole).get('enabled', True)
            for item in selected_items
        )
        
        # Only show enable/disable if selection is consistent
        if len(selected_items) == 1 or all(
            not item.data(Qt.UserRole).get('enabled', True)
            for item in selected_items
        ):
            if any_disabled:
                enable_action = menu.addAction("Enable")
                enable_action.triggered.connect(lambda: self.toggle_selected_mods(True))
            else:
                disable_action = menu.addAction("Disable")
                disable_action.triggered.connect(lambda: self.toggle_selected_mods(False))
        
        menu.addSeparator()
        
        uninstall_action = menu.addAction("Uninstall")
        uninstall_action.triggered.connect(self.uninstall_selected_mods)
        
        menu.exec(self.installed_mods_list.viewport().mapToGlobal(position))
    
    def update_ui_state(self):
        """Update UI elements based on selection"""
        has_selection = len(self.installed_mods_list.selectedItems()) > 0
        self.uninstall_btn.setEnabled(has_selection)
    
    # Settings Methods
    def browse_beamng_path(self):
        """Open a dialog to select the BeamNG.drive directory"""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select BeamNG.drive Directory",
            self.path_edit.text()
        )
        
        if path:
            self.path_edit.setText(path)
            # Update mod folder path automatically
            self.mod_folder_edit.setText(os.path.join(path, "mods"))
            self.update_folder_status()
    
    def search_computer_for_mods(self):
        """Open advanced search dialog to scan entire computer for mod folders"""
        search_dialog = FolderSearchDialog(self)
        if search_dialog.exec_() == QDialog.Accepted:
            selected_folder = search_dialog.get_selected_folder()
            if selected_folder:
                self.mod_folder_edit.setText(selected_folder)
                self.update_folder_status()
                QMessageBox.information(
                    self,
                    "Folder Selected",
                    f"Mod folder updated to:\n{selected_folder}"
                )
    
    def save_settings(self):
        """Save application settings"""
        new_path = self.path_edit.text()
        new_mod_path = self.mod_folder_edit.text()
        
        if new_mod_path != config.get('paths.beamng_mods_dir', ''):
            config.set('paths.beamng_mods_dir', new_mod_path)
            
            # Update mod manager paths
            self.mod_manager.mods_dir = new_mod_path
            self.mod_manager.backup_dir = os.path.join(os.path.dirname(new_mod_path), "mods_backup")
            
            # Rescan mods
            self.scan_mods()
        
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
    
    # Help Methods
    def check_for_updates(self):
        """Check for application updates"""
        # TODO: Implement update checking
        QMessageBox.information(
            self,
            "Check for Updates",
            "This feature will be implemented in a future version."
        )
    
    def show_documentation(self):
        """Open documentation in default web browser"""
        # TODO: Add actual documentation URL
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation will be available in a future version."
        )
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About BeamNG Mod Manager",
            "<h2>BeamNG.drive Mod Manager</h2>"
            f"<p>Version {config.get('app.version', '1.0.0')}</p>"
            "<p>A powerful mod manager for BeamNG.drive</p>"
            "<p>© 2023 Your Name</p>"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Set application info
    app.setApplicationName("BeamNG Mod Manager")
    app.setApplicationVersion(config.get('app.version', '1.0.0'))
    app.setOrganizationName("YourOrganization")
    
    # Create and show the main window
    try:
        window = BeamNGModManager()
        window.show()
        
        # Run the application
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"A critical error occurred:\n{str(e)}\n\nThe application will now exit.",
            QMessageBox.Ok
        )
        sys.exit(1)
