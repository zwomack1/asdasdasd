import os
import threading
import time
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QProgressBar, QListWidget, QListWidgetItem,
                            QMessageBox, QCheckBox, QGroupBox, QTextEdit)

class FolderScanThread(QThread):
    """Thread for scanning folders to find BeamNG mod directories"""
    progress = pyqtSignal(str)  # Current folder being scanned
    found_folder = pyqtSignal(str, int)  # Path and mod count
    finished = pyqtSignal(list)  # List of found folders
    
    def __init__(self, search_paths=None, deep_scan=False):
        super().__init__()
        self.search_paths = search_paths or self.get_default_search_paths()
        self.deep_scan = deep_scan
        self.should_stop = False
        self.found_folders = []
        
    def get_default_search_paths(self):
        """Get default paths to search for BeamNG folders"""
        paths = []
        
        # User directories
        user_home = os.path.expanduser("~")
        paths.extend([
            os.path.join(user_home, "Documents"),
            os.path.join(user_home, "AppData", "Local"),
            os.path.join(user_home, "AppData", "Roaming"),
        ])
        
        # Common game installation directories
        common_game_dirs = [
            "C:\\Program Files (x86)\\Steam\\steamapps\\common",
            "C:\\Program Files\\Steam\\steamapps\\common",
            "D:\\Games",
            "D:\\Steam\\steamapps\\common",
            "E:\\Games",
            "E:\\Steam\\steamapps\\common",
        ]
        
        for game_dir in common_game_dirs:
            if os.path.exists(game_dir):
                paths.append(game_dir)
        
        # If deep scan, add all drives
        if self.deep_scan:
            import string
            for drive in string.ascii_uppercase:
                drive_path = f"{drive}:\\"
                if os.path.exists(drive_path):
                    paths.append(drive_path)
        
        return paths
    
    def stop_scan(self):
        """Stop the scanning process"""
        self.should_stop = True
    
    def is_beamng_mod_folder(self, folder_path):
        """Check if a folder is likely a BeamNG mod directory"""
        folder_name = os.path.basename(folder_path).lower()
        
        # Check if folder name suggests it's a BeamNG mod folder
        beamng_indicators = [
            "beamng.drive",
            "beamng",
            "beam.ng",
            "mods"
        ]
        
        # Check if folder contains BeamNG-related files or subfolders
        if any(indicator in folder_name for indicator in beamng_indicators):
            # Look for mods subfolder or mod files
            mods_path = os.path.join(folder_path, "mods")
            if os.path.exists(mods_path) and os.path.isdir(mods_path):
                return mods_path
            
            # Check if current folder contains mod-like files
            if self.contains_mod_files(folder_path):
                return folder_path
        
        return None
    
    def contains_mod_files(self, folder_path):
        """Check if folder contains BeamNG mod files"""
        try:
            items = os.listdir(folder_path)
            
            # Look for common mod file patterns
            mod_patterns = ['.zip', '.jbeam', '.lua', '.json']
            mod_folders = 0
            
            for item in items:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    mod_folders += 1
                elif any(item.lower().endswith(pattern) for pattern in mod_patterns):
                    return True
            
            # If there are multiple folders, it might be a mods directory
            return mod_folders >= 2
            
        except (PermissionError, OSError):
            return False
    
    def count_mods_in_folder(self, folder_path):
        """Count the number of mods in a folder"""
        try:
            items = os.listdir(folder_path)
            mod_count = 0
            
            for item in items:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    mod_count += 1
                elif item.lower().endswith('.zip'):
                    mod_count += 1
            
            return mod_count
        except (PermissionError, OSError):
            return 0
    
    def scan_directory(self, directory, max_depth=3, current_depth=0):
        """Recursively scan a directory for BeamNG mod folders"""
        if self.should_stop or current_depth > max_depth:
            return
        
        try:
            self.progress.emit(f"Scanning: {directory}")
            
            # Check if current directory is a BeamNG mod folder
            mod_folder = self.is_beamng_mod_folder(directory)
            if mod_folder:
                mod_count = self.count_mods_in_folder(mod_folder)
                if mod_count > 0:
                    self.found_folder.emit(mod_folder, mod_count)
                    self.found_folders.append((mod_folder, mod_count))
            
            # Scan subdirectories
            for item in os.listdir(directory):
                if self.should_stop:
                    break
                
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Skip certain system directories to speed up scan
                    skip_dirs = ['windows', 'program files', '$recycle.bin', 'system volume information']
                    if item.lower() not in skip_dirs:
                        self.scan_directory(item_path, max_depth, current_depth + 1)
        
        except (PermissionError, OSError, FileNotFoundError):
            # Skip directories we can't access
            pass
    
    def run(self):
        """Run the folder scanning process"""
        self.found_folders = []
        
        for search_path in self.search_paths:
            if self.should_stop:
                break
            
            if os.path.exists(search_path):
                max_depth = 2 if self.deep_scan else 3
                self.scan_directory(search_path, max_depth)
        
        # Sort by mod count (descending)
        self.found_folders.sort(key=lambda x: x[1], reverse=True)
        self.finished.emit(self.found_folders)


class FirstTimeSetupDialog(QDialog):
    """Dialog for first-time setup to find BeamNG mod folder"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeamNG Mod Manager - First Time Setup")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        self.selected_folder = None
        self.scan_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Welcome message
        welcome_label = QLabel(
            "<h2>Welcome to BeamNG Mod Manager!</h2>"
            "<p>Let's find your BeamNG.drive mod folder to get started.</p>"
        )
        layout.addWidget(welcome_label)
        
        # Scan options
        scan_group = QGroupBox("Scan Options")
        scan_layout = QVBoxLayout(scan_group)
        
        self.quick_scan_radio = QCheckBox("Quick Scan (Recommended)")
        self.quick_scan_radio.setChecked(True)
        self.quick_scan_radio.setToolTip("Scans common locations like Documents, AppData, and Steam folders")
        
        self.deep_scan_radio = QCheckBox("Deep Scan (Entire Computer)")
        self.deep_scan_radio.setToolTip("Scans all drives - may take several minutes")
        
        scan_layout.addWidget(self.quick_scan_radio)
        scan_layout.addWidget(self.deep_scan_radio)
        layout.addWidget(scan_group)
        
        # Scan controls
        scan_controls = QHBoxLayout()
        
        self.start_scan_btn = QPushButton("Start Automatic Scan")
        self.start_scan_btn.clicked.connect(self.start_scan)
        
        self.manual_browse_btn = QPushButton("Browse Manually")
        self.manual_browse_btn.clicked.connect(self.browse_manually)
        
        self.stop_scan_btn = QPushButton("Stop Scan")
        self.stop_scan_btn.clicked.connect(self.stop_scan)
        self.stop_scan_btn.setEnabled(False)
        
        scan_controls.addWidget(self.start_scan_btn)
        scan_controls.addWidget(self.manual_browse_btn)
        scan_controls.addWidget(self.stop_scan_btn)
        layout.addLayout(scan_controls)
        
        # Progress
        self.progress_label = QLabel("Ready to scan...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        
        # Results
        results_label = QLabel("Found BeamNG Mod Folders:")
        self.results_list = QListWidget()
        self.results_list.itemDoubleClicked.connect(self.select_folder)
        
        layout.addWidget(results_label)
        layout.addWidget(self.results_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("Select Folder")
        self.select_btn.clicked.connect(self.select_folder)
        self.select_btn.setEnabled(False)
        
        self.skip_btn = QPushButton("Skip Setup")
        self.skip_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.select_btn)
        button_layout.addWidget(self.skip_btn)
        layout.addLayout(button_layout)
    
    def start_scan(self):
        """Start the folder scanning process"""
        deep_scan = self.deep_scan_radio.isChecked()
        
        self.start_scan_btn.setEnabled(False)
        self.stop_scan_btn.setEnabled(True)
        self.results_list.clear()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create and start scan thread
        self.scan_thread = FolderScanThread(deep_scan=deep_scan)
        self.scan_thread.progress.connect(self.update_progress)
        self.scan_thread.found_folder.connect(self.add_found_folder)
        self.scan_thread.finished.connect(self.scan_finished)
        self.scan_thread.start()
    
    def stop_scan(self):
        """Stop the current scan"""
        if self.scan_thread:
            self.scan_thread.stop_scan()
            self.scan_thread.wait()
        
        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Scan stopped.")
    
    def update_progress(self, current_folder):
        """Update progress display"""
        self.progress_label.setText(f"Scanning: {current_folder}")
    
    def add_found_folder(self, folder_path, mod_count):
        """Add a found folder to the results list"""
        item = QListWidgetItem(f"{folder_path} ({mod_count} mods)")
        item.setData(32, folder_path)  # Store path in user data
        self.results_list.addItem(item)
        
        if self.results_list.count() == 1:
            self.select_btn.setEnabled(True)
            self.results_list.setCurrentRow(0)
    
    def scan_finished(self, found_folders):
        """Handle scan completion"""
        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if found_folders:
            self.progress_label.setText(f"Scan complete! Found {len(found_folders)} mod folders.")
        else:
            self.progress_label.setText("Scan complete. No mod folders found.")
            QMessageBox.information(
                self,
                "No Folders Found",
                "No BeamNG mod folders were found automatically.\n"
                "You can browse manually or skip setup and configure later."
            )
    
    def browse_manually(self):
        """Open file dialog to browse for mod folder manually"""
        from PyQt5.QtWidgets import QFileDialog
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select BeamNG Mod Folder",
            os.path.expanduser("~/Documents")
        )
        
        if folder:
            self.selected_folder = folder
            self.accept()
    
    def select_folder(self):
        """Select the currently highlighted folder"""
        current_item = self.results_list.currentItem()
        if current_item:
            self.selected_folder = current_item.data(32)
            self.accept()
    
    def get_selected_folder(self):
        """Get the selected mod folder path"""
        return self.selected_folder


class FolderSearchDialog(QDialog):
    """Advanced folder search dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search for BeamNG Mod Folders")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        
        self.scan_thread = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Search options
        options_group = QGroupBox("Search Options")
        options_layout = QVBoxLayout(options_group)
        
        self.search_all_drives = QCheckBox("Search all drives")
        self.search_common_locations = QCheckBox("Search common game locations")
        self.search_common_locations.setChecked(True)
        
        options_layout.addWidget(self.search_all_drives)
        options_layout.addWidget(self.search_common_locations)
        layout.addWidget(options_group)
        
        # Controls
        controls = QHBoxLayout()
        
        self.search_btn = QPushButton("Start Search")
        self.search_btn.clicked.connect(self.start_search)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_search)
        self.stop_btn.setEnabled(False)
        
        controls.addWidget(self.search_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Progress
        self.status_label = QLabel("Ready to search...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
        # Results
        layout.addWidget(QLabel("Search Results:"))
        self.results_list = QListWidget()
        layout.addWidget(self.results_list)
        
        # Log
        layout.addWidget(QLabel("Search Log:"))
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        layout.addWidget(self.log_text)
        
        # Buttons
        buttons = QHBoxLayout()
        
        self.select_btn = QPushButton("Select")
        self.select_btn.clicked.connect(self.accept)
        self.select_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        buttons.addWidget(self.select_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)
    
    def start_search(self):
        """Start the search process"""
        self.search_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.results_list.clear()
        self.log_text.clear()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Determine search paths
        search_paths = []
        if self.search_common_locations.isChecked():
            scanner = FolderScanThread()
            search_paths.extend(scanner.get_default_search_paths())
        
        deep_scan = self.search_all_drives.isChecked()
        
        self.scan_thread = FolderScanThread(search_paths, deep_scan)
        self.scan_thread.progress.connect(self.update_status)
        self.scan_thread.found_folder.connect(self.add_result)
        self.scan_thread.finished.connect(self.search_finished)
        self.scan_thread.start()
    
    def stop_search(self):
        """Stop the search"""
        if self.scan_thread:
            self.scan_thread.stop_scan()
            self.scan_thread.wait()
        
        self.search_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def update_status(self, message):
        """Update status display"""
        self.status_label.setText(message)
        self.log_text.append(message)
    
    def add_result(self, folder_path, mod_count):
        """Add search result"""
        item = QListWidgetItem(f"{folder_path} ({mod_count} mods)")
        item.setData(32, folder_path)
        self.results_list.addItem(item)
        
        if self.results_list.count() == 1:
            self.select_btn.setEnabled(True)
            self.results_list.setCurrentRow(0)
    
    def search_finished(self, results):
        """Handle search completion"""
        self.search_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Search complete. Found {len(results)} folders.")
    
    def get_selected_folder(self):
        """Get selected folder"""
        current_item = self.results_list.currentItem()
        if current_item:
            return current_item.data(32)
        return None
