import os
import json
import shutil
from pathlib import Path
from datetime import datetime

class ModManager:
    def __init__(self, mods_dir):
        self.mods_dir = mods_dir
        self.backup_dir = os.path.join(os.path.dirname(mods_dir), "mods_backup")
        self.ensure_directories()
        self.mods_cache = {}
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.mods_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def scan_mods(self):
        """Scan the mods directory and return a list of installed mods"""
        self.mods_cache.clear()
        
        if not os.path.exists(self.mods_dir):
            return []
            
        for item in os.listdir(self.mods_dir):
            mod_path = os.path.join(self.mods_dir, item)
            
            # Handle mod folders
            if os.path.isdir(mod_path) and not item.startswith('.'):
                mod_info = self.get_mod_info(mod_path)
                if mod_info:
                    self.mods_cache[item] = mod_info
            
            # Handle zip files
            elif item.lower().endswith('.zip'):
                mod_info = self.get_zip_mod_info(mod_path)
                if mod_info:
                    self.mods_cache[item] = mod_info
                    
        return list(self.mods_cache.values())
    
    def get_mod_info(self, mod_path):
        """Extract mod information from mod directory"""
        mod_info = {
            'name': os.path.basename(mod_path),
            'path': mod_path,
            'size': self.get_folder_size(mod_path),
            'modified': datetime.fromtimestamp(os.path.getmtime(mod_path)).isoformat(),
            'enabled': not mod_path.endswith('.disabled')
        }
        
        # Check for mod info file (modInfo.json or similar)
        info_files = [f for f in os.listdir(mod_path) if f.lower() == 'modinfo.json']
        if info_files:
            try:
                with open(os.path.join(mod_path, info_files[0]), 'r') as f:
                    mod_info.update(json.load(f))
            except Exception as e:
                print(f"Error reading mod info: {e}")
                
        return mod_info
    
    def get_zip_mod_info(self, zip_path):
        """Extract mod information from zip file"""
        mod_info = {
            'name': os.path.splitext(os.path.basename(zip_path))[0],
            'path': zip_path,
            'size': os.path.getsize(zip_path),
            'modified': datetime.fromtimestamp(os.path.getmtime(zip_path)).isoformat(),
            'enabled': True,
            'type': 'archive',
            'description': 'Mod archive (not extracted)'
        }
        
        return mod_info
    
    def get_folder_size(self, path):
        """Calculate total size of a directory in bytes"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except (OSError, IOError):
                    pass  # Skip files that can't be accessed
        return total
    
    def install_mod(self, mod_archive):
        """Install a mod from archive file"""
        # TODO: Implement mod archive extraction
        pass
    
    def uninstall_mod(self, mod_name):
        """Uninstall a mod by name"""
        if mod_name in self.mods_cache:
            mod_path = self.mods_cache[mod_name]['path']
            try:
                if os.path.isdir(mod_path):
                    shutil.rmtree(mod_path)
                elif os.path.isfile(mod_path):
                    os.remove(mod_path)
                return True
            except Exception as e:
                print(f"Error uninstalling mod: {e}")
                return False
        return False
    
    def toggle_mod(self, mod_name, enable=True):
        """Enable or disable a mod"""
        if mod_name in self.mods_cache:
            mod_path = self.mods_cache[mod_name]['path']
            new_path = mod_path
            
            if enable and mod_path.endswith('.disabled'):
                new_path = mod_path[:-9]  # Remove .disabled
            elif not enable and not mod_path.endswith('.disabled'):
                new_path = f"{mod_path}.disabled"
                
            if new_path != mod_path:
                try:
                    os.rename(mod_path, new_path)
                    self.mods_cache[mod_name]['path'] = new_path
                    self.mods_cache[mod_name]['enabled'] = enable
                    return True
                except Exception as e:
                    print(f"Error toggling mod: {e}")
        return False
    
    def backup_mods(self):
        """Create a backup of all installed mods"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"mods_backup_{timestamp}")
        
        try:
            shutil.copytree(self.mods_dir, backup_path)
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def restore_backup(self, backup_path):
        """Restore mods from a backup"""
        if not os.path.exists(backup_path):
            return False
            
        try:
            # Clear existing mods
            if os.path.exists(self.mods_dir):
                shutil.rmtree(self.mods_dir)
            
            # Restore from backup
            shutil.copytree(backup_path, self.mods_dir)
            return True
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
