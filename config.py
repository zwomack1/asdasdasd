import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Manages application configuration"""
    
    DEFAULT_CONFIG = {
        'app': {
            'name': 'BeamNG.drive Mod Manager',
            'version': '1.0.0',
            'theme': 'dark',
            'language': 'en',
            'check_for_updates': True,
            'backup_before_install': True,
            'max_backups': 5,
            'auto_scan': True,
            'notifications_enabled': True,
        },
        'paths': {
            'beamng_mods_dir': '',  # Will be auto-detected
            'backup_dir': '',  # Will be set relative to mods dir
            'log_file': 'beamng_mod_manager.log',
            'cache_dir': os.path.join(os.path.expanduser('~'), '.beamng_mod_manager', 'cache'),
        },
        'ui': {
            'window_width': 1024,
            'window_height': 768,
            'window_maximized': False,
            'sidebar_width': 200,
            'show_hidden_mods': False,
            'mod_columns': ['name', 'version', 'author', 'enabled'],
            'sort_column': 'name',
            'sort_order': 'asc',
        },
        'network': {
            'use_proxy': False,
            'proxy_url': '',
            'proxy_port': 0,
            'proxy_username': '',
            'proxy_password': '',
            'timeout': 30,
        },
        'repositories': {
            'official': 'https://beamng.com/mods',
            'community': 'https://www.beamng.com/resources/',
            'custom_repositories': [],
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        config_dir = os.path.join(os.path.expanduser('~'), '.beamng_mod_manager')
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, 'config.json')
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self._merge_configs(self.config, loaded_config)
            
            # Set default paths if not configured
            if not self.config['paths']['beamng_mods_dir']:
                self._auto_detect_mods_dir()
            
            if not self.config['paths']['backup_dir'] and self.config['paths']['beamng_mods_dir']:
                self.config['paths']['backup_dir'] = os.path.join(
                    os.path.dirname(self.config['paths']['beamng_mods_dir']),
                    'mods_backup'
                )
                
            # Ensure backup directory exists
            if self.config['paths']['backup_dir']:
                os.makedirs(self.config['paths']['backup_dir'], exist_ok=True)
                
            # Ensure cache directory exists
            if self.config['paths']['cache_dir']:
                os.makedirs(self.config['paths']['cache_dir'], exist_ok=True)
                
            self.save()  # Save to ensure any new defaults are persisted
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # If there's an error, use default config and save it
            self.config = self.DEFAULT_CONFIG.copy()
            self.save()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults"""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def _auto_detect_mods_dir(self):
        """Try to auto-detect the BeamNG.drive mods directory"""
        common_paths = [
            os.path.expanduser("~/Documents/BeamNG.drive/mods"),
            "C:\\Program Files (x86)\\Steam\\steamapps\\common\\BeamNG.drive\\mods",
            "D:\\Games\\BeamNG.drive\\mods"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.config['paths']['beamng_mods_dir'] = path
                return
        
        # If not found, use a default path and create it
        default_path = os.path.expanduser("~/Documents/BeamNG.drive/mods")
        os.makedirs(default_path, exist_ok=True)
        self.config['paths']['beamng_mods_dir'] = default_path
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot notation key
        
        Example: config.get('app.theme') -> 'dark'
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Set a configuration value by dot notation key
        
        Args:
            key: Dot notation key (e.g., 'app.theme')
            value: Value to set
            save: Whether to save the config after setting the value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        if save:
            self.save()
    
    def save(self):
        """Save the current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()

# Global configuration instance
config = Config()
