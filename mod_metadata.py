"""
mod_metadata.py - Handles extraction and processing of mod metadata
"""
import os
import json
import zipfile
import py7zr
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ModMetadataExtractor:
    """Handles extraction and processing of mod metadata from various sources"""
    
    # Common mod metadata filenames
    METADATA_FILES = ['mod.json', 'info.json', 'modinfo.json']
    
    @staticmethod
    def extract_metadata(mod_path: str) -> Tuple[Dict[str, Any], bool]:
        """
        Extract metadata from a mod file or directory
        
        Args:
            mod_path: Path to mod file (.zip, .7z) or directory
            
        Returns:
            Tuple of (metadata_dict, is_archive)
        """
        mod_path = os.path.abspath(mod_path)
        
        if os.path.isdir(mod_path):
            return ModMetadataExtractor._extract_from_dir(mod_path), False
        elif os.path.isfile(mod_path):
            if mod_path.lower().endswith('.zip'):
                return ModMetadataExtractor._extract_from_zip(mod_path), True
            elif mod_path.lower().endswith(('.7z', '.7zip')):
                return ModMetadataExtractor._extract_from_7z(mod_path), True
        
        logger.warning(f"Unsupported mod format: {mod_path}")
        return {}, False
    
    @staticmethod
    def _extract_from_dir(mod_dir: str) -> Dict[str, Any]:
        """Extract metadata from a directory"""
        metadata = {}
        
        # Try to find and parse metadata files
        for meta_file in ModMetadataExtractor.METADATA_FILES:
            meta_path = os.path.join(mod_dir, meta_file)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    break
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse {meta_path}: {e}")
        
        # If no metadata file found, try to infer from directory structure
        if not metadata:
            mod_name = os.path.basename(mod_dir)
            metadata = {
                'name': mod_name,
                'version': '1.0.0',
                'description': 'No description available',
                'author': 'Unknown',
                'inferred': True
            }
        
        # Add common fields if missing
        metadata.setdefault('name', os.path.basename(mod_dir))
        metadata.setdefault('version', '1.0.0')
        metadata.setdefault('description', 'No description available')
        metadata.setdefault('author', 'Unknown')
        
        # Add file information
        metadata['path'] = mod_dir
        metadata['mod_type'] = 'directory'
        metadata['last_modified'] = os.path.getmtime(mod_dir)
        
        return metadata
    
    @staticmethod
    def _extract_from_zip(zip_path: str) -> Dict[str, Any]:
        """Extract metadata from a zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                return ModMetadataExtractor._extract_from_archive(zip_ref, zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            logger.error(f"Failed to read zip file {zip_path}: {e}")
            return {}
    
    @staticmethod
    def _extract_from_7z(seven_zip_path: str) -> Dict[str, Any]:
        """Extract metadata from a 7z file"""
        try:
            with py7zr.SevenZipFile(seven_zip_path, 'r') as zip_ref:
                return ModMetadataExtractor._extract_from_archive(zip_ref, seven_zip_path)
        except (py7zr.Bad7zFile, OSError) as e:
            logger.error(f"Failed to read 7z file {seven_zip_path}: {e}")
            return {}
    
    @staticmethod
    def _extract_from_archive(archive, archive_path: str) -> Dict[str, Any]:
        """Common extraction logic for archive files"""
        metadata = {}
        mod_name = os.path.splitext(os.path.basename(archive_path))[0]
        
        # Look for metadata files in the archive
        for member in archive.namelist():
            # Skip directories and non-metadata files
            if member.endswith('/'):
                continue
                
            filename = os.path.basename(member)
            if filename.lower() in [f.lower() for f in ModMetadataExtractor.METADATA_FILES]:
                try:
                    # Read the metadata file content
                    if hasattr(archive, 'read'):  # zipfile
                        content = archive.read(member)
                    else:  # py7zr
                        content = archive.read(member)[member].read()
                    
                    # Parse JSON content
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    
                    metadata = json.loads(content)
                    break
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse {member} in {archive_path}: {e}")
        
        # If no metadata file found, use archive name as mod name
        if not metadata:
            metadata = {
                'name': mod_name,
                'version': '1.0.0',
                'description': 'No description available',
                'author': 'Unknown',
                'inferred': True
            }
        
        # Add common fields if missing
        metadata.setdefault('name', mod_name)
        metadata.setdefault('version', '1.0.0')
        metadata.setdefault('description', 'No description available')
        metadata.setdefault('author', 'Unknown')
        
        # Add file information
        metadata['path'] = archive_path
        metadata['mod_type'] = 'archive'
        metadata['archive_format'] = 'zip' if isinstance(archive, zipfile.ZipFile) else '7z'
        metadata['size'] = os.path.getsize(archive_path)
        metadata['last_modified'] = os.path.getmtime(archive_path)
        
        return metadata

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """
        Validate that the metadata contains required fields
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        required_fields = ['name']
        return all(field in metadata for field in required_fields)

    @staticmethod
    def get_mod_icon(mod_path: str) -> Optional[str]:
        """
        Get the path to the mod's icon if it exists
        
        Args:
            mod_path: Path to the mod directory or archive
            
        Returns:
            Optional[str]: Path to the icon file or None if not found
        """
        # Common icon filenames
        icon_files = ['icon.png', 'thumbnail.png', 'preview.jpg', 'icon.jpg']
        
        if os.path.isdir(mod_path):
            # Check for icon in mod directory
            for icon_file in icon_files:
                icon_path = os.path.join(mod_path, icon_file)
                if os.path.exists(icon_path):
                    return icon_path
        else:
            # For archives, we'd need to extract the icon to a temp location
            # This is a simplified version that just checks for existence
            try:
                if mod_path.lower().endswith('.zip'):
                    with zipfile.ZipFile(mod_path, 'r') as zip_ref:
                        for icon_file in icon_files:
                            if icon_file in zip_ref.namelist():
                                return f"{mod_path}/{icon_file}"
                elif mod_path.lower().endswith(('.7z', '.7zip')):
                    with py7zr.SevenZipFile(mod_path, 'r') as zip_ref:
                        for icon_file in icon_files:
                            if icon_file in zip_ref.getnames():
                                return f"{mod_path}/{icon_file}"
            except Exception as e:
                logger.warning(f"Error checking for icon in {mod_path}: {e}")
        
        return None
