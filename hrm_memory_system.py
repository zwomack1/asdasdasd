#!/usr/bin/env python3
"""
HRM Brain Memory System - Advanced Persistent Memory and Training
A comprehensive memory system that stores everything like a brain for continuous learning
"""

import sys
import os
import json
import pickle
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import threading
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class HRMMemorySystem:
    """Comprehensive memory system that acts like a brain for the AI"""

    def __init__(self, storage_path: str = "brain_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Initialize storage files
        self.memory_db_path = self.storage_path / "memory.db"
        self.knowledge_base_path = self.storage_path / "knowledge_base.json"
        self.training_data_path = self.storage_path / "training_data.pkl"
        self.file_registry_path = self.storage_path / "file_registry.json"
        self.rpg_data_path = self.storage_path / "rpg_data.json"

        # Initialize memory structures
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.knowledge_base = {}
        self.file_registry = {}
        self.rpg_context = {}
        self.training_data = []

        # Memory management
        self.max_short_term_items = 1000
        self.max_training_items = 50000
        self.auto_save_interval = 300  # 5 minutes
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize database and load existing data
        self._init_database()
        self._load_memory()

        # Start auto-save thread
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()

    def _init_database(self):
        """Initialize SQLite database for structured data"""
        self.conn = sqlite3.connect(str(self.memory_db_path))
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp REAL,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_contents (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                content TEXT,
                content_type TEXT,
                processed_at REAL,
                metadata TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                context TEXT,
                messages TEXT,
                rpg_state TEXT,
                timestamp REAL
            )
        ''')

        self.conn.commit()

    def _load_memory(self):
        """Load existing memory from storage"""
        try:
            # Load knowledge base
            if self.knowledge_base_path.exists():
                with open(self.knowledge_base_path, 'r') as f:
                    self.knowledge_base = json.load(f)

            # Load training data
            if self.training_data_path.exists():
                with open(self.training_data_path, 'rb') as f:
                    self.training_data = pickle.load(f)

            # Load file registry
            if self.file_registry_path.exists():
                with open(self.file_registry_path, 'r') as f:
                    self.file_registry = json.load(f)

            # Load RPG data
            if self.rpg_data_path.exists():
                with open(self.rpg_data_path, 'r') as f:
                    self.rpg_context = json.load(f)

            print(f"✅ Loaded memory: {len(self.knowledge_base)} knowledge items, "
                  f"{len(self.training_data)} training samples, "
                  f"{len(self.file_registry)} files")

        except Exception as e:
            print(f"⚠️ Memory loading error: {e}")

    def _auto_save_loop(self):
        """Automatic saving loop"""
        while True:
            time.sleep(self.auto_save_interval)
            self._save_memory()

    def _save_memory(self):
        """Save memory to persistent storage"""
        try:
            # Save knowledge base
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)

            # Save training data
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.training_data, f)

            # Save file registry
            with open(self.file_registry_path, 'w') as f:
                json.dump(self.file_registry, f, indent=2)

            # Save RPG data
            with open(self.rpg_data_path, 'w') as f:
                json.dump(self.rpg_context, f, indent=2)

            print(f"💾 Memory saved at {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            print(f"❌ Memory save error: {e}")

    def store_memory(self, content: str, memory_type: str = "general",
                    metadata: Dict[str, Any] = None, importance: float = 0.5):
        """Store information in memory with categorization"""
        memory_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()

        memory_item = {
            'id': memory_id,
            'content': content,
            'type': memory_type,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'importance': importance
        }

        # Store in database
        self.cursor.execute('''
            INSERT OR REPLACE INTO memory_items
            (id, type, content, metadata, timestamp, importance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            memory_id,
            memory_type,
            content,
            json.dumps(metadata or {}),
            time.time(),
            importance
        ))
        self.conn.commit()

        # Store in memory structures
        if memory_type == "short_term":
            self.short_term_memory[memory_id] = memory_item
            self._cleanup_short_term_memory()
        else:
            self.long_term_memory[memory_id] = memory_item

        # Auto-train on new content
        self.executor.submit(self._auto_train_on_content, content, memory_type)

        return memory_id

    def _cleanup_short_term_memory(self):
        """Clean up short-term memory to prevent overflow"""
        if len(self.short_term_memory) > self.max_short_term_items:
            # Remove least important items
            sorted_items = sorted(
                self.short_term_memory.items(),
                key=lambda x: x[1]['importance']
            )
            items_to_remove = len(self.short_term_memory) - self.max_short_term_items
            for i in range(items_to_remove):
                del self.short_term_memory[sorted_items[i][0]]

    def _auto_train_on_content(self, content: str, content_type: str):
        """Automatically train on new content"""
        try:
            # Add to training data
            training_sample = {
                'content': content,
                'type': content_type,
                'timestamp': time.time(),
                'processed': False
            }

            self.training_data.append(training_sample)

            # Keep training data size manageable
            if len(self.training_data) > self.max_training_items:
                self.training_data = self.training_data[-self.max_training_items:]

            # Update knowledge base
            self._update_knowledge_base(content, content_type)

            print(f"🎓 Auto-trained on: {content[:50]}...")

        except Exception as e:
            print(f"❌ Auto-training error: {e}")

    def _update_knowledge_base(self, content: str, content_type: str):
        """Update knowledge base with new information"""
        # Extract keywords and concepts
        words = content.lower().split()
        keywords = [word for word in words if len(word) > 3]

        for keyword in keywords[:10]:  # Limit to top 10 keywords
            if keyword not in self.knowledge_base:
                self.knowledge_base[keyword] = []

            # Store content reference
            content_ref = {
                'content': content[:200],  # Store preview
                'type': content_type,
                'timestamp': time.time()
            }

            self.knowledge_base[keyword].append(content_ref)

            # Keep only recent references
            if len(self.knowledge_base[keyword]) > 20:
                self.knowledge_base[keyword] = self.knowledge_base[keyword][-20:]

    def recall_memory(self, query: str, memory_type: str = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query"""
        query_lower = query.lower()

        # Search in database
        search_conditions = []
        params = []

        if memory_type:
            search_conditions.append("type = ?")
            params.append(memory_type)

        # Add text search
        search_conditions.append("content LIKE ?")
        params.append(f"%{query}%")

        where_clause = " AND ".join(search_conditions) if search_conditions else "1=1"

        self.cursor.execute(f'''
            SELECT * FROM memory_items
            WHERE {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', params + [limit])

        results = []
        for row in self.cursor.fetchall():
            result = {
                'id': row[0],
                'type': row[1],
                'content': row[2],
                'metadata': json.loads(row[3]),
                'timestamp': row[4],
                'access_count': row[5],
                'importance': row[6]
            }
            results.append(result)

        # Update access counts
        for result in results:
            self.cursor.execute('''
                UPDATE memory_items
                SET access_count = access_count + 1
                WHERE id = ?
            ''', (result['id'],))

        self.conn.commit()

        return results

    def search_knowledge(self, query: str, limit: int = 5) -> Dict[str, List]:
        """Search knowledge base for relevant information"""
        query_words = query.lower().split()
        results = {}

        for word in query_words:
            if word in self.knowledge_base:
                results[word] = self.knowledge_base[word][-limit:]

        return results

    def process_file_upload(self, file_path: str, file_type: str = None) -> str:
        """Process uploaded file and add to memory"""
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)

            # Check if already processed
            if file_hash in self.file_registry:
                return f"File already processed: {file_path_obj.name}"

            # Read and process file based on type
            content = self._extract_file_content(file_path, file_type)

            # Store file metadata
            file_info = {
                'hash': file_hash,
                'path': str(file_path_obj),
                'name': file_path_obj.name,
                'size': file_path_obj.stat().st_size,
                'type': file_type or self._detect_file_type(file_path),
                'processed_at': time.time()
            }

            self.file_registry[file_hash] = file_info

            # Store content in database
            self.cursor.execute('''
                INSERT OR REPLACE INTO file_contents
                (file_hash, file_path, content, content_type, processed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                file_hash,
                str(file_path_obj),
                content,
                file_type or self._detect_file_type(file_path),
                time.time(),
                json.dumps(file_info)
            ))
            self.conn.commit()

            # Add to memory and auto-train
            memory_id = self.store_memory(
                content,
                memory_type="file_upload",
                metadata={
                    'file_path': str(file_path_obj),
                    'file_hash': file_hash,
                    'file_type': file_type
                },
                importance=0.8
            )

            print(f"📁 Processed file: {file_path_obj.name}")
            return f"Successfully processed file: {file_path_obj.name} (Memory ID: {memory_id})"

        except Exception as e:
            error_msg = f"File processing error: {e}"
            print(f"❌ {error_msg}")
            return error_msg

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_file_content(self, file_path: str, file_type: str = None) -> str:
        """Extract content from various file types"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
            # Text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif file_extension in ['.pdf']:
            # PDF files
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                return f"[PDF Content - Install PyPDF2 to extract text from {file_path}]"

        elif file_extension in ['.docx']:
            # Word documents
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                return f"[DOCX Content - Install python-docx to extract text from {file_path}]"

        else:
            # Binary or unsupported files
            return f"[Binary file: {file_path} - Content extraction not supported for {file_extension} files]"

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        extension = Path(file_path).suffix.lower()
        type_map = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.py': 'text/python',
            '.js': 'text/javascript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.json': 'application/json',
            '.pdf': 'application/pdf',
            '.docx': 'application/docx',
            '.jpg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }
        return type_map.get(extension, 'application/octet-stream')

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        self.cursor.execute("SELECT COUNT(*) FROM memory_items")
        total_memories = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT type, COUNT(*) FROM memory_items GROUP BY type")
        type_counts = dict(self.cursor.fetchall())

        self.cursor.execute("SELECT COUNT(*) FROM file_contents")
        total_files = self.cursor.fetchone()[0]

        return {
            'total_memories': total_memories,
            'memory_by_type': type_counts,
            'total_files': total_files,
            'knowledge_base_size': len(self.knowledge_base),
            'training_data_size': len(self.training_data),
            'short_term_memory_size': len(self.short_term_memory),
            'long_term_memory_size': len(self.long_term_memory)
        }

    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old memory data"""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        # Remove old memories with low importance
        self.cursor.execute('''
            DELETE FROM memory_items
            WHERE timestamp < ? AND importance < 0.7
        ''', (cutoff_time,))

        deleted_count = self.cursor.rowcount
        self.conn.commit()

        print(f"🧹 Cleaned up {deleted_count} old memory items")
        return deleted_count

    def export_memory(self, export_path: str):
        """Export entire memory to file"""
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'version': '1.0'
            },
            'knowledge_base': self.knowledge_base,
            'file_registry': self.file_registry,
            'rpg_context': self.rpg_context,
            'training_data_count': len(self.training_data)
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"📤 Memory exported to: {export_path}")
        return export_path

    def close(self):
        """Clean shutdown"""
        self._save_memory()
        self.conn.close()
        self.executor.shutdown(wait=True)
        print("🧠 Memory system shut down")
