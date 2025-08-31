#!/usr/bin/env python3
"""
HRM-Gemini AI Web API Backend
FastAPI-based REST API for the Gemini-like web interface
"""

import sys
import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize FastAPI app
app = FastAPI(
    title="HRM-Gemini AI API",
    description="Advanced AI system with memory, RPG, and file processing capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_cors_middleware(
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core systems
memory_system = None
file_upload_system = None
rpg_chatbot = None
performance_monitor = None

def initialize_systems():
    """Initialize all HRM systems"""
    global memory_system, file_upload_system, rpg_chatbot, performance_monitor

    try:
        # Initialize Memory System
        from hrm_memory_system import HRMMemorySystem
        memory_system = HRMMemorySystem()
        print("✅ Memory System initialized")

        # Initialize File Upload System
        from file_upload_system import FileUploadSystem
        file_upload_system = FileUploadSystem(memory_system=memory_system)
        print("✅ File Upload System initialized")

        # Initialize RPG Chatbot
        from rpg_chatbot import RPGChatbot
        rpg_chatbot = RPGChatbot(memory_system=memory_system)
        print("✅ RPG Chatbot initialized")

        # Initialize Performance Monitor
        from performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor(None)  # Will be set to controller later
        print("✅ Performance Monitor initialized")

        return True

    except Exception as e:
        print(f"❌ System initialization error: {e}")
        return False

# Initialize systems on startup
@app.on_event("startup")
async def startup_event():
    """Initialize systems on application startup"""
    initialize_systems()

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "web_user"
    context: Optional[Dict[str, Any]] = None

class MemoryRequest(BaseModel):
    content: str
    memory_type: Optional[str] = "general"
    importance: Optional[float] = 0.5
    user_id: Optional[str] = "web_user"

class RPGRequest(BaseModel):
    command: str
    args: Optional[List[str]] = []
    user_id: Optional[str] = "web_user"

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_info: Optional[Dict[str, Any]] = None
    content_preview: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "memory": memory_system is not None,
            "file_upload": file_upload_system is not None,
            "rpg": rpg_chatbot is not None,
            "performance": performance_monitor is not None
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")

        # Process the chat message
        response = memory_system.process_input(
            request.message,
            request.user_id,
            "text",
            request.context
        )

        # Record performance
        if performance_monitor:
            performance_monitor.record_command(request.message, 0.1)  # Placeholder timing

        return {
            "response": response,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

@app.post("/api/memory/remember")
async def remember_endpoint(request: MemoryRequest):
    """Store information in memory"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")

        memory_id = memory_system.store_memory(
            request.content,
            request.memory_type,
            importance=request.importance
        )

        return {
            "memory_id": memory_id,
            "status": "stored",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory storage error: {e}")

@app.get("/api/memory/recall")
async def recall_endpoint(query: str, user_id: str = "web_user", limit: int = 5):
    """Recall information from memory"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")

        results = memory_system.recall_memory(query, limit=limit)

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory recall error: {e}")

@app.get("/api/memory/search")
async def search_endpoint(query: str, user_id: str = "web_user"):
    """Search knowledge base"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")

        results = memory_system.search_knowledge(query)

        return {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@app.get("/api/memory/stats")
async def memory_stats_endpoint():
    """Get memory system statistics"""
    try:
        if not memory_system:
            raise HTTPException(status_code=503, detail="Memory system not available")

        stats = memory_system.get_memory_stats()

        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

@app.post("/api/upload")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    description: str = "",
    user_id: str = "web_user"
):
    """Upload and process a file"""
    try:
        if not file_upload_system:
            raise HTTPException(status_code=503, detail="File upload system not available")

        # Save uploaded file temporarily
        temp_path = project_root / "temp" / f"{uuid.uuid4()}_{file.filename}"
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file
        result = file_upload_system.upload_file(
            str(temp_path),
            description=description
        )

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

        if result['success']:
            return UploadResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result['error'])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@app.post("/api/rpg/command")
async def rpg_command_endpoint(request: RPGRequest):
    """Process RPG commands"""
    try:
        if not rpg_chatbot:
            raise HTTPException(status_code=503, detail="RPG system not available")

        response = rpg_chatbot.process_rpg_command(
            request.user_id,
            request.command,
            request.args
        )

        return {
            "response": response,
            "command": request.command,
            "args": request.args,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RPG command error: {e}")

@app.get("/api/rpg/status")
async def rpg_status_endpoint(user_id: str = "web_user"):
    """Get RPG system status"""
    try:
        if not rpg_chatbot:
            raise HTTPException(status_code=503, detail="RPG system not available")

        status = rpg_chatbot.get_rpg_status()

        return {
            "status": status,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RPG status error: {e}")

@app.get("/api/files")
async def list_files_endpoint(user_id: str = "web_user"):
    """List uploaded files"""
    try:
        if not file_upload_system:
            raise HTTPException(status_code=503, detail="File system not available")

        files = file_upload_system.list_uploaded_files()

        return {
            "files": files,
            "count": len(files),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File list error: {e}")

@app.get("/api/performance/dashboard")
async def performance_dashboard_endpoint():
    """Get performance dashboard data"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")

        data = performance_monitor.get_dashboard_data()

        return {
            "dashboard": data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance dashboard error: {e}")

@app.post("/api/performance/report")
async def performance_report_endpoint(filename: Optional[str] = None):
    """Generate and export performance report"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")

        report_path = performance_monitor.export_report(filename)

        return {
            "report_path": str(report_path),
            "status": "exported",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {e}")

@app.get("/api/system/status")
async def system_status_endpoint():
    """Get comprehensive system status"""
    try:
        status = {
            "overall_health": "healthy",
            "systems": {
                "memory": memory_system is not None,
                "file_upload": file_upload_system is not None,
                "rpg": rpg_chatbot is not None,
                "performance": performance_monitor is not None
            },
            "timestamp": datetime.now().isoformat()
        }

        # Get detailed stats if available
        if memory_system:
            status["memory_stats"] = memory_system.get_memory_stats()

        if performance_monitor:
            status["performance"] = performance_monitor.get_dashboard_data()

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System status error: {e}")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("🚀 Starting HRM-Gemini AI Web API...")
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
