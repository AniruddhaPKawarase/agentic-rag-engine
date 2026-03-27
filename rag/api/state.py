"""Shared FastAPI app state and runtime dependencies."""
import asyncio
import datetime
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "gpt-4.1")

# Import web search function
try:
    from services.web_search import web_search
    WEB_SEARCH_AVAILABLE = True
    print("✅ Web search module imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import web_search module: {e}")
    # Create a mock web search function for testing
    def web_search(query: str):
        """Mock web search function for testing"""
        return {
            "answer": f"This is a mock response for web search query: {query}",
            "sources": [
                {"title": "Mock Source 1", "url": "https://example.com/source1"},
                {"title": "Mock Source 2", "url": "https://example.com/source2"}
            ]
        }
    WEB_SEARCH_AVAILABLE = False

# Import your retrieval function
try:
    from retrieve import retrieve_context, initialize_index
    print("✅ Retrieve module imported successfully")
except Exception as e:
    print(f"⚠️ Could not import retrieve module: {e}. Using mock retrieval.")
    # Mock retrieval function — signature MUST match the real function
    def retrieve_context(query: str, top_k: int = 5, min_score: float = 0.1,
                        filter_source_type: Optional[str] = None,
                        filter_project_id: Optional[int] = None) -> List[Dict]:
        """Mock retrieval function for testing"""
        return []

    def initialize_index(project_id: Optional[int] = None) -> None:
        """Mock initialize_index for testing"""
        pass
    
# Import memory manager
try:
    from memory_manager import get_memory_manager, estimate_tokens, create_session_from_query
    MEMORY_MANAGER = get_memory_manager(
        storage_path="./conversation_sessions",
        max_sessions=200,
        max_tokens_per_session=10000,
        enable_persistence=True
    )
    print("✅ Memory manager initialized")
except ImportError:
    print("⚠️ Memory manager not available. Conversations will be stateless.")
    MEMORY_MANAGER = None


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="Construction Documentation QA API v2",
    description="REST API for construction documentation — RAG/Web/Hybrid with follow-up questions, hallucination rollback, token tracking, and streaming",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "https://ai.ifieldsmart.com,https://ai5.ifieldsmart.com,http://localhost:3000,http://localhost:8501").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
