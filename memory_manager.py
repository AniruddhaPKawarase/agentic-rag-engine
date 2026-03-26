"""
memory_manager.py
Conversation memory management for RAG application
"""

import json
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime
import pickle
import os
from pathlib import Path

# ==============================
# Data Models
# ==============================

@dataclass
class Message:
    """Represents a single message in conversation"""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: float
    tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tokens": self.tokens,
            "metadata": self.metadata or {}
        }

@dataclass
class ConversationContext:
    """Represents the relevant context for a conversation"""
    project_id: Optional[int] = None
    filter_source_type: Optional[str] = None
    recent_topics: List[str] = None
    custom_instructions: str = ""
    conversation_start_question: Optional[str] = None  # Add this field for backward compatibility
    
    def __post_init__(self):
        if self.recent_topics is None:
            self.recent_topics = []

@dataclass
class ConversationSummary:
    """Summary of older conversation parts"""
    summary_text: str
    message_count: int
    start_time: float
    end_time: float
    key_points: List[str]

@dataclass
class ConversationSession:
    """Complete conversation session"""
    session_id: str
    created_at: float
    last_accessed: float
    messages: List[Message]
    context: ConversationContext
    summaries: List[ConversationSummary]
    total_tokens: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.summaries is None:
            self.summaries = []
    
    def get_last_message_by_role(self, role: str) -> Optional[str]:
        """Get the content of the last message with the given role."""
        for msg in reversed(self.messages):
            if msg.role == role:
                return msg.content
        return None

    def get_last_user_message(self) -> Optional[str]:
        """Get the PREVIOUS user message content (not the current query).
        The current query is always the last user message, so we return second-to-last."""
        user_msgs = [m for m in self.messages if m.role == "user"]
        if len(user_msgs) >= 2:
            return user_msgs[-2].content
        # If only 1 user message, it's the current query — return it anyway
        # (better than None for first-question follow-ups)
        if len(user_msgs) == 1:
            return user_msgs[0].content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message content."""
        return self.get_last_message_by_role("assistant")

    def get_formatted_messages(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Format messages for OpenAI API"""
        formatted = []
        
        # Add conversation summary if exists
        if self.summaries:
            summary_text = "\n\n".join([s.summary_text for s in self.summaries[-2:]])  # Last 2 summaries
            if summary_text:
                formatted.append({
                    "role": "system",
                    "content": f"Previous conversation summary:\n{summary_text}\n\nCurrent conversation:"
                })
        
        # Add recent messages (with token limit consideration)
        for msg in self.messages[-10:]:  # Last 10 messages max
            if msg.role == "system" and not include_system:
                continue
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return formatted
    
    def add_message(self, role: str, content: str, tokens: int = 0, metadata: Optional[Dict] = None):
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            tokens=tokens,
            metadata=metadata
        )
        self.messages.append(message)
        self.total_tokens += tokens
        self.last_accessed = time.time()
    
    def should_summarize(self, max_tokens: int = 4000, max_messages: int = 20) -> bool:
        """Check if conversation needs summarization"""
        # Check token count
        if self.total_tokens > max_tokens:
            return True
        
        # Check message count
        if len(self.messages) > max_messages:
            return True
        
        # Check time gap (if old messages exist)
        if len(self.messages) > 5:
            oldest_time = self.messages[0].timestamp
            newest_time = self.messages[-1].timestamp
            if newest_time - oldest_time > 3600:  # 1 hour gap
                return True
        
        return False
    
    def get_full_conversation_history(self, include_summaries: bool = True) -> List[Dict[str, str]]:
        """Get complete conversation history including summaries"""
        formatted = []
        
        # Add all summaries first
        if include_summaries and self.summaries:
            all_summaries = "\n\n".join([s.summary_text for s in self.summaries])
            formatted.append({
                "role": "system",
                "content": f"Previous conversations summary:\n{all_summaries}"
            })
        
        # Add ALL messages (not just recent ones)
        for msg in self.messages:
            formatted.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            })
        
        return formatted

    def get_conversation_index(self) -> str:
        """
        Return a compact numbered list of ALL user questions in this session.

        Token-efficient: only questions are included (not full responses),
        and each is capped at 120 characters. Designed to be injected into every
        prompt so the LLM can answer meta-questions like "What was my first question?"

        Returns:
            A string like:
            "1. What are the fire safety requirements? (first question)
             2. Tell me about HVAC specifications
             3. What was my first question? (current question)"

            Returns empty string if no user messages exist.
        """
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        if not user_messages:
            return ""

        lines = []
        for i, msg in enumerate(user_messages, 1):
            content = msg.content.strip()
            if len(content) > 120:
                content = content[:117] + "..."
            label = ""
            if i == 1:
                label = " (first question)"
            elif i == len(user_messages):
                label = " (most recent question)"
            lines.append(f"{i}. {content}{label}")

        return "\n".join(lines)

    def get_conversation_for_llm(
        self,
        max_tokens: int = 4000,
        preserve_early_history: bool = True
    ) -> List[Dict[str, str]]:
        """Get conversation history optimized for LLM while preserving important context"""
        all_messages = []
        
        # Always include system messages if any
        system_msgs = []
        other_msgs = []
        
        # Separate system and other messages
        for msg in self.messages:
            if msg.role == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)
        
        # Include all system messages
        for msg in system_msgs:
            all_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Strategically select other messages
        selected_msgs = []
        
        if preserve_early_history and len(other_msgs) > 0:
            # Always include first message
            selected_msgs.append(other_msgs[0])
            
            # If there are many messages, include some from the middle
            if len(other_msgs) > 10:
                middle_index = len(other_msgs) // 2
                selected_msgs.append(other_msgs[middle_index])
            
            # Always include last 8 messages
            selected_msgs.extend(other_msgs[-8:])
            
            # Remove duplicates while preserving order
            seen_indices = set()
            unique_msgs = []
            for msg in selected_msgs:
                idx = other_msgs.index(msg)
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    unique_msgs.append(msg)
            selected_msgs = unique_msgs
        else:
            # Just take recent messages
            selected_msgs = other_msgs[-10:] if len(other_msgs) > 10 else other_msgs
        
        # Add selected messages
        for msg in selected_msgs:
            all_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add summaries context
        if self.summaries:
            summary_text = "\n\n".join([s.summary_text for s in self.summaries])
            all_messages.insert(0, {
                "role": "system",
                "content": f"Previous conversation summaries:\n{summary_text}\n\nCurrent conversation:"
            })
        
        # Token limit check (simplified)
        total_text = " ".join([m["content"] for m in all_messages])
        estimated_tokens = len(total_text) // 4
        
        if estimated_tokens > max_tokens:
            # Remove some middle messages but keep first and last
            if len(all_messages) > 6:
                # Keep: first system message, first user message, last 4 messages
                system_msg = all_messages[0] if all_messages[0]["role"] == "system" else None
                first_user_idx = next((i for i, m in enumerate(all_messages) if m["role"] == "user"), 0)
                
                filtered_messages = []
                if system_msg:
                    filtered_messages.append(system_msg)
                if first_user_idx > 0 and first_user_idx < len(all_messages):
                    filtered_messages.append(all_messages[first_user_idx])
                
                filtered_messages.extend(all_messages[-4:])
                all_messages = filtered_messages
        
        return all_messages

# ==============================
# Memory Manager
# ==============================

class MemoryManager:
    """Manages conversation memory with optimization"""
    
    def __init__(
        self,
        max_sessions: int = 100,
        max_tokens_per_session: int = 8000,
        max_messages_before_summary: int = 15,
        storage_path: Optional[str] = None,
        enable_persistence: bool = True
    ):
        self.max_sessions = max_sessions
        self.max_tokens_per_session = max_tokens_per_session
        self.max_messages_before_summary = max_messages_before_summary
        
        # Session storage
        self.sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        
        # Persistence
        self.storage_path = storage_path
        self.enable_persistence = enable_persistence
        
        if storage_path and enable_persistence:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self._load_sessions()
    
    def _generate_session_id(self, user_query: str, project_id: Optional[int] = None) -> str:
        """Generate a unique session ID"""
        # Create a hash based on query, project_id, and timestamp
        base_string = f"{user_query[:50]}_{project_id}_{time.time()}"
        session_hash = hashlib.md5(base_string.encode()).hexdigest()[:12]
        return f"session_{session_hash}"
    
    def create_session(
        self,
        user_query: str,
        project_id: Optional[int] = None,
        filter_source_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Create a new conversation session"""
        # Generate session ID if not provided
        if not session_id:
            session_id = self._generate_session_id(user_query, project_id)
        
        # Create context
        context = ConversationContext(
            project_id=project_id,
            filter_source_type=filter_source_type
        )
        
        # Create session
        session = ConversationSession(
            session_id=session_id,
            created_at=time.time(),
            last_accessed=time.time(),
            messages=[],
            context=context,
            summaries=[],
            metadata={
                "initial_query": user_query[:100],
                "created_date": datetime.now().isoformat()
            }
        )
        
        # Add to sessions (with LRU eviction if needed)
        self.sessions[session_id] = session
        
        if len(self.sessions) > self.max_sessions:
            # Remove least recently used session
            oldest_session_id = next(iter(self.sessions))
            del self.sessions[oldest_session_id]
        
        # Save if persistence is enabled
        if self.enable_persistence:
            self._save_session(session)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID. Falls back to S3 if not in memory."""
        session = self.sessions.get(session_id)
        if session:
            # Update access time and move to end (most recent)
            session.last_accessed = time.time()
            self.sessions.move_to_end(session_id)
            return session

        # S3 fallback: load individual session if evicted from memory
        session = self._load_session_from_s3(session_id)
        if session:
            session.last_accessed = time.time()
            self.sessions[session_id] = session
            self.sessions.move_to_end(session_id)
            # Evict LRU if over capacity
            if len(self.sessions) > self.max_sessions:
                oldest_id = next(iter(self.sessions))
                del self.sessions[oldest_id]
            print(f"Session {session_id} restored from S3 (cache miss)")
        return session

    def _load_session_from_s3(self, session_id: str) -> Optional[ConversationSession]:
        """Load a single session from S3 by session_id."""
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return None
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from s3_utils.operations import download_bytes
            from s3_utils.helpers import session_key
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "rag-agent")
            s3_k = session_key(s3_prefix, f"{session_id}.json")
            raw = download_bytes(s3_k)
            if raw:
                session_data = json.loads(raw.decode("utf-8"))
                return self._deserialize_session(session_data)
        except Exception as e:
            print(f"S3 single session load failed for {session_id}: {e}")
        return None
    
    def add_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: int = 0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add a message to an existing session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.add_message(role, content, tokens, metadata)
        
        # Check if we need to summarize
        if session.should_summarize():
            self._summarize_session(session)
        
        # Save if persistence is enabled
        if self.enable_persistence:
            self._save_session(session)
        
        return True
    
    def _summarize_session(self, session: ConversationSession):
        """Summarize older parts of conversation WITHOUT deleting original messages"""
        if len(session.messages) < 8:  # Increased minimum for summarization
            return
        
        # Only summarize if we have enough messages
        if len(session.messages) > self.max_messages_before_summary:
            # Take first half for summarization (but keep them)
            split_point = len(session.messages) // 2
            messages_to_summarize = session.messages[:split_point]
            
            # Create summary
            summary_text = self._create_detailed_summary(messages_to_summarize)
            
            summary = ConversationSummary(
                summary_text=summary_text,
                message_count=len(messages_to_summarize),
                start_time=messages_to_summarize[0].timestamp,
                end_time=messages_to_summarize[-1].timestamp,
                key_points=self._extract_key_points(messages_to_summarize)
            )
            
            # Add summary but DON'T remove original messages
            session.summaries.append(summary)
        
    def _create_detailed_summary(self, messages: List[Message]) -> str:
        """Create a more detailed summary of messages"""
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        if not user_messages:
            return "No user messages to summarize."
        
        # Extract key information
        first_query = user_messages[0].content
        important_queries = []
        
        # Identify important queries (first, and any with technical terms)
        technical_terms = ["requirement", "specification", "drawing", "code", "standard", 
                          "material", "installation", "compliance", "fire", "safety"]
        
        for i, msg in enumerate(user_messages):
            content_lower = msg.content.lower()
            is_important = (i == 0 or  # First query
                           i == len(user_messages) - 1 or  # Last query
                           any(term in content_lower for term in technical_terms) or
                           len(msg.content.split()) > 10)  # Detailed query
            
            if is_important:
                query_preview = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
                important_queries.append(f"Query {i+1}: {query_preview}")
        
        summary_parts = [
            f"Conversation covered {len(messages)} messages ({len(user_messages)} user queries).",
            f"First query was about: '{first_query[:100]}...'" if len(first_query) > 100 else f"First query: {first_query}",
        ]
        
        if important_queries:
            summary_parts.append("Important questions asked:")
            summary_parts.extend(important_queries[:5])  # Top 5 important queries
        
        return "\n".join(summary_parts)
    
    def _extract_key_points(self, messages: List[Message]) -> List[str]:
        """Extract key points from messages"""
        key_points = []
        
        for msg in messages:
            if msg.role == "user":
                # Extract key phrases (simplified)
                content = msg.content.lower()
                if "?" in content:
                    # Try to extract the question
                    question = content.split("?")[0].strip()
                    if len(question) > 10 and len(question) < 100:
                        key_points.append(f"Q: {question[:80]}...")
        
        return list(set(key_points))[:5]  # Return up to 5 unique key points
    
    def get_conversation_history(
        self,
        session_id: str,
        max_tokens: int = 3000,
        include_summaries: bool = True
    ) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        # Get formatted messages
        messages = session.get_formatted_messages(include_system=True)
        
        # Calculate token count (rough estimate)
        total_chars = sum(len(m["content"]) for m in messages)
        estimated_tokens = total_chars // 4
        
        # If too long, use sliding window
        if estimated_tokens > max_tokens:
            # Keep most recent messages that fit within limit
            kept_messages = []
            current_tokens = 0
            
            # Always include system message if present
            system_msgs = [m for m in messages if m["role"] == "system"]
            other_msgs = [m for m in messages if m["role"] != "system"]
            
            for msg in system_msgs:
                kept_messages.append(msg)
                current_tokens += len(msg["content"]) // 4
            
            # Add recent messages until limit
            for msg in reversed(other_msgs):
                msg_tokens = len(msg["content"]) // 4
                if current_tokens + msg_tokens <= max_tokens:
                    kept_messages.insert(len(system_msgs), msg)  # Insert after system messages
                    current_tokens += msg_tokens
                else:
                    break
        
        return messages
    
    def update_context(
        self,
        session_id: str,
        project_id: Optional[int] = None,
        filter_source_type: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ):
        """Update conversation context"""
        session = self.get_session(session_id)
        if not session:
            return
        
        if project_id is not None:
            session.context.project_id = project_id
        if filter_source_type is not None:
            session.context.filter_source_type = filter_source_type
        if custom_instructions is not None:
            session.context.custom_instructions = custom_instructions
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session — from S3 when STORAGE_BACKEND=s3, from local disk otherwise."""
        if session_id in self.sessions:
            del self.sessions[session_id]

            if self.enable_persistence:
                # --- S3 MODE: delete ONLY from S3 ---
                if os.getenv("STORAGE_BACKEND", "local") == "s3":
                    try:
                        import sys as _sys
                        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                        from s3_utils.operations import delete_object
                        from s3_utils.helpers import session_key
                        s3_prefix = os.getenv("S3_AGENT_PREFIX", "rag-agent")
                        s3_k = session_key(s3_prefix, f"{session_id}.json")
                        delete_object(s3_k)
                    except Exception:
                        pass
                # --- LOCAL MODE: delete from disk ---
                elif self.storage_path:
                    session_file = Path(self.storage_path) / f"{session_id}.json"
                    if session_file.exists():
                        session_file.unlink()

            return True
        return False
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "summary_count": len(session.summaries),
            "total_tokens": session.total_tokens,
            "created_at": datetime.fromtimestamp(session.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(session.last_accessed).isoformat(),
            "context": {
                "project_id": session.context.project_id,
                "filter_source_type": session.context.filter_source_type
            }
        }
    
    # ==============================
    # Persistence Methods
    # ==============================
    
    def _save_session(self, session: ConversationSession):
        """Save session — S3 only when STORAGE_BACKEND=s3, local disk otherwise."""
        if not self.enable_persistence:
            return

        # Convert to serializable format
        session_data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "messages": [msg.to_dict() for msg in session.messages],
            "context": asdict(session.context),
            "summaries": [
                {
                    "summary_text": s.summary_text,
                    "message_count": s.message_count,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "key_points": s.key_points
                }
                for s in session.summaries
            ],
            "total_tokens": session.total_tokens,
            "metadata": session.metadata
        }

        # --- S3 MODE: save ONLY to S3 (no local file) ---
        if os.getenv("STORAGE_BACKEND", "local") == "s3":
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from s3_utils.operations import upload_bytes
                from s3_utils.helpers import session_key
                s3_prefix = os.getenv("S3_AGENT_PREFIX", "rag-agent")
                s3_k = session_key(s3_prefix, f"{session.session_id}.json")
                upload_bytes(
                    json.dumps(session_data, ensure_ascii=False).encode("utf-8"),
                    s3_k,
                )
            except Exception as e:
                print(f"S3 session upload failed: {e}")
            return
        # --- END S3 MODE ---

        # LOCAL MODE: save to disk (original behavior)
        if not self.storage_path:
            return
        session_file = Path(self.storage_path) / f"{session.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def _load_sessions(self):
        """Load sessions — from S3 when STORAGE_BACKEND=s3, from local disk otherwise."""

        # --- S3 MODE: load ONLY from S3 (skip local disk entirely) ---
        if os.getenv("STORAGE_BACKEND", "local") == "s3":
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from s3_utils.operations import list_objects, download_bytes
                s3_prefix = os.getenv("S3_AGENT_PREFIX", "rag-agent")
                prefix = f"{s3_prefix}/conversation_sessions/"
                objects = list_objects(prefix, max_keys=self.max_sessions)
                for obj in objects:
                    try:
                        raw = download_bytes(obj["Key"])
                        if raw is None:
                            continue
                        session_data = json.loads(raw.decode("utf-8"))
                        session = self._deserialize_session(session_data)
                        self.sessions[session.session_id] = session
                    except Exception as e:
                        print(f"S3 session restore failed for {obj['Key']}: {e}")
                if self.sessions:
                    print(f"Loaded {len(self.sessions)} sessions from S3")
            except Exception as e:
                print(f"S3 session listing failed: {e}")
            return
        # --- END S3 MODE ---

        # LOCAL MODE: load from disk (original behavior)
        if not self.storage_path or not Path(self.storage_path).exists():
            return

        session_files = list(Path(self.storage_path).glob("session_*.json"))

        for session_file in session_files[:self.max_sessions]:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                session = self._deserialize_session(session_data)
                self.sessions[session.session_id] = session
            except Exception as e:
                print(f"Error loading session from {session_file}: {e}")
                session_file.unlink()

    def _deserialize_session(self, session_data: dict) -> ConversationSession:
        """Reconstruct a ConversationSession from a dict (shared by S3 and local loaders)."""
        context_data = session_data.get("context", {})
        allowed_context_fields = [
            "project_id", "filter_source_type", "recent_topics",
            "custom_instructions", "conversation_start_question"
        ]
        filtered_context = {k: v for k, v in context_data.items() if k in allowed_context_fields}
        if "recent_topics" not in filtered_context:
            filtered_context["recent_topics"] = []

        return ConversationSession(
            session_id=session_data["session_id"],
            created_at=session_data["created_at"],
            last_accessed=session_data["last_accessed"],
            messages=[
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    tokens=msg.get("tokens", 0),
                    metadata=msg.get("metadata"),
                )
                for msg in session_data["messages"]
            ],
            context=ConversationContext(**filtered_context),
            summaries=[
                ConversationSummary(
                    summary_text=s["summary_text"],
                    message_count=s["message_count"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    key_points=s["key_points"],
                )
                for s in session_data.get("summaries", [])
            ],
            total_tokens=session_data.get("total_tokens", 0),
            metadata=session_data.get("metadata", {}),
        )

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        current_time = time.time()
        old_sessions = []
        
        for session_id, session in list(self.sessions.items()):
            if current_time - session.last_accessed > max_age_hours * 3600:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            self.clear_session(session_id)
        
        return len(old_sessions)

# ==============================
# Global Memory Manager Instance
# ==============================

# Global memory manager instance
_memory_manager = None

def get_memory_manager(
    storage_path: Optional[str] = "./conversation_sessions",
    **kwargs
) -> MemoryManager:
    """Get or create global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            storage_path=storage_path,
            **kwargs
        )
    return _memory_manager

# ==============================
# Helper Functions
# ==============================

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text"""
    # Rough estimate: 1 token ≈ 4 characters for English
    return len(text) // 4

def create_session_from_query(
    query: str,
    project_id: Optional[int] = None,
    filter_source_type: Optional[str] = None
) -> str:
    """Helper to create a session from query"""
    memory_manager = get_memory_manager()
    return memory_manager.create_session(
        user_query=query,
        project_id=project_id,
        filter_source_type=filter_source_type
    )

