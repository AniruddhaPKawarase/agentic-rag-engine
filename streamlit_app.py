"""
Streamlit UI for the Unified RAG Agent.

Connects to the sandbox VM APIs at http://54.197.189.113:8000/rag/
Covers: chat, search modes, session management, source docs, debug tools.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
import sseclient
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://54.197.189.113:8000/rag"

KNOWN_PROJECTS = {
    7166: "Project 7166",
    7201: "Project 7201",
    7212: "Project 7212",
    7222: "Project 7222",
    7223: "Project 7223",
    7277: "Project 7277",
    7292: "Project 7292",
    7298: "Project 7298 (Granville Hotel)",
    2361: "Project 2361",
}

SEARCH_MODES = {
    "rag": "RAG (Project Data)",
    "web": "Web Search",
    "hybrid": "Hybrid (RAG + Web)",
}

ENGINE_OPTIONS = {
    "auto": "Auto (Agentic-first + Fallback)",
    "agentic": "Agentic Only (MongoDB / GPT-4.1)",
    "traditional": "Traditional Only (FAISS / GPT-4o)",
}

SOURCE_TYPE_OPTIONS = {
    None: "All Sources",
    "drawing": "Drawings Only",
    "specification": "Specifications Only",
}

CONFIDENCE_COLORS = {
    "high": "#22c55e",
    "medium": "#f59e0b",
    "low": "#ef4444",
}


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = st.session_state.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def api_post(path: str, payload: dict) -> dict:
    """POST request to the RAG API."""
    try:
        resp = requests.post(
            _url(path), json=payload, headers=_headers(), timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (120s). The query may be too complex."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Cannot connect to {BASE_URL}. Is the VM running?"}
    except requests.exceptions.HTTPError as exc:
        return {"success": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def api_get(path: str, params: Optional[dict] = None) -> dict:
    """GET request to the RAG API."""
    try:
        resp = requests.get(
            _url(path), params=params, headers=_headers(), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": f"Cannot connect to {BASE_URL}."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def api_delete(path: str) -> dict:
    """DELETE request to the RAG API."""
    try:
        resp = requests.delete(_url(path), headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def api_stream(path: str, payload: dict):
    """SSE streaming POST — yields parsed event dicts."""
    try:
        resp = requests.post(
            _url(path), json=payload, headers=_headers(),
            stream=True, timeout=120,
        )
        resp.raise_for_status()
        client = sseclient.SSEClient(resp)
        for event in client.events():
            if event.data == "[DONE]":
                break
            try:
                yield json.loads(event.data)
            except json.JSONDecodeError:
                yield {"delta": event.data}
    except Exception as exc:
        yield {"error": str(exc)}


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "messages": [],
        "session_id": None,
        "project_id": 7298,
        "search_mode": "rag",
        "engine": "auto",
        "filter_source_type": None,
        "filter_drawing_name": "",
        "set_id": None,
        "streaming_enabled": False,
        "api_key": "",
        "show_sources": True,
        "show_debug": False,
        "last_response": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_confidence_badge(confidence: str, score: Optional[float] = None) -> str:
    color = CONFIDENCE_COLORS.get(confidence, "#6b7280")
    label = confidence.upper()
    if score is not None:
        label += f" ({score:.0%})"
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{label}</span>'


def render_engine_badge(engine: str, fallback: bool = False) -> str:
    colors = {"agentic": "#8b5cf6", "traditional": "#3b82f6"}
    color = colors.get(engine, "#6b7280")
    label = engine.capitalize()
    if fallback:
        label += " (fallback)"
        color = "#f97316"
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{label}</span>'


def render_source_documents(sources: list[dict]) -> None:
    """Render source documents as expandable cards."""
    if not sources:
        return
    st.markdown("##### Source Documents")
    for i, src in enumerate(sources, 1):
        display = src.get("display_title") or src.get("file_name") or src.get("s3_path", "Unknown")
        s3_path = src.get("s3_path", "")
        download_url = src.get("download_url")

        with st.expander(f"{i}. {display}", expanded=False):
            if s3_path:
                st.code(s3_path, language=None)
            if download_url:
                st.markdown(f"[Download PDF]({download_url})")
            file_name = src.get("file_name", "")
            if file_name and file_name != display:
                st.caption(f"File: {file_name}")


def render_web_sources(web_sources: list[dict]) -> None:
    """Render web search sources as clickable links."""
    if not web_sources:
        return
    st.markdown("##### Web Sources")
    for src in web_sources:
        title = src.get("title", "Link")
        url = src.get("url", "")
        if url:
            st.markdown(f"- [{title}]({url})")
        else:
            st.markdown(f"- {title}")


def render_metrics_row(response: dict) -> None:
    """Render a row of metric cards for the response."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        elapsed = response.get("processing_time_ms", 0)
        st.metric("Response Time", f"{elapsed:,} ms")
    with col2:
        cost = response.get("cost_usd") or 0.0
        debug = response.get("debug_info") or {}
        if not cost and debug:
            cost = debug.get("agentic_cost_usd", 0.0)
        st.metric("Cost", f"${cost:.4f}")
    with col3:
        count = response.get("retrieval_count") or response.get("s3_path_count", 0)
        st.metric("Sources Found", count)
    with col4:
        steps = 0
        if debug:
            steps = debug.get("agentic_steps", 0)
        st.metric("Agent Steps", steps)


def render_token_usage(response: dict) -> None:
    """Render token usage breakdown."""
    token_usage = response.get("token_usage") or response.get("token_tracking")
    if not token_usage:
        return
    with st.expander("Token Usage", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Prompt", f"{token_usage.get('prompt_tokens', 0):,}")
        with cols[1]:
            st.metric("Completion", f"{token_usage.get('completion_tokens', 0):,}")
        with cols[2]:
            st.metric("Total", f"{token_usage.get('total_tokens', 0):,}")


def render_follow_up_questions(questions: list[str]) -> None:
    """Render follow-up question suggestions as clickable buttons."""
    if not questions:
        return
    st.markdown("##### Suggested Follow-ups")
    for q in questions:
        if st.button(q, key=f"followup_{hash(q)}"):
            st.session_state["followup_query"] = q
            st.rerun()


def render_session_stats(stats: dict) -> None:
    """Render session engine usage stats."""
    engine_usage = stats.get("engine_usage", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Agentic Queries", engine_usage.get("agentic", 0))
    with col2:
        st.metric("Traditional Queries", engine_usage.get("traditional", 0))
    with col3:
        st.metric("Fallbacks", engine_usage.get("fallback", 0))
    with col4:
        st.metric("Total Cost", f"${stats.get('total_cost_usd', 0):.4f}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
        st.title("VCS RAG Agent")
        st.caption(f"Connected to: `{BASE_URL}`")

        # --- Connection ---
        with st.expander("Connection", expanded=False):
            st.session_state["api_key"] = st.text_input(
                "API Key (optional)", value=st.session_state.get("api_key", ""),
                type="password",
            )
            custom_url = st.text_input("Custom Base URL", value="")
            if custom_url:
                global BASE_URL
                BASE_URL = custom_url.rstrip("/")

            if st.button("Test Connection"):
                with st.spinner("Testing..."):
                    result = api_get("/health")
                if result.get("status") == "healthy":
                    st.success("Connected! Both engines healthy.")
                elif result.get("error"):
                    st.error(result["error"])
                else:
                    st.warning(f"Response: {json.dumps(result, indent=2)}")

        st.divider()

        # --- Project ---
        st.subheader("Project")
        project_options = list(KNOWN_PROJECTS.keys())
        current_idx = project_options.index(st.session_state["project_id"]) if st.session_state["project_id"] in project_options else 0
        st.session_state["project_id"] = st.selectbox(
            "Select Project",
            options=project_options,
            format_func=lambda x: f"{x} — {KNOWN_PROJECTS[x]}",
            index=current_idx,
        )
        custom_pid = st.number_input("Or enter custom Project ID", min_value=1, max_value=999999, value=0, step=1)
        if custom_pid > 0:
            st.session_state["project_id"] = custom_pid

        st.divider()

        # --- Search Settings ---
        st.subheader("Search Settings")
        st.session_state["search_mode"] = st.radio(
            "Search Mode",
            options=list(SEARCH_MODES.keys()),
            format_func=lambda x: SEARCH_MODES[x],
            index=list(SEARCH_MODES.keys()).index(st.session_state["search_mode"]),
        )

        engine_val = st.session_state["engine"]
        engine_key = engine_val if engine_val in ENGINE_OPTIONS else "auto"
        st.session_state["engine"] = st.radio(
            "Engine",
            options=list(ENGINE_OPTIONS.keys()),
            format_func=lambda x: ENGINE_OPTIONS[x],
            index=list(ENGINE_OPTIONS.keys()).index(engine_key),
            help="Auto = Agentic first, falls back to Traditional on low confidence.",
        )

        st.session_state["filter_source_type"] = st.selectbox(
            "Filter Source Type",
            options=list(SOURCE_TYPE_OPTIONS.keys()),
            format_func=lambda x: SOURCE_TYPE_OPTIONS[x],
        )

        st.session_state["filter_drawing_name"] = st.text_input(
            "Filter by Drawing Name",
            value=st.session_state.get("filter_drawing_name", ""),
            placeholder="e.g. M-101A",
        )

        set_id_val = st.number_input("Set ID (optional)", min_value=0, value=0, step=1)
        st.session_state["set_id"] = set_id_val if set_id_val > 0 else None

        st.divider()

        # --- Display Settings ---
        st.subheader("Display")
        st.session_state["show_sources"] = st.toggle("Show Sources", value=st.session_state.get("show_sources", True))
        st.session_state["show_debug"] = st.toggle("Show Debug Info", value=st.session_state.get("show_debug", False))
        st.session_state["streaming_enabled"] = st.toggle("Enable Streaming", value=st.session_state.get("streaming_enabled", False))

        st.divider()

        # --- Session ---
        st.subheader("Session")
        if st.session_state.get("session_id"):
            st.success(f"Active: `{st.session_state['session_id'][:12]}...`")
        else:
            st.info("No active session")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("New Session", use_container_width=True):
                with st.spinner("Creating..."):
                    result = api_post("/sessions/create", {"project_id": st.session_state["project_id"]})
                if result.get("success"):
                    st.session_state["session_id"] = result["session_id"]
                    st.session_state["messages"] = []
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed"))

        with col_s2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["last_response"] = None
                st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Page
# ---------------------------------------------------------------------------

def page_chat() -> None:
    st.header("Chat with Construction Documents")

    # Check for follow-up query injection
    followup = st.session_state.pop("followup_query", None)

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]
                # Badges
                badges = ""
                if meta.get("confidence"):
                    badges += render_confidence_badge(meta["confidence"], meta.get("confidence_score"))
                    badges += "  "
                if meta.get("engine_used"):
                    badges += render_engine_badge(meta["engine_used"], meta.get("fallback_used", False))
                    badges += "  "
                model = meta.get("model_used", "")
                if model:
                    badges += f'<span style="background:#374151;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;">{model}</span>'
                if badges:
                    st.markdown(badges, unsafe_allow_html=True)

                # Metrics row
                render_metrics_row(meta)

                # Sources
                if st.session_state["show_sources"]:
                    render_source_documents(meta.get("source_documents", []))
                    render_web_sources(meta.get("web_sources", []))

                # Token usage
                render_token_usage(meta)

                # Follow-up questions
                render_follow_up_questions(meta.get("follow_up_questions", []))

                # Debug
                if st.session_state["show_debug"] and meta.get("debug_info"):
                    with st.expander("Debug Info", expanded=False):
                        st.json(meta["debug_info"])

    # Chat input
    user_query = followup or st.chat_input("Ask about your construction project...")

    if user_query:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build request payload
        engine_val = st.session_state["engine"]
        payload = {
            "query": user_query,
            "project_id": st.session_state["project_id"],
            "search_mode": st.session_state["search_mode"],
            "engine": engine_val if engine_val != "auto" else None,
            "session_id": st.session_state.get("session_id"),
            "set_id": st.session_state.get("set_id"),
            "filter_source_type": st.session_state.get("filter_source_type"),
            "filter_drawing_name": st.session_state.get("filter_drawing_name") or None,
            "generate_document": True,
        }
        # Clean None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Build conversation history from recent messages (last 10)
        history = []
        for msg in st.session_state["messages"][-10:]:
            if msg["role"] in ("user", "assistant"):
                history.append({"role": msg["role"], "content": msg["content"]})
        if len(history) > 1:
            payload["conversation_history"] = history[:-1]  # Exclude the current query

        with st.chat_message("assistant"):
            if st.session_state.get("streaming_enabled"):
                # --- Streaming mode ---
                answer_placeholder = st.empty()
                full_answer = ""
                final_data = {}
                with st.spinner("Connecting to stream..."):
                    for chunk in api_stream("/query/stream", payload):
                        if "error" in chunk:
                            st.error(chunk["error"])
                            break
                        if chunk.get("type") == "token":
                            full_answer += chunk.get("delta", "")
                            answer_placeholder.markdown(full_answer + "...")
                        elif chunk.get("type") == "done":
                            full_answer = chunk.get("answer", full_answer)
                            final_data = chunk
                        elif "answer" in chunk:
                            # Full result as single event (fallback streaming)
                            full_answer = chunk.get("answer", "")
                            final_data = chunk

                answer_placeholder.markdown(full_answer)
                response = final_data if final_data else {"answer": full_answer}
            else:
                # --- Blocking mode ---
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    response = api_post("/query", payload)
                    client_elapsed = int((time.time() - start_time) * 1000)

                if response.get("error") and not response.get("answer"):
                    st.error(f"Error: {response['error']}")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"Error: {response['error']}",
                    })
                    return

                answer = response.get("answer", "No answer received.")
                st.markdown(answer)

            # Store response
            answer = response.get("answer", "No answer received.")
            confidence = response.get("confidence") or response.get("agentic_confidence", "")
            metadata = {
                "confidence": confidence,
                "confidence_score": response.get("confidence_score"),
                "engine_used": response.get("engine_used", ""),
                "fallback_used": response.get("fallback_used", False),
                "model_used": response.get("model_used", ""),
                "processing_time_ms": response.get("processing_time_ms", 0),
                "cost_usd": response.get("cost_usd", 0),
                "retrieval_count": response.get("retrieval_count", 0),
                "s3_path_count": response.get("s3_path_count", 0),
                "source_documents": response.get("source_documents", []),
                "web_sources": response.get("web_sources", []),
                "follow_up_questions": response.get("follow_up_questions", []),
                "token_usage": response.get("token_usage"),
                "token_tracking": response.get("token_tracking"),
                "debug_info": response.get("debug_info"),
                "search_mode": response.get("search_mode", ""),
                "session_stats": response.get("session_stats"),
            }

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata,
            })
            st.session_state["last_response"] = response

            # Update session_id if returned
            if response.get("session_id"):
                st.session_state["session_id"] = response["session_id"]

            # Badges
            badges = ""
            if confidence:
                badges += render_confidence_badge(confidence, response.get("confidence_score"))
                badges += "  "
            if response.get("engine_used"):
                badges += render_engine_badge(response["engine_used"], response.get("fallback_used", False))
                badges += "  "
            model = response.get("model_used", "")
            if model:
                badges += f'<span style="background:#374151;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em;">{model}</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)

            render_metrics_row(response)

            if st.session_state["show_sources"]:
                render_source_documents(response.get("source_documents", []))
                render_web_sources(response.get("web_sources", []))

            render_token_usage(response)
            render_follow_up_questions(response.get("follow_up_questions", []))

            if st.session_state["show_debug"] and response.get("debug_info"):
                with st.expander("Debug Info", expanded=False):
                    st.json(response["debug_info"])


# ---------------------------------------------------------------------------
# Sessions Page
# ---------------------------------------------------------------------------

def page_sessions() -> None:
    st.header("Session Management")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Create New Session", use_container_width=True):
            with st.spinner("Creating..."):
                result = api_post("/sessions/create", {"project_id": st.session_state["project_id"]})
            if result.get("success"):
                st.session_state["session_id"] = result["session_id"]
                st.success(f"Created: `{result['session_id']}`")
            else:
                st.error(result.get("error", "Failed"))

    with col2:
        if st.button("Refresh Sessions", use_container_width=True):
            st.rerun()

    with col3:
        if st.session_state.get("session_id"):
            if st.button("Delete Active Session", type="primary", use_container_width=True):
                with st.spinner("Deleting..."):
                    result = api_delete(f"/sessions/{st.session_state['session_id']}")
                if result.get("success"):
                    st.session_state["session_id"] = None
                    st.session_state["messages"] = []
                    st.success("Session deleted.")
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed"))

    st.divider()

    # List sessions
    with st.spinner("Loading sessions..."):
        sessions_resp = api_get("/sessions")

    if sessions_resp.get("error"):
        st.error(sessions_resp["error"])
        return

    sessions = sessions_resp.get("sessions", [])
    if not sessions:
        st.info("No sessions found. Create one to get started.")
        return

    st.subheader(f"Active Sessions ({len(sessions)})")
    for sess in sessions:
        sid = sess if isinstance(sess, str) else sess.get("session_id", str(sess))
        is_active = sid == st.session_state.get("session_id")
        label = f"{'-> ' if is_active else ''}{sid[:20]}..."

        with st.expander(label, expanded=is_active):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Use This Session", key=f"use_{sid}"):
                    st.session_state["session_id"] = sid
                    st.session_state["messages"] = []
                    st.rerun()
            with col_b:
                if st.button("View Stats", key=f"stats_{sid}"):
                    with st.spinner("Loading stats..."):
                        stats = api_get(f"/sessions/{sid}/stats")
                    if stats.get("success") or stats.get("engine_usage"):
                        render_session_stats(stats)
                    else:
                        st.json(stats)
            with col_c:
                if st.button("View History", key=f"hist_{sid}"):
                    with st.spinner("Loading history..."):
                        history = api_get(f"/sessions/{sid}/conversation")
                    conversation = history.get("conversation", [])
                    if conversation:
                        for msg in conversation:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            st.chat_message(role).markdown(content[:500])
                    else:
                        st.info("No messages in this session.")

            # Delete button
            if st.button("Delete", key=f"del_{sid}", type="secondary"):
                with st.spinner("Deleting..."):
                    result = api_delete(f"/sessions/{sid}")
                if result.get("success"):
                    if sid == st.session_state.get("session_id"):
                        st.session_state["session_id"] = None
                    st.success("Deleted.")
                    st.rerun()


# ---------------------------------------------------------------------------
# Document Pinning Page
# ---------------------------------------------------------------------------

def page_pinning() -> None:
    st.header("Document Pinning")
    st.caption("Pin specific documents to a session for scoped FAISS search (Traditional engine).")

    sid = st.session_state.get("session_id")
    if not sid:
        st.warning("Create or select a session first (see sidebar or Sessions page).")
        return

    st.info(f"Active Session: `{sid[:20]}...`")

    st.subheader("Pin Documents")
    doc_ids_input = st.text_area(
        "Document IDs (one per line)",
        placeholder="e.g.\nifieldsmart/proj7298/Drawings/abc123\nifieldsmart/proj7298/Drawings/def456",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Pin Documents", use_container_width=True):
            doc_ids = [d.strip() for d in doc_ids_input.strip().split("\n") if d.strip()]
            if not doc_ids:
                st.warning("Enter at least one document ID.")
            else:
                with st.spinner("Pinning..."):
                    result = api_post(f"/sessions/{sid}/pin-document", {"document_ids": doc_ids})
                if result.get("success"):
                    st.success(f"Pinned {len(doc_ids)} document(s).")
                else:
                    st.error(result.get("error", "Failed"))

    with col2:
        if st.button("Unpin All Documents", use_container_width=True):
            doc_ids = [d.strip() for d in doc_ids_input.strip().split("\n") if d.strip()]
            if not doc_ids:
                st.warning("Enter document IDs to unpin.")
            else:
                with st.spinner("Unpinning..."):
                    result = api_delete(f"/sessions/{sid}/pin-document")
                if result.get("success"):
                    st.success("Documents unpinned.")
                else:
                    st.error(result.get("error", "Failed"))


# ---------------------------------------------------------------------------
# Quick Query Page
# ---------------------------------------------------------------------------

def page_quick_query() -> None:
    st.header("Quick Query")
    st.caption("Simplified query — returns only answer, sources, and confidence. No session tracking.")

    query = st.text_area("Your question", height=80, placeholder="e.g. What HVAC equipment is specified?")

    if st.button("Ask", type="primary", disabled=not query.strip()):
        payload = {
            "query": query.strip(),
            "project_id": st.session_state["project_id"],
        }
        with st.spinner("Querying..."):
            result = api_post("/quick-query", payload)

        if result.get("error") and not result.get("answer"):
            st.error(result["error"])
            return

        st.markdown("### Answer")
        st.markdown(result.get("answer", "No answer."))

        col1, col2 = st.columns(2)
        with col1:
            conf = result.get("confidence", "unknown")
            st.markdown(f"**Confidence:** {render_confidence_badge(conf)}", unsafe_allow_html=True)
        with col2:
            engine = result.get("engine_used", "unknown")
            st.markdown(f"**Engine:** {render_engine_badge(engine)}", unsafe_allow_html=True)

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for i, src in enumerate(sources, 1):
                name = src.get("name", src.get("s3_path", "Unknown"))
                sheet = src.get("sheet_number", "")
                st.markdown(f"{i}. **{name}** {f'(Sheet: {sheet})' if sheet else ''}")


# ---------------------------------------------------------------------------
# Web Search Page
# ---------------------------------------------------------------------------

def page_web_search() -> None:
    st.header("Web Search")
    st.caption("Search the web for construction industry information using the Traditional engine's OpenAI web search tool.")

    query = st.text_area("Search query", height=80, placeholder="e.g. Latest ASHRAE 90.1 energy code requirements")

    if st.button("Search", type="primary", disabled=not query.strip()):
        payload = {
            "query": query.strip(),
            "project_id": st.session_state["project_id"],
        }
        with st.spinner("Searching the web..."):
            result = api_post("/web-search", payload)

        if result.get("error") and not result.get("success"):
            st.error(result["error"])
            return

        inner = result.get("result", result)
        answer = inner.get("answer") or inner.get("web_answer", "No results.")
        st.markdown("### Answer")
        st.markdown(answer)

        web_sources = inner.get("web_sources", inner.get("sources", []))
        render_web_sources(web_sources)


# ---------------------------------------------------------------------------
# Debug / Diagnostics Page
# ---------------------------------------------------------------------------

def page_debug() -> None:
    st.header("Debug & Diagnostics")

    tab1, tab2, tab3, tab4 = st.tabs(["Health", "Config", "Test Retrieve", "Debug Pipeline"])

    with tab1:
        if st.button("Check Health"):
            with st.spinner("Checking..."):
                result = api_get("/health")
            if result.get("status") == "healthy":
                st.success("All engines healthy!")
            st.json(result)

    with tab2:
        if st.button("Load Config"):
            with st.spinner("Loading..."):
                result = api_get("/config")
            st.json(result)

    with tab3:
        st.subheader("Test FAISS Retrieval")
        st.caption("Test raw FAISS vector search without running the LLM generation pipeline.")
        test_query = st.text_input("Test query", value="HVAC equipment")
        test_pid = st.number_input("Project ID", value=st.session_state["project_id"], min_value=1)
        test_top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

        if st.button("Run Test Retrieve"):
            with st.spinner("Retrieving..."):
                result = api_get("/test-retrieve", params={
                    "query": test_query,
                    "project_id": test_pid,
                    "top_k": test_top_k,
                })
            if result.get("success"):
                st.success(f"Found {result.get('results_count', 0)} results")
                for i, chunk in enumerate(result.get("results", []), 1):
                    with st.expander(f"Result {i} — Score: {chunk.get('similarity', chunk.get('score', 'N/A'))}"):
                        st.text(chunk.get("text", "")[:500])
                        meta_keys = ["drawing_id", "pdf_name", "s3_path", "page", "source_type"]
                        meta_display = {k: chunk.get(k) for k in meta_keys if chunk.get(k)}
                        if meta_display:
                            st.json(meta_display)
            else:
                st.error(result.get("error", "Failed"))

    with tab4:
        if st.button("Run Debug Pipeline"):
            with st.spinner("Loading..."):
                result = api_get("/debug-pipeline")
            st.json(result)


# ---------------------------------------------------------------------------
# Full Response Inspector
# ---------------------------------------------------------------------------

def page_inspector() -> None:
    st.header("Response Inspector")
    st.caption("View the raw JSON response from the last query.")

    last = st.session_state.get("last_response")
    if last:
        st.json(last)
    else:
        st.info("No response yet. Send a query from the Chat page first.")


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="VCS Unified RAG Agent",
        page_icon="https://img.icons8.com/fluency/96/artificial-intelligence.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stChatMessage { max-width: 100%; }
        .stMetric { text-align: center; }
        div[data-testid="stMetricValue"] { font-size: 1.2rem; }
        section[data-testid="stSidebar"] > div { padding-top: 1rem; }
        .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    init_state()
    render_sidebar()

    # Navigation
    page = st.radio(
        "Navigate",
        options=["Chat", "Quick Query", "Web Search", "Sessions", "Document Pinning", "Debug", "Inspector"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Chat":
        page_chat()
    elif page == "Quick Query":
        page_quick_query()
    elif page == "Web Search":
        page_web_search()
    elif page == "Sessions":
        page_sessions()
    elif page == "Document Pinning":
        page_pinning()
    elif page == "Debug":
        page_debug()
    elif page == "Inspector":
        page_inspector()


if __name__ == "__main__":
    main()
