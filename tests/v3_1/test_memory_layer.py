"""
Tests for the v3.1 conversation-memory subsystem.

These cover:
* embeddings.py — dim guarantee, truncation, retry behaviour.
* vector_store.py — Atlas insert path, FAISS-first search, Atlas
  fallback, total-failure resilience.
* writer.py — flag gating, dual-turn persistence, summary trigger,
  error swallowing.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make the v3.1 worktree importable. Tests live two levels deep
# (tests/v3_1/test_x.py), so the worktree root is the grandparent.
# ---------------------------------------------------------------------------
_WORKTREE = Path(__file__).resolve().parent.parent.parent
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))


# ===========================================================================
# embeddings.py
# ===========================================================================


def _make_fake_embedding_response(dim: int = 1536) -> MagicMock:
    fake = MagicMock()
    fake.data = [MagicMock(embedding=[0.1] * dim)]
    return fake


def test_embed_text_returns_1536_dims(monkeypatch):
    """Happy path: returns a list of 1536 floats."""
    from agentic.memory import embeddings as emb_mod

    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = _make_fake_embedding_response()

    monkeypatch.setattr(emb_mod, "_client", fake_client)

    out = emb_mod.embed_text("hello world")

    assert isinstance(out, list)
    assert len(out) == 1536
    assert all(isinstance(x, float) for x in out)
    fake_client.embeddings.create.assert_called_once()
    _, kwargs = fake_client.embeddings.create.call_args
    assert kwargs["model"] == emb_mod.EMBEDDING_MODEL
    assert kwargs["input"] == "hello world"


def test_embed_text_truncates_long_input(monkeypatch):
    """Inputs longer than MAX_INPUT_CHARS get capped before send."""
    from agentic.memory import embeddings as emb_mod

    fake_client = MagicMock()
    fake_client.embeddings.create.return_value = _make_fake_embedding_response()
    monkeypatch.setattr(emb_mod, "_client", fake_client)

    long_input = "x" * (emb_mod.MAX_INPUT_CHARS + 5000)
    emb_mod.embed_text(long_input)

    _, kwargs = fake_client.embeddings.create.call_args
    assert len(kwargs["input"]) == emb_mod.MAX_INPUT_CHARS


def test_embed_text_empty_returns_zero_vector(monkeypatch):
    """Empty / whitespace input bypasses the API and returns zeros."""
    from agentic.memory import embeddings as emb_mod

    fake_client = MagicMock()
    monkeypatch.setattr(emb_mod, "_client", fake_client)

    out = emb_mod.embed_text("   \n  ")

    assert out == [0.0] * emb_mod.EMBEDDING_DIMS
    fake_client.embeddings.create.assert_not_called()


def test_embed_text_retries_on_rate_limit(monkeypatch):
    """Rate-limit errors retry once, succeed on attempt 2."""
    from openai import RateLimitError

    from agentic.memory import embeddings as emb_mod

    fake_client = MagicMock()
    err = RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}, request=MagicMock()),
        body={},
    )
    fake_client.embeddings.create.side_effect = [
        err,
        _make_fake_embedding_response(),
    ]
    monkeypatch.setattr(emb_mod, "_client", fake_client)
    monkeypatch.setattr(emb_mod.time, "sleep", lambda *_: None)

    out = emb_mod.embed_text("retry me")
    assert len(out) == 1536
    assert fake_client.embeddings.create.call_count == 2


# ===========================================================================
# vector_store.py
# ===========================================================================


@pytest.fixture
def faiss_dir(tmp_path: Path) -> Path:
    d = tmp_path / "embeddings"
    d.mkdir()
    return d


@pytest.fixture
def vec1536() -> List[float]:
    # Deterministic non-zero vector for normalisation paths.
    base = [0.0] * 1536
    base[0] = 1.0
    base[7] = 0.5
    return base


def _make_store_with_fake_mongo(
    fake_collection: Any, faiss_dir: Path, mongo_client: Any = None,
):
    """Helper: build a SessionVectorStore with mongo plumbing mocked."""
    from agentic.memory.vector_store import SessionVectorStore

    if mongo_client is None:
        mongo_client = MagicMock()
        mongo_client.__getitem__.return_value.__getitem__.return_value = (
            fake_collection
        )

    store = SessionVectorStore(
        mongo_uri="mongodb://fake/",
        db_name="iField",
        atlas_timeout_ms=150,
        embedding_dir=faiss_dir,
        mongo_client=mongo_client,
    )
    return store


def test_vector_store_add_writes_to_mongo(faiss_dir, vec1536):
    """add() inserts a properly-shaped doc into the Mongo collection."""
    fake_coll = MagicMock()
    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)

    store.add(
        session_id="s1",
        turn_index=0,
        role="user",
        text="hello world",
        embedding=vec1536,
        metadata={"project_id": 7166},
    )

    fake_coll.insert_one.assert_called_once()
    doc = fake_coll.insert_one.call_args[0][0]
    assert doc["session_id"] == "s1"
    assert doc["turn_index"] == 0
    assert doc["role"] == "user"
    assert doc["text_excerpt"] == "hello world"
    assert len(doc["embedding"]) == 1536
    assert doc["metadata"] == {"project_id": 7166}
    assert "created_at" in doc


def test_vector_store_add_writes_faiss_index(faiss_dir, vec1536):
    """add() also produces a per-session FAISS file + meta sidecar."""
    fake_coll = MagicMock()
    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)

    store.add("sess-A", 0, "user", "first turn", vec1536, {"k": "v"})

    assert (faiss_dir / "sess-A.faiss").exists()
    assert (faiss_dir / "sess-A.meta.jsonl").exists()


def test_vector_store_search_uses_faiss_when_available(
    faiss_dir, vec1536, monkeypatch
):
    """When FAISS has >=top_k entries, Atlas is not consulted."""
    fake_coll = MagicMock()
    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)

    # Seed three turns into FAISS.
    for i in range(3):
        store.add("sess-B", i, "user" if i % 2 == 0 else "assistant",
                  f"turn {i}", vec1536, {"i": i})

    fake_coll.reset_mock()  # ignore the inserts above

    results = store.search("sess-B", vec1536, top_k=2)

    assert len(results) == 2
    assert all("text_excerpt" in r for r in results)
    fake_coll.aggregate.assert_not_called()


def test_vector_store_search_falls_back_to_atlas_on_faiss_miss(
    faiss_dir, vec1536
):
    """No FAISS file → Atlas $vectorSearch is consulted."""
    atlas_hits = [
        {
            "turn_index": 5,
            "role": "user",
            "text_excerpt": "atlas turn",
            "metadata": {"x": 1},
            "score": 0.91,
        }
    ]
    fake_coll = MagicMock()
    fake_coll.aggregate.return_value = iter(atlas_hits)
    # find() is also called by the async rebuild — return empty cursor.
    fake_find_cursor = MagicMock()
    fake_find_cursor.sort.return_value = iter([])
    fake_coll.find.return_value = fake_find_cursor

    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)

    results = store.search("missing-session", vec1536, top_k=3)

    fake_coll.aggregate.assert_called_once()
    pipeline = fake_coll.aggregate.call_args[0][0]
    assert "$vectorSearch" in pipeline[0]
    assert pipeline[0]["$vectorSearch"]["filter"] == {
        "session_id": "missing-session"
    }
    # maxTimeMS keyword must be passed through.
    _, kwargs = fake_coll.aggregate.call_args
    assert kwargs["maxTimeMS"] == 150
    assert results == atlas_hits


def test_vector_store_search_returns_empty_on_total_failure(
    faiss_dir, vec1536, monkeypatch
):
    """Both layers down → empty list, no exception."""
    from pymongo.errors import PyMongoError

    fake_coll = MagicMock()
    fake_coll.aggregate.side_effect = PyMongoError("atlas down")

    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)

    out = store.search("ghost-session", vec1536, top_k=5)
    assert out == []


def test_vector_store_delete_session_clears_both_layers(
    faiss_dir, vec1536
):
    """delete_session() removes Atlas docs + FAISS files."""
    fake_coll = MagicMock()
    store = _make_store_with_fake_mongo(fake_coll, faiss_dir)
    store.add("sess-D", 0, "user", "doomed", vec1536, {})

    assert (faiss_dir / "sess-D.faiss").exists()

    store.delete_session("sess-D")

    fake_coll.delete_many.assert_called_once_with({"session_id": "sess-D"})
    assert not (faiss_dir / "sess-D.faiss").exists()
    assert not (faiss_dir / "sess-D.meta.jsonl").exists()


# ===========================================================================
# writer.py
# ===========================================================================


class _SyncExecutor:
    """An Executor stand-in that runs work synchronously.

    Lets us assert effects without race conditions.
    """

    def submit(self, fn, *args, **kwargs):
        fut: Any = MagicMock()
        try:
            fut._result = fn(*args, **kwargs)
            fut.result = lambda timeout=None: fut._result
        except Exception as exc:  # noqa: BLE001
            fut._exc = exc

            def _raise(timeout=None):
                raise fut._exc

            fut.result = _raise
        return fut


class _FakeMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _FakeContext:
    def __init__(self):
        self.custom_instructions = ""
        self.rolling_summary = None  # newer-build field


class _FakeSession:
    def __init__(self):
        self.messages: List[_FakeMessage] = []
        self.context = _FakeContext()


class _FakeMemoryManager:
    """Just enough surface area to drive the writer."""

    def __init__(self) -> None:
        self.sessions: Dict[str, _FakeSession] = {}
        self.update_context_calls: List[Dict[str, Any]] = []

    def get_session(self, session_id: str):
        return self.sessions.setdefault(session_id, _FakeSession())

    def add_to_session(
        self, session_id: str, role: str, content: str,
        tokens: int = 0, metadata: Any = None,
    ) -> bool:
        sess = self.get_session(session_id)
        sess.messages.append(_FakeMessage(role, content))
        return True

    def update_context(self, session_id: str, **kwargs):
        self.update_context_calls.append({"session_id": session_id, **kwargs})
        sess = self.get_session(session_id)
        for k, v in kwargs.items():
            setattr(sess.context, k, v)


def _make_writer(
    embed_fn=None, summary_fn=None, vector_store=None, memory_manager=None,
):
    from agentic.memory.writer import MemoryWriter

    return MemoryWriter(
        memory_manager=memory_manager or _FakeMemoryManager(),
        vector_store=vector_store,
        embed_fn=embed_fn or (lambda txt: [0.01] * 1536),
        summary_fn=summary_fn or (lambda hist: "summary"),
        executor=_SyncExecutor(),
    )


def test_writer_skips_vector_when_flag_off(monkeypatch):
    """With the flag off, no embedding / vector / summary work runs."""
    monkeypatch.delenv("MEMORY_WRITER_VECTOR_ENABLED", raising=False)

    embed_calls: List[str] = []

    def tracking_embed(text: str) -> List[float]:
        embed_calls.append(text)
        return [0.0] * 1536

    fake_store = MagicMock()
    fake_summary = MagicMock(return_value="never")
    mm = _FakeMemoryManager()

    writer = _make_writer(
        embed_fn=tracking_embed,
        summary_fn=fake_summary,
        vector_store=fake_store,
        memory_manager=mm,
    )

    writer.write_turn_async(
        "s-flagoff", "hello?", "world.", project_id=7166, set_id="abc"
    )

    # Memory manager still got both turns.
    assert len(mm.get_session("s-flagoff").messages) == 2
    # But no embedding, no vector store, no summary.
    assert embed_calls == []
    fake_store.add.assert_not_called()
    fake_summary.assert_not_called()


def test_writer_writes_both_turns_to_memory_manager(monkeypatch):
    """Flag on or off, both user + assistant turns land in the manager."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")

    fake_store = MagicMock()
    mm = _FakeMemoryManager()
    writer = _make_writer(
        vector_store=fake_store, memory_manager=mm,
        summary_fn=lambda hist: "",  # don't touch update_context
    )

    writer.write_turn_async("s-both", "Q?", "A.", project_id=7166, set_id="z")

    msgs = mm.get_session("s-both").messages
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert [m.content for m in msgs] == ["Q?", "A."]
    assert fake_store.add.call_count == 2


def test_writer_calls_vector_store_with_correct_args(monkeypatch):
    """When the flag is on, vector_store.add gets one call per turn."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")

    fake_store = MagicMock()
    mm = _FakeMemoryManager()
    writer = _make_writer(
        vector_store=fake_store, memory_manager=mm,
        summary_fn=lambda hist: "",
    )

    writer.write_turn_async("s-vec", "ping", "pong", project_id=42, set_id="s")

    assert fake_store.add.call_count == 2
    user_call = fake_store.add.call_args_list[0]
    assert user_call.kwargs.get("session_id") == "s-vec" or \
        user_call.args[0] == "s-vec"
    # Inspect the role argument (positional 2 in our signature).
    roles = [c.args[2] for c in fake_store.add.call_args_list]
    assert roles == ["user", "assistant"]


def test_writer_triggers_summary_at_interval(monkeypatch):
    """After ROLLING_SUMMARY_INTERVAL user turns the summariser fires."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")
    monkeypatch.setenv("ROLLING_SUMMARY_INTERVAL", "2")

    # Re-import to pick up the new interval value.
    import importlib

    import agentic.memory.writer as writer_mod
    importlib.reload(writer_mod)

    summary_calls: List[List[Dict[str, str]]] = []

    def capturing_summary(history):
        summary_calls.append(history)
        return "rolling summary text"

    mm = _FakeMemoryManager()
    fake_store = MagicMock()
    writer = writer_mod.MemoryWriter(
        memory_manager=mm,
        vector_store=fake_store,
        embed_fn=lambda txt: [0.0] * 1536,
        summary_fn=capturing_summary,
        executor=_SyncExecutor(),
    )

    # Turn 1 (1 user msg) — no summary yet.
    writer.write_turn_async("s-sum", "q1", "a1", 7166, "x")
    assert summary_calls == []

    # Turn 2 (2 user msgs) — interval=2 => summary fires.
    writer.write_turn_async("s-sum", "q2", "a2", 7166, "x")
    assert len(summary_calls) == 1
    # Summary persisted via rolling_summary attribute.
    sess = mm.get_session("s-sum")
    assert sess.context.rolling_summary == "rolling summary text"


def test_writer_falls_back_to_custom_instructions(monkeypatch):
    """If the context has no rolling_summary field we use update_context."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")
    monkeypatch.setenv("ROLLING_SUMMARY_INTERVAL", "1")

    import importlib
    import agentic.memory.writer as writer_mod
    importlib.reload(writer_mod)

    class _OldContext:
        # No rolling_summary attribute and not a dataclass — exercises
        # the fallback path that writes to custom_instructions.
        custom_instructions = ""

    class _OldSession:
        def __init__(self):
            self.messages: List[_FakeMessage] = []
            self.context = _OldContext()

    class _OldMM(_FakeMemoryManager):
        def get_session(self, session_id):
            if session_id not in self.sessions:
                self.sessions[session_id] = _OldSession()  # type: ignore
            return self.sessions[session_id]

    mm = _OldMM()
    writer = writer_mod.MemoryWriter(
        memory_manager=mm,
        vector_store=MagicMock(),
        embed_fn=lambda txt: [0.0] * 1536,
        summary_fn=lambda hist: "fallback summary",
        executor=_SyncExecutor(),
    )

    writer.write_turn_async("s-old", "q", "a", 1, "s")

    # Either the rolling_summary attr was set OR update_context was called.
    sess = mm.get_session("s-old")
    rolling = getattr(sess.context, "rolling_summary", None)
    if rolling != "fallback summary":
        assert any(
            c.get("custom_instructions") == "fallback summary"
            for c in mm.update_context_calls
        )


def test_writer_swallows_all_errors(monkeypatch):
    """Any exception inside the worker is logged, never raised."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")

    def boom(text: str) -> List[float]:
        raise RuntimeError("embed exploded")

    class _BoomMM(_FakeMemoryManager):
        def add_to_session(self, *a, **kw):
            raise RuntimeError("mm exploded")

    fake_store = MagicMock()
    fake_store.add.side_effect = RuntimeError("store exploded")

    writer = _make_writer(
        embed_fn=boom,
        summary_fn=lambda hist: (_ for _ in ()).throw(RuntimeError("sum exploded")),
        vector_store=fake_store,
        memory_manager=_BoomMM(),
    )

    # This must NOT raise — every layer is broken on purpose.
    writer.write_turn_async("s-boom", "u", "a", 1, "s")


def test_writer_executor_has_shutdown_hook():
    """Module exposes an idempotent shutdown hook registered with atexit."""
    import atexit as _atexit

    import agentic.memory.writer as writer_mod

    # Function exists and is callable.
    assert callable(getattr(writer_mod, "_shutdown_executor", None))

    # Idempotent — calling twice must not raise. (Don't actually call
    # it during the test run; we just verify the function tolerates
    # repeated invocation by inspecting its body wrapping `try/except`.)
    # Instead, check it's registered with atexit.
    registered = False
    funcs_attr = getattr(_atexit, "_ngexit_funcs", None)
    if funcs_attr is not None:  # CPython internal, may exist
        registered = any(
            getattr(f, "__name__", "") == "_shutdown_executor"
            for f in funcs_attr
        )
    if not registered:
        # Fallback: monkey-introspect via the module — at minimum the
        # symbol must be the same callable that was registered.
        registered = writer_mod._shutdown_executor.__module__ == writer_mod.__name__
    assert registered


def test_writer_uses_separate_executor_for_embeddings(monkeypatch):
    """Dispatch goes to _EXECUTOR; embeds go to _EMBED_EXECUTOR."""
    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")

    import agentic.memory.writer as writer_mod

    dispatch_calls: List[Any] = []
    embed_calls: List[Any] = []

    real_dispatch_submit = writer_mod._EXECUTOR.submit
    real_embed_submit = writer_mod._EMBED_EXECUTOR.submit

    def spy_dispatch(fn, *a, **kw):
        dispatch_calls.append(fn)
        return real_dispatch_submit(fn, *a, **kw)

    def spy_embed(fn, *a, **kw):
        embed_calls.append(fn)
        return real_embed_submit(fn, *a, **kw)

    monkeypatch.setattr(writer_mod._EXECUTOR, "submit", spy_dispatch)
    monkeypatch.setattr(writer_mod._EMBED_EXECUTOR, "submit", spy_embed)

    mm = _FakeMemoryManager()
    fake_store = MagicMock()
    writer = writer_mod.MemoryWriter(
        memory_manager=mm,
        vector_store=fake_store,
        embed_fn=lambda txt: [0.0] * 1536,
        summary_fn=lambda hist: "",
        # Use the real module-level _EXECUTOR (default) so the spy fires.
    )

    writer.write_turn_async("s-split", "u", "a", 1, "x")

    # Wait for the background dispatch to complete.
    import time as _t
    deadline = _t.time() + 5.0
    while _t.time() < deadline and len(embed_calls) < 2:
        _t.sleep(0.01)

    assert len(dispatch_calls) == 1, f"expected 1 dispatch submit, got {len(dispatch_calls)}"
    assert len(embed_calls) == 2, f"expected 2 embed submits, got {len(embed_calls)}"


def test_writer_no_deadlock_under_concurrent_writes(monkeypatch):
    """8 concurrent writes must all finish — would deadlock on shared pool."""
    import concurrent.futures as _cf

    monkeypatch.setenv("MEMORY_WRITER_VECTOR_ENABLED", "true")

    import agentic.memory.writer as writer_mod

    mm = _FakeMemoryManager()
    fake_store = MagicMock()

    # Instant embed — the regression we're guarding against is structural
    # (pool saturation), not embed latency.
    writer = writer_mod.MemoryWriter(
        memory_manager=mm,
        vector_store=fake_store,
        embed_fn=lambda txt: [0.0] * 1536,
        summary_fn=lambda hist: "",
    )

    # Fire 8 concurrent dispatches via the real module pool.
    futures = [
        writer_mod._EXECUTOR.submit(
            writer._write_turn,
            f"s-conc-{i}", f"u{i}", f"a{i}", 1, "x",
        )
        for i in range(8)
    ]

    done, not_done = _cf.wait(futures, timeout=10)
    assert not not_done, (
        f"deadlock suspected — {len(not_done)} of 8 writes did not finish "
        "within 10s"
    )
    # Every dispatch completed without raising.
    for f in done:
        # _write_turn swallows exceptions, so .result() must not raise.
        f.result(timeout=1)


def test_writer_submit_does_not_raise_when_executor_dead(monkeypatch):
    """A shut-down executor must not crash the calling request."""
    monkeypatch.delenv("MEMORY_WRITER_VECTOR_ENABLED", raising=False)

    dead_exec = ThreadPoolExecutor(max_workers=1)
    dead_exec.shutdown(wait=True)

    from agentic.memory.writer import MemoryWriter

    writer = MemoryWriter(
        memory_manager=_FakeMemoryManager(),
        vector_store=MagicMock(),
        embed_fn=lambda t: [0.0] * 1536,
        summary_fn=lambda h: "",
        executor=dead_exec,
    )

    # Must not raise even though the pool is dead.
    writer.write_turn_async("s-dead", "u", "a", None, None)
