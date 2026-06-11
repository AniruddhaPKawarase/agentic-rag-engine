import copy
from shared import session_tracker as st


FULL_SOURCE = {
    "s3_path": "ifieldsmart/spec/pdf123",
    "file_name": "300-13-00-Sanitary",
    "display_title": "SANITARY WASTE AND VENT PIPING",
    "download_url": "https://s3.amazonaws.com/x.pdf?X-Amz-Expires=3600",
    "pdf_name": "300-13-00-Sanitary",
    "drawing_name": "221300",
    "drawing_title": "SANITARY WASTE AND VENT PIPING",
    "page": 7,
    "text_excerpt": "1. Test for leaks...",
    "csi_division": "221300",
    "source_document_type": "specification",
    "bbox_pt": [115.22, 272.79, 542.85, 537.08],
    "bbox_px": [240.04, 568.33, 1130.95, 1118.93],
    "page_width_pt": 612.0,
    "page_height_pt": 792.0,
    "text_blocks": [{"text": "1. Test...", "bbox_pt": [1, 2, 3, 4]}],
    "parent_id": 105465,
    "fragment_count": 1,
    "doc_number": 1,
}


def test_slim_keeps_display_fields_drops_heavy():
    out = st._slim_source_documents([FULL_SOURCE])
    assert len(out) == 1
    slim = out[0]
    for k in st._SLIM_SOURCE_FIELDS:
        if k in FULL_SOURCE:
            assert slim[k] == FULL_SOURCE[k]
    for k in ("download_url", "text_excerpt", "bbox_pt", "bbox_px",
              "page_width_pt", "page_height_pt", "text_blocks",
              "parent_id", "fragment_count"):
        assert k not in slim


def test_slim_does_not_mutate_input():
    before = copy.deepcopy(FULL_SOURCE)
    st._slim_source_documents([FULL_SOURCE])
    assert FULL_SOURCE == before


def test_slim_handles_empty_and_missing_fields():
    assert st._slim_source_documents([]) == []
    assert st._slim_source_documents(None) == []
    partial = {"file_name": "x"}
    out = st._slim_source_documents([partial])
    assert out[0]["file_name"] == "x"
    assert "page" not in out[0]


def test_post_user_session_includes_turn_fields(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"data": {"sessionId": "sess_x"}}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _Resp()

    monkeypatch.setattr(st.requests, "post", _fake_post)

    st._post_user_session(
        user_id=1, project_id=1, agent_id="drawing-agent",
        s3_bucket_path="bucket/key.json",
        session_id="sess_x", turn_id="turn-uuid", title="My Title",
        question="What is X?", is_active=True,
        follow_up_questions=["Q1?"],
        source_documents=[{"file_name": "f"}],
    )
    body = captured["body"]
    assert body["sessionId"] == "sess_x"
    assert body["turnId"] == "turn-uuid"
    assert body["title"] == "My Title"
    assert body["question"] == "What is X?"
    assert body["isActive"] is True
    assert body["follow_up_questions"] == ["Q1?"]
    assert body["source_documents"] == [{"file_name": "f"}]
    assert body["userId"] == 1 and body["agent"] == "drawing-agent"


def test_post_user_session_minimal_still_works(monkeypatch):
    captured = {}
    class _Resp:
        status_code = 200
        def json(self): return {"data": {"sessionId": "s"}}
    monkeypatch.setattr(st.requests, "post",
                        lambda url, json=None, timeout=None: (captured.update(body=json) or _Resp()))
    st._post_user_session(user_id=1, project_id=1, agent_id="drawing-agent",
                          s3_bucket_path="b/k")
    assert "turnId" not in captured["body"]
    assert captured["body"]["s3BucketPath"] == "b/k"


def test_push_session_turn_writes_s3_full_and_posts_slim(monkeypatch):
    s3_calls = {}
    post_calls = {}

    monkeypatch.setattr(st, "_s3_bucket", lambda: "agentic-ai-production")
    monkeypatch.setattr(st, "_generate_presigned_get_url", lambda *a, **k: None)
    monkeypatch.setattr(
        st, "_write_s3_json",
        lambda *, bucket, key, payload: s3_calls.update(
            bucket=bucket, key=key, payload=payload) or True,
    )
    monkeypatch.setattr(
        st, "_post_user_session",
        lambda **kw: post_calls.update(kw) or "ifield-sid",
    )

    res = st.push_session_turn(
        user_id=1, project_id=1, agent_id="drawing-agent",
        session_id="sess_abc", turn_id="turn-uuid", title="T",
        question="Q?", follow_up_questions=["f1"],
        source_documents=[FULL_SOURCE],
    )

    s3_sources = s3_calls["payload"]["source_documents"]
    assert "text_blocks" in s3_sources[0] and "bbox_pt" in s3_sources[0]
    assert s3_calls["key"] == "sessions/1/1/drawing-agent/turn-uuid.json"

    assert "text_blocks" not in post_calls["source_documents"][0]
    assert post_calls["source_documents"][0]["file_name"] == FULL_SOURCE["file_name"]
    assert post_calls["session_id"] == "sess_abc"
    assert post_calls["turn_id"] == "turn-uuid"
    assert post_calls["s3_bucket_path"] == "agentic-ai-production/sessions/1/1/drawing-agent/turn-uuid.json"

    assert res["ifield_pushed"] is True
    assert res["s3_written"] is True


def test_push_session_turn_never_raises_on_failure(monkeypatch):
    monkeypatch.setattr(st, "_write_s3_json", lambda **kw: (_ for _ in ()).throw(RuntimeError("s3 down")))
    monkeypatch.setattr(st, "_post_user_session", lambda **kw: None)
    res = st.push_session_turn(
        user_id=1, project_id=1, agent_id="drawing-agent",
        session_id="s", turn_id="t", title="T", question="Q",
        follow_up_questions=[], source_documents=[],
    )
    assert res["ifield_pushed"] is False
    assert res["s3_written"] is False


def test_list_sessions_hits_list_endpoint(monkeypatch):
    captured = {}
    class _Resp:
        status_code = 200
        text = ""
        def json(self): return {"data": [{"sessionId": "s1", "title": "T1"}]}
    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url; captured["params"] = params
        return _Resp()
    monkeypatch.setattr(st.requests, "get", _fake_get)
    monkeypatch.setattr(st, "_userSession_url", lambda: "https://mongo.ifieldsmart.com/api/userSession")

    out = st.list_sessions(user_id=1, project_id=1, agent_id="drawing-agent")
    assert captured["url"] == "https://mongo.ifieldsmart.com/api/userSession/list"
    assert captured["params"] == {"userId": 1, "projectId": 1, "agent": "drawing-agent"}
    assert out["success"] is True
    assert out["data"][0]["sessionId"] == "s1"
    assert out["count"] == 1


def test_get_session_history_hits_history_endpoint(monkeypatch):
    captured = {}
    class _Resp:
        status_code = 200
        text = ""
        def json(self): return {"data": [{"turnId": "t1"}, {"turnId": "t2"}]}
    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return _Resp()
    monkeypatch.setattr(st.requests, "get", _fake_get)
    monkeypatch.setattr(st, "_userSession_url", lambda: "https://mongo.ifieldsmart.com/api/userSession")

    out = st.get_session_history(session_id="sess-9a8b")
    assert captured["url"] == "https://mongo.ifieldsmart.com/api/userSession/history/sess-9a8b"
    assert out["success"] is True
    assert out["count"] == 2


def test_get_session_history_handles_error(monkeypatch):
    def _boom(url, params=None, timeout=None):
        raise RuntimeError("net")
    monkeypatch.setattr(st.requests, "get", _boom)
    out = st.get_session_history(session_id="x")
    assert out["success"] is False
    assert out["data"] == []


def test_resolve_title_reuses_existing(monkeypatch):
    monkeypatch.setattr(
        st, "get_session_history",
        lambda *, session_id: {"success": True, "data": [
            {"turnId": "t1", "title": "Existing Title"},
            {"turnId": "t2", "title": "Existing Title"},
        ]},
    )
    title = st.resolve_session_title(session_id="s", question="new q")
    assert title == "Existing Title"


def test_resolve_title_generates_when_none(monkeypatch):
    monkeypatch.setattr(
        st, "get_session_history",
        lambda *, session_id: {"success": True, "data": []},
    )
    title = st.resolve_session_title(
        session_id="s", question="What are the testing regulations for drainage systems?")
    assert title.startswith("What are the testing regulations")
    assert len(title) <= 80


def test_resolve_title_falls_back_on_exception(monkeypatch):
    def _boom(*, session_id):
        raise RuntimeError("boom")
    monkeypatch.setattr(st, "get_session_history", _boom)
    title = st.resolve_session_title(session_id="s", question="Fallback question?")
    assert title.startswith("Fallback question")


def test_get_session_history_forwards_scope_params(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"data": []}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(st.requests, "get", _fake_get)
    monkeypatch.setattr(st, "_userSession_url", lambda: "https://mongo.ifieldsmart.com/api/userSession")

    st.get_session_history(session_id="sess-1", user_id=155, project_id=7325)
    assert captured["params"] == {"userId": 155, "projectId": 7325}

    st.get_session_history(session_id="sess-1")
    assert captured["params"] is None


def test_push_session_turn_skips_when_ids_missing(monkeypatch):
    called = {"s3": False, "post": False}
    monkeypatch.setattr(st, "_write_s3_json", lambda **kw: called.update(s3=True) or True)
    monkeypatch.setattr(st, "_post_user_session", lambda **kw: called.update(post=True) or "x")
    res = st.push_session_turn(
        user_id=1, project_id=1, agent_id="drawing-agent",
        session_id="", turn_id="", title="T", question="Q",
        follow_up_questions=[], source_documents=[],
    )
    assert res["ifield_pushed"] is False
    assert res["s3_written"] is False
    assert res["s3_bucket_path"] is None
    assert called == {"s3": False, "post": False}  # no I/O attempted


def test_push_session_turn_uses_presigned_url_as_s3bucketpath(monkeypatch):
    post_calls = {}
    monkeypatch.setattr(st, "_s3_bucket", lambda: "agentic-ai-production")
    monkeypatch.setattr(st, "_write_s3_json", lambda **kw: True)
    monkeypatch.setattr(st, "_generate_presigned_get_url",
                        lambda *a, **k: "https://signed.example/x?sig=abc")
    monkeypatch.setattr(st, "_post_user_session",
                        lambda **kw: post_calls.update(kw) or "sid")
    res = st.push_session_turn(
        user_id=1, project_id=1, agent_id="drawing-agent",
        session_id="s", turn_id="t", title="T", question="Q",
        follow_up_questions=[], source_documents=[],
    )
    assert post_calls["s3_bucket_path"] == "https://signed.example/x?sig=abc"
    assert post_calls["presigned_url"] == "https://signed.example/x?sig=abc"
    assert res["presigned_url"] == "https://signed.example/x?sig=abc"
