"""Phase 03-4: 통합 테스트

handle_stream_message, render_chat_tab on_stream 연동 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
import inspect


class TestHandleStreamMessage:
    """handle_stream_message 함수 테스트"""

    def test_handle_stream_message_exists(self):
        """handle_stream_message 함수 존재"""
        from app import handle_stream_message
        assert callable(handle_stream_message)

    @patch("app.st")
    def test_handle_stream_message_yields(self, mock_st):
        """Generator 동작"""
        from app import handle_stream_message
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.current_session = "test-session"
        mock_st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_st.session_state.messages = []
        mock_st.session_state.summary = ""
        mock_st.session_state.summary_history = []
        mock_st.session_state.pdf_description = ""
        mock_st.session_state.normal_turn_ids = []
        mock_st.session_state.chunks = []

        settings = {"gemini_api_key": "test-key"}
        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))

        gen = handle_stream_message("안녕", settings, embed_repo)
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

    @patch("app.st")
    def test_handle_stream_message_error_on_no_key(self, mock_st):
        """API 키 없을 때 에러 yield"""
        from app import handle_stream_message
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {}

        chunks = list(handle_stream_message("안녕", settings, embed_repo))
        assert any(c.get("type") == "token" for c in chunks)
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"].get("error") is True

    @patch("app.st")
    def test_handle_stream_message_error_on_token_limit(self, mock_st):
        """토큰 제한 초과 시 에러 yield"""
        from app import handle_stream_message, TOKEN_LIMIT
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.current_session = "test-session"
        mock_st.session_state.token_usage = {"input": 0, "output": 0, "total": TOKEN_LIMIT + 1}
        mock_st.session_state.messages = []

        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {"gemini_api_key": "test-key"}

        chunks = list(handle_stream_message("안녕", settings, embed_repo))
        assert any(c.get("type") == "token" for c in chunks)
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"].get("error") is True


class TestRenderChatTabWithOnStream:
    """render_chat_tab에 on_stream 연동 테스트"""

    def test_render_chat_tab_with_on_stream(self):
        """render_chat_tab 호출에 on_stream 포함 가능"""
        from component.chat_tab import render_chat_tab
        sig = inspect.signature(render_chat_tab)

        # on_stream과 use_streaming 모두 존재
        assert "on_stream" in sig.parameters
        assert "use_streaming" in sig.parameters

        # on_send는 여전히 필수
        assert "on_send" in sig.parameters

    def test_app_main_uses_on_stream(self):
        """app.py에서 render_chat_tab에 on_stream 전달 확인"""
        import ast
        with open("app.py", "r") as f:
            source = f.read()

        tree = ast.parse(source)
        # render_chat_tab 호출에서 on_stream 키워드 확인
        found_on_stream = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "render_chat_tab":
                    for kw in node.keywords:
                        if kw.arg == "on_stream":
                            found_on_stream = True
        assert found_on_stream, "render_chat_tab call should include on_stream parameter"
