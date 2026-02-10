"""Phase 03-5: Thinking Mode 통합 테스트

app.py _create_graph_builder()가 thinking 파라미터를 전달하는지 검증
"""
import pytest
from unittest.mock import patch, MagicMock
import inspect


class TestCreateGraphBuilderThinking:
    """_create_graph_builder() thinking 파라미터 전달 테스트"""

    def test_create_graph_builder_passes_thinking_budget(self):
        """thinking_budget이 ReactGraphBuilder에 전달됨"""
        from app import _create_graph_builder

        src = inspect.getsource(_create_graph_builder)
        assert "thinking_budget" in src

    def test_create_graph_builder_passes_show_thoughts(self):
        """show_thoughts가 ReactGraphBuilder에 전달됨"""
        from app import _create_graph_builder

        src = inspect.getsource(_create_graph_builder)
        assert "show_thoughts" in src

    @patch("app.st")
    def test_create_graph_builder_with_thinking_settings(self, mock_st):
        """thinking 설정이 ReactGraphBuilder 생성자에 전달됨"""
        mock_st.session_state.chunks = []

        from app import _create_graph_builder

        settings = {
            "gemini_api_key": "test-key",
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 8192,
            "seed": None,
            "max_iterations": 5,
            "thinking_budget": 1024,
            "show_thoughts": True,
        }

        with patch("app.ReactGraphBuilder") as MockBuilder:
            _create_graph_builder(settings, MagicMock())
            call_kwargs = MockBuilder.call_args[1]
            assert call_kwargs["thinking_budget"] == 1024
            assert call_kwargs["show_thoughts"] is True

    @patch("app.st")
    def test_create_graph_builder_default_thinking(self, mock_st):
        """thinking 설정 없으면 기본값 사용"""
        mock_st.session_state.chunks = []

        from app import _create_graph_builder

        settings = {
            "gemini_api_key": "test-key",
        }

        with patch("app.ReactGraphBuilder") as MockBuilder:
            _create_graph_builder(settings, MagicMock())
            call_kwargs = MockBuilder.call_args[1]
            assert call_kwargs["thinking_budget"] == 0
            assert call_kwargs["show_thoughts"] is False
