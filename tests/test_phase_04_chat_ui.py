"""Phase 04 Step 4: Chat Tab UI 테스트

_mode_badge, _format_graph_path, _render_turn_metadata,
_handle_streaming_response metadata, render_chat_tab 렌더링 테스트
"""
import pytest
from unittest.mock import patch, MagicMock, call
from domain.message import Message


class TestModeBadge:
    """_mode_badge 함수 테스트"""

    def test_mode_badge_casual(self):
        """casual 모드 → 🟢 casual"""
        from component.chat_tab import _mode_badge

        assert _mode_badge("casual") == "🟢 casual"

    def test_mode_badge_normal(self):
        """normal 모드 → 🔵 normal"""
        from component.chat_tab import _mode_badge

        assert _mode_badge("normal") == "🔵 normal"

    def test_mode_badge_reasoning(self):
        """reasoning 모드 → 🟣 reasoning"""
        from component.chat_tab import _mode_badge

        assert _mode_badge("reasoning") == "🟣 reasoning"

    def test_mode_badge_unknown(self):
        """알 수 없는 모드 → ⚪ unknown_mode"""
        from component.chat_tab import _mode_badge

        assert _mode_badge("unknown_mode") == "⚪ unknown_mode"


class TestFormatGraphPath:
    """_format_graph_path 함수 테스트"""

    def test_format_graph_path_normal(self):
        """일반 경로 → 화살표 형식"""
        from component.chat_tab import _format_graph_path

        result = _format_graph_path(["summary_node", "llm_node"])
        assert result == "summary_node → llm_node → END"

    def test_format_graph_path_with_tools(self):
        """도구 포함 경로 → 모든 노드 포함"""
        from component.chat_tab import _format_graph_path

        result = _format_graph_path(
            ["summary_node", "llm_node", "tool_node", "llm_node"]
        )
        assert result == "summary_node → llm_node → tool_node → llm_node → END"

    def test_format_graph_path_casual(self):
        """캐주얼 경로 → casual_bypass → END"""
        from component.chat_tab import _format_graph_path

        result = _format_graph_path(["casual_bypass"])
        assert result == "casual_bypass → END"

    def test_format_graph_path_empty(self):
        """빈 경로 → N/A"""
        from component.chat_tab import _format_graph_path

        result = _format_graph_path([])
        assert result == "N/A"


class TestRenderTurnMetadata:
    """_render_turn_metadata 함수 테스트"""

    @patch("component.chat_tab.st")
    def test_render_turn_metadata_basic(self, mock_st):
        """기본 assistant 메시지 → 📊 실행 상세 expander 호출"""
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        from component.chat_tab import _render_turn_metadata

        msg = Message(
            turn_id=1,
            role="assistant",
            content="Hello",
            mode="normal",
            model_used="gemini-2.0-flash",
        )
        _render_turn_metadata(msg)

        mock_st.expander.assert_called_with("📊 실행 상세", expanded=False)

    @patch("component.chat_tab.st")
    def test_render_turn_metadata_with_tools(self, mock_st):
        """function_calls 포함 → 도구 이름 표시"""
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        from component.chat_tab import _render_turn_metadata

        msg = Message(
            turn_id=2,
            role="assistant",
            content="Result",
            mode="normal",
            function_calls=[
                {"name": "web_search"},
                {"name": "calculator"},
            ],
        )
        _render_turn_metadata(msg)

        # Check that st.markdown was called with tool names
        markdown_calls = [
            str(c) for c in mock_st.markdown.call_args_list
        ]
        tool_call_found = any("web_search" in c and "calculator" in c for c in markdown_calls)
        assert tool_call_found, f"Tool names not found in markdown calls: {markdown_calls}"

    @patch("component.chat_tab.st")
    def test_render_turn_metadata_with_thinking(self, mock_st):
        """thinking_budget > 0 → 🧠 Thinking 표시"""
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        from component.chat_tab import _render_turn_metadata

        msg = Message(
            turn_id=3,
            role="assistant",
            content="Deep answer",
            mode="reasoning",
            thinking_budget=4096,
        )
        _render_turn_metadata(msg)

        markdown_calls = [str(c) for c in mock_st.markdown.call_args_list]
        thinking_found = any("Thinking" in c and "4096" in c for c in markdown_calls)
        assert thinking_found, f"Thinking budget not found in markdown calls: {markdown_calls}"

    @patch("component.chat_tab.st")
    def test_render_turn_metadata_with_summary(self, mock_st):
        """summary_triggered=True → 📋 Summary 표시"""
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        from component.chat_tab import _render_turn_metadata

        msg = Message(
            turn_id=3,
            role="assistant",
            content="Answer",
            mode="normal",
            summary_triggered=True,
        )
        _render_turn_metadata(msg)

        markdown_calls = [str(c) for c in mock_st.markdown.call_args_list]
        summary_found = any("Summary" in c and "트리거됨" in c for c in markdown_calls)
        assert summary_found, f"Summary trigger not found in markdown calls: {markdown_calls}"

    @patch("component.chat_tab.st")
    def test_render_turn_metadata_casual(self, mock_st):
        """mode=casual → 🟢 casual badge 표시"""
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_col = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        from component.chat_tab import _render_turn_metadata

        msg = Message(
            turn_id=1,
            role="assistant",
            content="Hi!",
            mode="casual",
            is_casual=True,
        )
        _render_turn_metadata(msg)

        # col2.markdown should be called with casual badge
        col2 = mock_st.columns.return_value[1]
        col2_markdown_calls = [str(c) for c in col2.markdown.call_args_list]
        casual_found = any("casual" in c for c in col2_markdown_calls)
        assert casual_found, f"Casual badge not found in col2 markdown calls: {col2_markdown_calls}"


class TestHandleStreamingResponseMetadata:
    """_handle_streaming_response Phase 04 메타데이터 테스트"""

    @patch("component.chat_tab.st")
    def test_streaming_response_includes_metadata(self, mock_st):
        """반환 dict에 mode, graph_path, summary_triggered, is_casual 포함"""
        mock_st.empty.return_value = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "token", "content": "Answer"}
            yield {
                "type": "done",
                "metadata": {
                    "text": "Answer",
                    "model_used": "gemini-2.0-flash",
                    "mode": "reasoning",
                    "graph_path": ["summary_node", "llm_node"],
                    "summary_triggered": True,
                    "is_casual": False,
                },
            }

        result = _handle_streaming_response(mock_stream, "test")

        assert result["mode"] == "reasoning"
        assert result["graph_path"] == ["summary_node", "llm_node"]
        assert result["summary_triggered"] is True
        assert result["is_casual"] is False


class TestChatTabRendering:
    """render_chat_tab에서 _render_turn_metadata 호출 확인"""

    @patch("component.chat_tab._render_turn_metadata")
    @patch("component.chat_tab.st")
    def test_render_chat_tab_calls_metadata(self, mock_st, mock_render_meta):
        """assistant 메시지에 대해 _render_turn_metadata가 호출됨"""
        # Setup mock context managers
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.chat_input.return_value = None  # No user input

        from component.chat_tab import render_chat_tab

        user_msg = Message(turn_id=1, role="user", content="Hello")
        assistant_msg = Message(
            turn_id=1,
            role="assistant",
            content="Hi there!",
            mode="normal",
            graph_path=["summary_node", "llm_node"],
        )

        render_chat_tab(
            on_send=MagicMock(),
            messages=[user_msg, assistant_msg],
        )

        # _render_turn_metadata should be called once for the assistant message
        mock_render_meta.assert_called_once_with(assistant_msg)
