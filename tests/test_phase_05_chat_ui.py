"""Phase 05 Step 5: Chat Tab UI 교육 팁 통합 테스트

_render_turn_metadata의 교육 팁 호출,
_handle_streaming_response의 actual_prompts 전달,
_render_welcome_state의 streaming education 표시.
"""
import pytest
from unittest.mock import patch, MagicMock
from domain.message import Message


class TestRenderTurnMetadataEducation:
    """_render_turn_metadata 교육 팁 통합 테스트"""

    @patch("component.chat_tab.st")
    @patch("component.chat_tab.get_prompt_education")
    def test_calls_get_prompt_education(self, mock_prompt_edu, mock_st):
        """_render_turn_metadata가 get_prompt_education 호출"""
        from component.chat_tab import _render_turn_metadata

        mock_prompt_edu.return_value = {}
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        msg = Message(
            turn_id=1, role="assistant", content="test",
            actual_prompts={"system_prompt": "test", "user_messages_count": 1, "context_turns": 0},
        )
        _render_turn_metadata(msg)

        mock_prompt_edu.assert_called_once_with(msg.actual_prompts)

    @patch("component.chat_tab.st")
    @patch("component.chat_tab.get_tool_education")
    def test_calls_get_tool_education(self, mock_tool_edu, mock_st):
        """_render_turn_metadata가 get_tool_education 호출"""
        from component.chat_tab import _render_turn_metadata

        mock_tool_edu.return_value = {}
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        msg = Message(
            turn_id=1, role="assistant", content="test",
            function_calls=[{"name": "web_search"}],
        )
        _render_turn_metadata(msg)

        mock_tool_edu.assert_called_once_with(["web_search"])


class TestHandleStreamingResponseActualPrompts:
    """_handle_streaming_response의 actual_prompts 전달 테스트"""

    @patch("component.chat_tab.st")
    def test_returns_actual_prompts_from_metadata(self, mock_st):
        """_handle_streaming_response가 actual_prompts를 반환"""
        from component.chat_tab import _handle_streaming_response

        mock_st.empty.return_value = MagicMock()

        actual_prompts_data = {
            "system_prompt": "test prompt",
            "user_messages_count": 1,
            "context_turns": 2,
        }

        def mock_stream(user_input):
            yield {"type": "token", "content": "Hello"}
            yield {
                "type": "done",
                "metadata": {
                    "model_used": "gemini-2.0-flash",
                    "tool_results": {},
                    "summary": "",
                    "summary_history": [],
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "normal_turn_ids": [],
                    "normal_turn_count": 0,
                    "thought_process": "",
                    "mode": "normal",
                    "graph_path": ["llm_node"],
                    "summary_triggered": False,
                    "is_casual": False,
                    "actual_prompts": actual_prompts_data,
                },
            }

        result = _handle_streaming_response(mock_stream, "test")
        assert "actual_prompts" in result
        assert result["actual_prompts"]["system_prompt"] == "test prompt"


class TestRenderWelcomeStateStreaming:
    """_render_welcome_state 스트리밍 교육 테스트"""

    @patch("component.chat_tab.st")
    @patch("component.chat_tab.get_streaming_education")
    def test_welcome_state_shows_streaming_education(self, mock_stream_edu, mock_st):
        """_render_welcome_state가 streaming education 표시"""
        from component.chat_tab import _render_welcome_state

        mock_stream_edu.return_value = {
            "title": "스트리밍 응답",
            "explanation": "스트리밍은 ...",
            "terms": [{"term": "TTFT", "desc": "..."}],
        }

        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        _render_welcome_state()

        mock_stream_edu.assert_called_once()
