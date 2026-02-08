"""Phase 03-4: 스트리밍 UI 테스트

_handle_streaming_response(), render_chat_tab 시그니처 변경 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
import inspect


class TestRenderChatTabSignature:
    """render_chat_tab 시그니처 변경 테스트"""

    def test_render_chat_tab_accepts_on_stream(self):
        """on_stream 파라미터 허용"""
        from component.chat_tab import render_chat_tab

        sig = inspect.signature(render_chat_tab)
        assert "on_stream" in sig.parameters

    def test_render_chat_tab_accepts_use_streaming(self):
        """use_streaming 파라미터 허용"""
        from component.chat_tab import render_chat_tab

        sig = inspect.signature(render_chat_tab)
        assert "use_streaming" in sig.parameters

    def test_render_chat_tab_default_use_streaming_true(self):
        """use_streaming 기본값은 True"""
        from component.chat_tab import render_chat_tab

        sig = inspect.signature(render_chat_tab)
        assert sig.parameters["use_streaming"].default is True

    def test_render_chat_tab_default_on_stream_none(self):
        """on_stream 기본값은 None"""
        from component.chat_tab import render_chat_tab

        sig = inspect.signature(render_chat_tab)
        assert sig.parameters["on_stream"].default is None


class TestHandleStreamingResponse:
    """_handle_streaming_response 함수 테스트"""

    def test_handle_streaming_response_exists(self):
        """_handle_streaming_response 함수 존재"""
        from component.chat_tab import _handle_streaming_response
        assert callable(_handle_streaming_response)

    @patch("component.chat_tab.st")
    def test_handle_streaming_response_returns_dict(self, mock_st):
        """반환값 dict 확인"""
        mock_st.empty.return_value = MagicMock()

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "token", "content": "Hello"}
            yield {"type": "done", "metadata": {"text": "Hello", "model_used": "test"}}

        result = _handle_streaming_response(mock_stream, "test")
        assert isinstance(result, dict)
        assert "text" in result

    @patch("component.chat_tab.st")
    def test_handle_streaming_response_collects_tokens(self, mock_st):
        """token 청크 수집"""
        mock_st.empty.return_value = MagicMock()

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "token", "content": "Hello "}
            yield {"type": "token", "content": "World"}
            yield {"type": "done", "metadata": {"text": "Hello World"}}

        result = _handle_streaming_response(mock_stream, "test")
        assert result["text"] == "Hello World"

    @patch("component.chat_tab.st")
    def test_handle_streaming_response_collects_tool_calls(self, mock_st):
        """tool_call 수집"""
        mock_st.empty.return_value = MagicMock()

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "tool_call", "name": "web_search"}
            yield {"type": "tool_result", "name": "web_search", "content": "결과"}
            yield {"type": "token", "content": "답변"}
            yield {"type": "done", "metadata": {"text": "답변", "tool_results": {"web_search": "결과"}}}

        result = _handle_streaming_response(mock_stream, "test")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "web_search"
        assert result["tool_calls"][0]["result"] == "결과"

    @patch("component.chat_tab.st")
    def test_handle_streaming_response_handles_done(self, mock_st):
        """done 이벤트 처리"""
        mock_st.empty.return_value = MagicMock()

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "token", "content": "응답"}
            yield {
                "type": "done",
                "metadata": {
                    "text": "응답",
                    "model_used": "gemini-2.0-flash",
                    "summary": "요약",
                    "summary_history": [],
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "normal_turn_ids": [1],
                    "normal_turn_count": 1,
                    "tool_results": {},
                }
            }

        result = _handle_streaming_response(mock_stream, "test")
        assert result["model_used"] == "gemini-2.0-flash"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["normal_turn_ids"] == [1]

    @patch("component.chat_tab.st")
    def test_handle_streaming_response_cursor_removed(self, mock_st):
        """완료 후 커서(▌) 제거 확인"""
        mock_placeholder = MagicMock()
        mock_status = MagicMock()
        mock_thought = MagicMock()
        mock_st.empty.side_effect = [mock_placeholder, mock_status, mock_thought]

        from component.chat_tab import _handle_streaming_response

        def mock_stream(user_input):
            yield {"type": "token", "content": "응답"}
            yield {"type": "done", "metadata": {"text": "응답"}}

        _handle_streaming_response(mock_stream, "test")

        # 마지막 호출은 커서 없이 최종 텍스트
        last_call = mock_placeholder.markdown.call_args_list[-1]
        assert "▌" not in last_call[0][0]
