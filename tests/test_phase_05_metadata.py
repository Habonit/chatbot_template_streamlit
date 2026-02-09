"""Phase 05 Step 3: app.py 메타데이터 전달 테스트

handle_chat_message, handle_stream_message에서
actual_prompts를 Message에 전달하는지 검증.
"""
import pytest
from unittest.mock import patch, MagicMock
from domain.message import Message


class TestHandleChatMessageActualPrompts:
    """handle_chat_message가 actual_prompts를 Message에 전달하는지 검증"""

    @patch("app.st")
    @patch("app._create_graph_builder")
    def test_handle_chat_message_passes_actual_prompts(
        self, mock_create_builder, mock_st
    ):
        """handle_chat_message가 actual_prompts를 Message에 포함"""
        messages_list = []
        mock_session_state = MagicMock()
        mock_session_state.messages = messages_list
        mock_session_state.current_session = "test-session"
        mock_session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_session_state.summary = ""
        mock_session_state.summary_history = []
        mock_session_state.pdf_description = ""
        mock_session_state.chunks = []
        mock_session_state.normal_turn_ids = []
        mock_st.session_state = mock_session_state

        mock_builder = MagicMock()
        mock_builder.invoke.return_value = {
            "text": "응답",
            "tool_history": [],
            "tool_results": {},
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "model_used": "gemini-2.0-flash",
            "summary": "",
            "summary_history": [],
            "normal_turn_ids": [1],
            "mode": "normal",
            "graph_path": ["summary_node", "router_node", "llm_node"],
            "summary_triggered": False,
            "is_casual": False,
            "actual_prompts": {
                "system_prompt": "test system prompt",
                "user_messages_count": 1,
                "context_turns": 0,
            },
        }
        mock_create_builder.return_value = mock_builder

        from app import handle_chat_message

        settings = {"gemini_api_key": "test-key"}
        embed_repo = MagicMock()

        handle_chat_message("테스트", settings, embed_repo)

        # Message가 2개 추가됨 (user + assistant)
        assert len(messages_list) == 2
        assistant_msg = messages_list[1]
        assert isinstance(assistant_msg, Message)
        assert assistant_msg.actual_prompts["system_prompt"] == "test system prompt"


class TestHandleStreamMessageActualPrompts:
    """handle_stream_message가 actual_prompts를 Message에 전달하는지 검증"""

    @patch("app.st")
    @patch("app._create_graph_builder")
    def test_handle_stream_message_passes_actual_prompts(
        self, mock_create_builder, mock_st
    ):
        """handle_stream_message가 actual_prompts를 Message에 포함"""
        messages_list = []
        mock_session_state = MagicMock()
        mock_session_state.messages = messages_list
        mock_session_state.current_session = "test-session"
        mock_session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_session_state.summary = ""
        mock_session_state.summary_history = []
        mock_session_state.pdf_description = ""
        mock_session_state.chunks = []
        mock_session_state.normal_turn_ids = []
        mock_st.session_state = mock_session_state

        mock_builder = MagicMock()
        mock_builder.stream.return_value = iter([
            {"type": "token", "content": "응답"},
            {
                "type": "done",
                "metadata": {
                    "text": "응답",
                    "tool_history": [],
                    "tool_results": {},
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "model_used": "gemini-2.0-flash",
                    "summary": "",
                    "summary_history": [],
                    "normal_turn_ids": [1],
                    "mode": "normal",
                    "graph_path": ["summary_node", "router_node", "llm_node"],
                    "summary_triggered": False,
                    "is_casual": False,
                    "actual_prompts": {
                        "system_prompt": "stream system prompt",
                        "user_messages_count": 1,
                        "context_turns": 2,
                    },
                },
            },
        ])
        mock_create_builder.return_value = mock_builder

        from app import handle_stream_message

        settings = {"gemini_api_key": "test-key"}
        embed_repo = MagicMock()

        # consume generator
        list(handle_stream_message("테스트", settings, embed_repo))

        # Message가 2개 추가됨 (user + assistant)
        assert len(messages_list) == 2
        assistant_msg = messages_list[1]
        assert isinstance(assistant_msg, Message)
        assert assistant_msg.actual_prompts["system_prompt"] == "stream system prompt"
        assert assistant_msg.actual_prompts["context_turns"] == 2
