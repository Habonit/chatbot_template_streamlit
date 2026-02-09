"""Phase 05 Step 2: react_graph.py 프롬프트 캡처 테스트

_llm_node, _casual_node에서 actual_prompts 캡처,
invoke(), stream() 결과에 actual_prompts 전달 확인.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestLlmNodeActualPrompts:
    """_llm_node의 actual_prompts 캡처 테스트"""

    def test_llm_node_captures_system_prompt(self):
        """_llm_node가 actual_prompts에 system_prompt 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")

        # Mock LLM
        mock_response = MagicMock()
        mock_response.content = "Hello"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke.return_value = mock_response
        builder._llm_with_tools = mock_llm_with_tools

        state = {
            "messages": [HumanMessage(content="test")],
            "summary": "",
            "summary_history": [],
            "pdf_description": "",
            "graph_path": ["summary_node", "router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        result = builder._llm_node(state)
        assert "actual_prompts" in result
        assert "system_prompt" in result["actual_prompts"]
        assert "당신은 유용한 AI 어시스턴트입니다" in result["actual_prompts"]["system_prompt"]
        assert result["actual_prompts"]["user_messages_count"] >= 1


class TestCasualNodeActualPrompts:
    """_casual_node의 actual_prompts 캡처 테스트"""

    def test_casual_node_captures_casual_prompt(self):
        """_casual_node가 actual_prompts에 casual_prompt 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "안녕하세요!"
        mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        builder._llm = MagicMock()
        builder._llm.invoke.return_value = mock_response

        state = {
            "messages": [HumanMessage(content="안녕")],
            "summary_history": [],
            "graph_path": ["summary_node", "router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        result = builder._casual_node(state)
        assert "actual_prompts" in result
        assert "casual_prompt" in result["actual_prompts"]
        assert result["actual_prompts"]["user_messages_count"] == 1


class TestInvokeActualPrompts:
    """invoke() 결과에 actual_prompts 전달 테스트"""

    @patch("service.react_graph.ReactGraphBuilder.build")
    def test_invoke_returns_actual_prompts(self, mock_build):
        """invoke() 결과에 actual_prompts 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key", db_path=":memory:")

        mock_result = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="response"),
            ],
            "input_tokens": 10,
            "output_tokens": 5,
            "summary": "",
            "summary_history": [],
            "mode": "normal",
            "is_casual": False,
            "normal_turn_ids": [1],
            "graph_path": ["summary_node", "router_node", "llm_node"],
            "summary_triggered": False,
            "actual_prompts": {
                "system_prompt": "test prompt",
                "user_messages_count": 1,
                "context_turns": 0,
            },
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = mock_result
        builder._graph = mock_graph

        result = builder.invoke(
            user_input="test",
            session_id="session-1",
            turn_count=1,
        )

        assert "actual_prompts" in result
        assert result["actual_prompts"]["system_prompt"] == "test prompt"


class TestStreamActualPrompts:
    """stream() done metadata에 actual_prompts 전달 테스트"""

    @patch("service.react_graph.ReactGraphBuilder.build")
    def test_stream_done_contains_actual_prompts(self, mock_build):
        """stream() done metadata에 actual_prompts 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key", db_path=":memory:")

        # Mock stream — 빈 이벤트 + done
        mock_graph = MagicMock()
        mock_graph.stream.return_value = []  # 빈 스트림

        mock_state_values = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="response"),
            ],
            "input_tokens": 10,
            "output_tokens": 5,
            "summary": "",
            "summary_history": [],
            "mode": "normal",
            "is_casual": False,
            "normal_turn_ids": [1],
            "graph_path": ["summary_node", "router_node", "llm_node"],
            "summary_triggered": False,
            "actual_prompts": {
                "system_prompt": "test prompt",
                "user_messages_count": 1,
                "context_turns": 0,
            },
        }
        mock_state = MagicMock()
        mock_state.values = mock_state_values
        mock_graph.get_state.return_value = mock_state
        builder._graph = mock_graph

        events = list(builder.stream(
            user_input="test",
            session_id="session-1",
            turn_count=1,
        ))

        # 마지막 이벤트는 done
        done_event = events[-1]
        assert done_event["type"] == "done"
        assert "actual_prompts" in done_event["metadata"]
        assert done_event["metadata"]["actual_prompts"]["system_prompt"] == "test prompt"
