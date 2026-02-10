"""Phase 03-4: 스트리밍 백엔드 테스트

stream(), _parse_message_chunk(), _stream_casual() 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Generator
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage


class TestStreamMethod:
    """stream() 메서드 테스트"""

    def test_stream_method_exists(self):
        """stream() 메서드 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        assert hasattr(builder, "stream")
        assert callable(builder.stream)

    def test_stream_returns_generator(self):
        """Generator 반환 확인"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        mock_state_values = {
            "messages": [AIMessage(content="안녕!")],
            "summary": "",
            "summary_history": [],
            "input_tokens": 5,
            "output_tokens": 3,
            "mode": "casual",
            "is_casual": True,
            "normal_turn_ids": [],
            "normal_turn_count": 0,
            "graph_path": ["summary_node", "router_node", "casual_node"],
            "summary_triggered": False,
        }
        mock_state = MagicMock()
        mock_state.values = mock_state_values

        with patch.object(builder._graph, "stream", return_value=iter([])), \
             patch.object(builder._graph, "get_state", return_value=mock_state):
            gen = builder.stream("안녕", session_id="test")
            assert hasattr(gen, "__iter__")
            assert hasattr(gen, "__next__")


class TestParseMessageChunk:
    """_parse_message_chunk 메서드 테스트"""

    def test_parse_message_chunk_exists(self):
        """_parse_message_chunk 메서드 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        assert hasattr(builder, "_parse_message_chunk")

    def test_parse_message_chunk_token(self):
        """AIMessageChunk 텍스트 → token 이벤트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        chunk = AIMessageChunk(content="Hello")
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "Hello"

    def test_parse_message_chunk_tool_call(self):
        """tool_call_chunks → tool_call 이벤트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [{"name": "web_search", "args": "", "id": "1", "index": 0}]

        buffer = []
        events = builder._parse_message_chunk(chunk, {}, buffer)

        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["name"] == "web_search"
        assert len(buffer) == 1

    def test_parse_message_chunk_tool_message(self):
        """ToolMessage → tool_result 이벤트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        chunk = ToolMessage(content="검색 결과입니다", name="web_search", tool_call_id="1")

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        assert events[0]["name"] == "web_search"
        assert events[0]["content"] == "검색 결과입니다"

    def test_parse_message_chunk_empty_content(self):
        """빈 AIMessageChunk는 이벤트 없음"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 0

    def test_parse_message_chunk_tool_result_truncated(self):
        """ToolMessage content가 500자 이상이면 잘림"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        long_content = "x" * 1000
        chunk = ToolMessage(content=long_content, name="test_tool", tool_call_id="1")

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events[0]["content"]) == 500


class TestStreamDoneMetadata:
    """stream done 이벤트 메타데이터 테스트"""

    def test_stream_done_metadata_keys(self):
        """done 이벤트에 필요한 키가 모두 있는지"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        mock_state_values = {
            "messages": [AIMessage(content="네!")],
            "summary": "",
            "summary_history": [],
            "input_tokens": 5,
            "output_tokens": 3,
            "mode": "casual",
            "is_casual": True,
            "normal_turn_ids": [],
            "normal_turn_count": 0,
            "graph_path": ["summary_node", "router_node", "casual_node"],
            "summary_triggered": False,
        }
        mock_state = MagicMock()
        mock_state.values = mock_state_values

        with patch.object(builder._graph, "stream", return_value=iter([])), \
             patch.object(builder._graph, "get_state", return_value=mock_state):
            chunks = list(builder.stream("안녕", session_id="test"))

        done = chunks[-1]
        assert done["type"] == "done"
        metadata = done["metadata"]

        required_keys = [
            "text", "tool_history", "model_used", "summary",
            "summary_history", "input_tokens", "output_tokens",
            "normal_turn_ids", "normal_turn_count",
        ]
        for key in required_keys:
            assert key in metadata, f"Missing key: {key}"
