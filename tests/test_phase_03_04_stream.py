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

        with patch.object(builder, "_stream_casual") as mock_casual:
            mock_casual.return_value = iter([
                {"type": "token", "content": "안녕!"},
                {"type": "done", "metadata": {}},
            ])
            gen = builder.stream("안녕", session_id="test")
            assert hasattr(gen, "__iter__")
            assert hasattr(gen, "__next__")


class TestStreamCasual:
    """_stream_casual 메서드 테스트"""

    def test_stream_casual_exists(self):
        """_stream_casual 메서드 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        assert hasattr(builder, "_stream_casual")

    def test_stream_casual_yields_tokens(self):
        """casual 스트리밍 → token + done"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk1 = MagicMock()
        mock_chunk1.content = "안녕"
        mock_chunk1.usage_metadata = None

        mock_chunk2 = MagicMock()
        mock_chunk2.content = "하세요!"
        mock_chunk2.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk1, mock_chunk2])):
            chunks = list(builder._stream_casual("안녕", "", [], []))

        token_chunks = [c for c in chunks if c["type"] == "token"]
        assert len(token_chunks) == 2
        assert token_chunks[0]["content"] == "안녕"
        assert token_chunks[1]["content"] == "하세요!"
        assert chunks[-1]["type"] == "done"

    def test_stream_casual_done_metadata(self):
        """done 이벤트 메타데이터 키 확인"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk = MagicMock()
        mock_chunk.content = "네!"
        mock_chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk])):
            chunks = list(builder._stream_casual("네", "요약", [{"a": 1}], [1, 2]))

        done = chunks[-1]
        assert done["type"] == "done"
        meta = done["metadata"]
        assert meta["text"] == "네!"
        assert meta["tool_history"] == []
        assert meta["is_casual"] is True
        assert meta["summary"] == "요약"
        assert meta["normal_turn_ids"] == [1, 2]
        assert meta["input_tokens"] == 5
        assert meta["output_tokens"] == 3

    def test_stream_casual_empty_content_skipped(self):
        """빈 content는 yield하지 않음"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk1 = MagicMock()
        mock_chunk1.content = ""
        mock_chunk1.usage_metadata = None

        mock_chunk2 = MagicMock()
        mock_chunk2.content = "응답"
        mock_chunk2.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk1, mock_chunk2])):
            chunks = list(builder._stream_casual("안녕", "", [], []))

        token_chunks = [c for c in chunks if c["type"] == "token"]
        assert len(token_chunks) == 1
        assert token_chunks[0]["content"] == "응답"


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
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        # Mock casual stream for simplicity
        mock_chunk = MagicMock()
        mock_chunk.content = "네!"
        mock_chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk])):
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
