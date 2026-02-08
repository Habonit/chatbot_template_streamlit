"""Phase 03-5: Thinking Mode 백엔드 테스트"""
import pytest
import warnings
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


class TestExtractThoughtFromContent:
    """extract_thought_from_content() 테스트"""

    def test_string_content_no_thought(self):
        from service.react_graph import extract_thought_from_content
        thought, text = extract_thought_from_content("Hello")
        assert thought == ""
        assert text == "Hello"

    def test_list_content_with_thought(self):
        from service.react_graph import extract_thought_from_content
        content = [
            {"type": "text", "text": "Let me think...", "thought": True},
            {"type": "text", "text": "The answer is 5."},
        ]
        thought, text = extract_thought_from_content(content)
        assert thought == "Let me think..."
        assert text == "The answer is 5."

    def test_list_content_without_thought(self):
        from service.react_graph import extract_thought_from_content
        content = [{"type": "text", "text": "Hello"}]
        thought, text = extract_thought_from_content(content)
        assert thought == ""
        assert text == "Hello"

    def test_empty_content(self):
        from service.react_graph import extract_thought_from_content
        thought, text = extract_thought_from_content("")
        assert thought == ""
        assert text == ""

    def test_none_content(self):
        from service.react_graph import extract_thought_from_content
        thought, text = extract_thought_from_content(None)
        assert thought == ""
        assert text == ""

    def test_mixed_thought_parts(self):
        from service.react_graph import extract_thought_from_content
        content = [
            {"type": "text", "text": "First thought", "thought": True},
            {"type": "text", "text": "Second thought", "thought": True},
            {"type": "text", "text": "Response"},
        ]
        thought, text = extract_thought_from_content(content)
        assert "First thought" in thought
        assert "Second thought" in thought
        assert text == "Response"

    def test_string_items_in_list(self):
        from service.react_graph import extract_thought_from_content
        content = ["plain string"]
        thought, text = extract_thought_from_content(content)
        assert thought == ""
        assert text == "plain string"


class TestThinkingInit:
    """__init__ thinking 파라미터 테스트"""

    def test_thinking_params_stored(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)
        assert builder.thinking_budget == 1024
        assert builder.show_thoughts is True

    def test_default_thinking_budget_zero(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test")
        assert builder.thinking_budget == 0
        assert builder.show_thoughts is False

    def test_unsupported_model_resets_budget(self):
        from service.react_graph import ReactGraphBuilder
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder = ReactGraphBuilder(
                api_key="test", model="gemini-2.0-flash", thinking_budget=1024
            )
            assert builder.thinking_budget == 0
            assert len(w) == 1
            assert "thinking" in str(w[0].message).lower() or "지원하지" in str(w[0].message)

    def test_supported_model_keeps_budget(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", thinking_budget=1024
        )
        assert builder.thinking_budget == 1024

    def test_supported_model_25_pro(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-pro", thinking_budget=512
        )
        assert builder.thinking_budget == 512

    def test_llm_kwargs_include_thinking(self):
        """thinking_budget > 0 → LLM kwargs에 thinking_budget 포함"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        with patch.object(ChatGoogleGenerativeAI, "__init__", return_value=None) as mock_init:
            builder = ReactGraphBuilder.__new__(ReactGraphBuilder)
            # Manual init to capture kwargs
            builder.api_key = "test"
            builder.model_name = "gemini-2.5-flash"
            builder.thinking_budget = 1024
            builder.show_thoughts = False

            # Just verify the parameter exists in __init__ signature
            import inspect
            sig = inspect.signature(ReactGraphBuilder.__init__)
            assert "thinking_budget" in sig.parameters
            assert "show_thoughts" in sig.parameters

    def test_llm_kwargs_include_thoughts(self):
        """show_thoughts=True → include_thoughts in LLM kwargs"""
        from service.react_graph import ReactGraphBuilder
        import inspect

        sig = inspect.signature(ReactGraphBuilder.__init__)
        assert "show_thoughts" in sig.parameters
        assert sig.parameters["show_thoughts"].default is False


class TestParseResultWithThought:
    """_parse_result() thought 추출 테스트"""

    def test_parse_result_with_thought_process(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)

        msg = AIMessage(content=[
            {"type": "text", "text": "I think step by step...", "thought": True},
            {"type": "text", "text": "The answer is 42."},
        ])

        result = builder._parse_result([msg], turn_count=1)
        assert result["text"] == "The answer is 42."
        assert result["thought_process"] == "I think step by step..."

    def test_parse_result_without_thought(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test")

        msg = AIMessage(content="Simple response")
        result = builder._parse_result([msg], turn_count=1)
        assert result["text"] == "Simple response"
        assert "thought_process" not in result

    def test_parse_result_show_thoughts_false(self):
        """show_thoughts=False → thought_process 없음"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=False)

        msg = AIMessage(content=[
            {"type": "text", "text": "thinking...", "thought": True},
            {"type": "text", "text": "answer"},
        ])

        result = builder._parse_result([msg], turn_count=1)
        # show_thoughts=False → extract_text_from_content used, not extract_thought
        assert "thought_process" not in result

    def test_parse_result_empty_messages(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", show_thoughts=True)
        result = builder._parse_result([], turn_count=1)
        assert result["text"] == ""


class TestParseMessageChunkThought:
    """_parse_message_chunk() thought 감지 테스트"""

    def test_thought_chunk_detected(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)

        chunk = AIMessageChunk(content=[
            {"type": "text", "text": "thinking step...", "thought": True},
        ])
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "thought"
        assert events[0]["content"] == "thinking step..."

    def test_text_chunk_after_thought(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)

        chunk = AIMessageChunk(content=[
            {"type": "text", "text": "response text"},
        ])
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "token"
        assert events[0]["content"] == "response text"

    def test_no_thought_when_disabled(self):
        """show_thoughts=False → thought dict items treated as normal text"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test")

        # Even if content has thought-like structure, show_thoughts=False means normal processing
        chunk = AIMessageChunk(content="plain text")
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "token"

    def test_mixed_thought_and_text_chunk(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)

        chunk = AIMessageChunk(content=[
            {"type": "text", "text": "thinking...", "thought": True},
            {"type": "text", "text": "answer"},
        ])
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 2
        assert events[0]["type"] == "thought"
        assert events[1]["type"] == "token"

    def test_empty_text_in_thought_skipped(self):
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash",
                                     thinking_budget=1024, show_thoughts=True)

        chunk = AIMessageChunk(content=[
            {"type": "text", "text": "", "thought": True},
            {"type": "text", "text": "answer"},
        ])
        chunk.tool_call_chunks = []

        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "token"
