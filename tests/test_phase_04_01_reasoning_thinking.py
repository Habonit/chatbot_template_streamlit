"""Phase 04-1: Reasoning Tool 전용 Thinking 분리 테스트

TDD Step 1: LLM 인스턴스 분리
TDD Step 2: reasoning 도구 구조화 반환
TDD Step 3: build() 파라미터 전달
TDD Step 4: _parse_result() JSON 기반 thought 추출
TDD Step 5: _parse_message_chunk() 유지 확인
TDD Step 6: sidebar + education_tips 변경
"""
import json
import warnings
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# =============================================================================
# Step 1: LLM 인스턴스 분리 테스트
# =============================================================================
class TestReasoningThinkingSeparation:
    """reasoning 전용 LLM 분리 테스트"""

    def test_llm_for_reasoning_created_when_thinking_budget_positive(self):
        """thinking_budget > 0 → _llm_for_reasoning이 별도 인스턴스"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", thinking_budget=1024
        )
        assert builder._llm_for_reasoning is not builder._llm

    def test_llm_for_reasoning_same_when_thinking_budget_zero(self):
        """thinking_budget == 0 → _llm_for_reasoning은 _llm과 동일"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", thinking_budget=0
        )
        assert builder._llm_for_reasoning is builder._llm

    def test_main_llm_has_no_thinking_budget(self):
        """메인 LLM에는 thinking 관련 설정이 없어야 함"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", thinking_budget=1024
        )
        # 메인 LLM은 thinking_budget이 없어야 함
        # ChatGoogleGenerativeAI는 Pydantic 모델이므로 kwargs 검증
        main_llm = builder._llm
        assert not getattr(main_llm, "thinking_budget", None)

    def test_reasoning_llm_has_thinking_budget(self):
        """reasoning LLM에는 thinking_budget 값이 설정되어 있어야 함"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash",
            thinking_budget=1024, show_thoughts=True,
        )
        assert builder._reasoning_thinking_budget == 1024

    def test_unsupported_model_no_reasoning_llm(self):
        """미지원 모델에서는 reasoning LLM도 메인과 동일"""
        from service.react_graph import ReactGraphBuilder
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            builder = ReactGraphBuilder(
                api_key="test", model="gemini-2.0-flash", thinking_budget=1024
            )
        assert builder._llm_for_reasoning is builder._llm
        assert builder.thinking_budget == 0


# =============================================================================
# Step 2: reasoning 도구 구조화 반환 테스트
# =============================================================================
class TestReasoningToolStructuredReturn:
    """reasoning 도구의 JSON 구조화 반환 테스트"""

    def test_reasoning_returns_json_with_thought_when_show_thoughts(self):
        """show_thoughts=True + thinking 활성화 → JSON에 thought 포함"""
        from service.tools import create_tools_with_services

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = [
            {"type": "text", "text": "step-by-step 추론...", "thought": True},
            {"type": "text", "text": "최종 분석 결과입니다."},
        ]
        mock_llm.invoke.return_value.usage_metadata = {
            "input_tokens": 10, "output_tokens": 20,
        }
        tools = create_tools_with_services(reasoning_llm=mock_llm, show_thoughts=True)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "테스트", "context": ""})
        parsed = json.loads(result)
        assert "thought" in parsed
        assert "analysis" in parsed
        assert parsed["thought"] == "step-by-step 추론..."
        assert parsed["analysis"] == "최종 분석 결과입니다."

    def test_reasoning_returns_text_only_when_no_show_thoughts(self):
        """show_thoughts=False → thought 없이 text만 반환"""
        from service.tools import create_tools_with_services

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = [
            {"type": "text", "text": "추론...", "thought": True},
            {"type": "text", "text": "결과입니다."},
        ]
        tools = create_tools_with_services(reasoning_llm=mock_llm, show_thoughts=False)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "테스트", "context": ""})
        assert result == "결과입니다."  # plain text, not JSON

    def test_reasoning_handles_string_content(self):
        """thinking 없는 LLM 응답 (str) → 그대로 반환"""
        from service.tools import create_tools_with_services

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "일반 분석 결과"
        tools = create_tools_with_services(reasoning_llm=mock_llm, show_thoughts=True)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "테스트", "context": ""})
        assert result == "일반 분석 결과"

    def test_reasoning_uses_reasoning_llm_over_main_llm(self):
        """reasoning_llm이 있으면 llm 대신 reasoning_llm 사용"""
        from service.tools import create_tools_with_services

        main_llm = MagicMock()
        reasoning_llm = MagicMock()
        reasoning_llm.invoke.return_value.content = "reasoning result"
        tools = create_tools_with_services(llm=main_llm, reasoning_llm=reasoning_llm)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        reasoning_tool.invoke({"question": "test", "context": ""})
        reasoning_llm.invoke.assert_called_once()
        main_llm.invoke.assert_not_called()

    def test_reasoning_fallback_to_main_llm_on_error(self):
        """reasoning LLM 실패 시 메인 LLM으로 fallback"""
        from service.tools import create_tools_with_services

        main_llm = MagicMock()
        main_llm.invoke.return_value.content = "fallback result"
        reasoning_llm = MagicMock()
        reasoning_llm.invoke.side_effect = Exception("API error")
        tools = create_tools_with_services(llm=main_llm, reasoning_llm=reasoning_llm)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "test", "context": ""})
        assert result == "fallback result"
        main_llm.invoke.assert_called_once()

    def test_reasoning_error_when_both_fail(self):
        """reasoning + main 둘 다 실패 → 에러 메시지"""
        from service.tools import create_tools_with_services

        main_llm = MagicMock()
        main_llm.invoke.side_effect = Exception("main error")
        reasoning_llm = MagicMock()
        reasoning_llm.invoke.side_effect = Exception("reasoning error")
        tools = create_tools_with_services(llm=main_llm, reasoning_llm=reasoning_llm)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "test", "context": ""})
        assert "오류" in result

    def test_reasoning_json_includes_token_info(self):
        """show_thoughts=True → JSON에 reasoning_tokens 포함"""
        from service.tools import create_tools_with_services

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = [
            {"type": "text", "text": "추론...", "thought": True},
            {"type": "text", "text": "결과"},
        ]
        mock_llm.invoke.return_value.usage_metadata = {
            "input_tokens": 100, "output_tokens": 50,
            "output_token_details": {"reasoning": 30},
        }
        tools = create_tools_with_services(reasoning_llm=mock_llm, show_thoughts=True)
        reasoning_tool = next(t for t in tools if t.name == "reasoning")
        result = reasoning_tool.invoke({"question": "test", "context": ""})
        parsed = json.loads(result)
        assert "reasoning_tokens" in parsed
        assert parsed["reasoning_tokens"]["thinking"] == 30


# =============================================================================
# Step 3: build() 파라미터 전달 테스트
# =============================================================================
class TestBuildPassesReasoningLlm:
    """build()가 reasoning 전용 LLM을 도구에 올바르게 전달하는지 테스트"""

    def test_build_passes_reasoning_llm_to_tools(self):
        """create_tools_with_services에 reasoning_llm으로 전달"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", thinking_budget=1024
        )
        with patch("service.react_graph.create_tools_with_services") as mock_create:
            mock_create.return_value = []
            with patch.object(
                ChatGoogleGenerativeAI, "bind_tools", return_value=MagicMock()
            ):
                builder.build()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["reasoning_llm"] is builder._llm_for_reasoning
            assert call_kwargs["llm"] is builder._llm

    def test_build_passes_show_thoughts_to_tools(self):
        """create_tools_with_services에 show_thoughts 전달"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash",
            thinking_budget=1024, show_thoughts=True,
        )
        with patch("service.react_graph.create_tools_with_services") as mock_create:
            mock_create.return_value = []
            with patch.object(
                ChatGoogleGenerativeAI, "bind_tools", return_value=MagicMock()
            ):
                builder.build()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["show_thoughts"] is True

    def test_show_thoughts_consistency_between_build_and_parse(self):
        """build()와 _parse_result()가 동일한 show_thoughts 사용"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash",
            thinking_budget=1024, show_thoughts=True,
        )
        with patch("service.react_graph.create_tools_with_services") as mock_create:
            mock_create.return_value = []
            with patch.object(
                ChatGoogleGenerativeAI, "bind_tools", return_value=MagicMock()
            ):
                builder.build()
            assert mock_create.call_args[1]["show_thoughts"] is builder.show_thoughts


# =============================================================================
# Step 4: _parse_result() JSON 기반 thought 추출 테스트
# =============================================================================
class TestParseResultThoughtFromJSON:
    """_parse_result가 reasoning 도구의 JSON에서 thought를 추출하는지 테스트"""

    def test_parse_result_no_thought_from_main_llm(self):
        """메인 LLM 응답에서 thought 추출 안 함 (show_thoughts=True여도)"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", show_thoughts=True
        )
        msg = AIMessage(content="일반 텍스트 응답")
        result = builder._parse_result([msg], turn_count=1)
        assert "thought_process" not in result or result.get("thought_process") == ""

    def test_parse_result_extracts_thought_from_reasoning_json(self):
        """reasoning 도구의 JSON 결과에서 thought_process 추출"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", show_thoughts=True
        )
        tool_msg = MagicMock()
        tool_msg.name = "reasoning"
        tool_msg.content = json.dumps({"thought": "step-by-step", "analysis": "분석"})
        tool_msg.type = "tool"
        tool_msg.tool_calls = []

        ai_msg = AIMessage(content="최종 답변")
        result = builder._parse_result([tool_msg, ai_msg], turn_count=0)
        assert result.get("thought_process") == "step-by-step"

    def test_parse_result_no_thought_when_show_thoughts_false(self):
        """show_thoughts=False → JSON이 있어도 thought 추출 안 함"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", show_thoughts=False
        )
        tool_msg = MagicMock()
        tool_msg.name = "reasoning"
        tool_msg.content = json.dumps({"thought": "추론...", "analysis": "결과"})
        tool_msg.type = "tool"
        tool_msg.tool_calls = []

        ai_msg = AIMessage(content="답변")
        result = builder._parse_result([tool_msg, ai_msg], turn_count=0)
        assert "thought_process" not in result

    def test_parse_result_handles_non_json_reasoning(self):
        """reasoning 도구가 plain text 반환 시 → thought 없음"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", show_thoughts=True
        )
        tool_msg = MagicMock()
        tool_msg.name = "reasoning"
        tool_msg.content = "plain text reasoning result"
        tool_msg.type = "tool"
        tool_msg.tool_calls = []

        ai_msg = AIMessage(content="답변")
        result = builder._parse_result([tool_msg, ai_msg], turn_count=0)
        assert "thought_process" not in result or result.get("thought_process") == ""

    def test_parse_result_reasoning_tokens_extracted(self):
        """reasoning 도구의 thinking 토큰 정보가 결과 dict에 포함"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash", show_thoughts=True
        )
        tool_msg = MagicMock()
        tool_msg.name = "reasoning"
        tool_msg.content = json.dumps({
            "thought": "추론", "analysis": "결과",
            "reasoning_tokens": {"input": 100, "output": 50, "thinking": 30},
        })
        tool_msg.type = "tool"
        tool_msg.tool_calls = []

        ai_msg = AIMessage(content="답변")
        result = builder._parse_result([tool_msg, ai_msg], turn_count=0)
        assert result["reasoning_tokens"]["input"] == 100
        assert result["reasoning_tokens"]["output"] == 50
        assert result["reasoning_tokens"]["thinking"] == 30


# =============================================================================
# Step 5: _parse_message_chunk() 기존 동작 유지 테스트
# =============================================================================
class TestParseMessageChunkPreserved:
    """스트리밍 chunk 파싱이 정상 동작하는지 테스트"""

    def test_normal_text_chunk_still_works(self):
        """일반 텍스트 스트리밍 정상 동작"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(
            api_key="test", model="gemini-2.5-flash",
            show_thoughts=True, thinking_budget=1024,
        )
        chunk = AIMessageChunk(content="일반 텍스트")
        chunk.tool_call_chunks = []
        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "token"

    def test_tool_message_chunk_still_works(self):
        """ToolMessage 스트리밍 정상 동작"""
        from service.react_graph import ReactGraphBuilder
        builder = ReactGraphBuilder(api_key="test", model="gemini-2.5-flash")
        chunk = ToolMessage(content="tool result", name="reasoning", tool_call_id="tc1")
        events = builder._parse_message_chunk(chunk, {}, [])
        assert len(events) == 1
        assert events[0]["type"] == "tool_result"
        assert events[0]["name"] == "reasoning"


# =============================================================================
# Step 6: sidebar + education_tips 테스트
# =============================================================================
class TestThinkingEducationUpdated:
    """education_tips.py 변경 테스트"""

    def test_thinking_education_mentions_reasoning_tool(self):
        """reasoning 도구 전용 설명 포함"""
        from component.education_tips import get_thinking_education
        result = get_thinking_education(1024, "")
        assert "reasoning 도구" in result["explanation"]
        assert "일반 대화" in result["explanation"]

    def test_thinking_education_title_changed(self):
        """title이 'Reasoning Thinking'으로 변경"""
        from component.education_tips import get_thinking_education
        result = get_thinking_education(1024, "")
        assert result["title"] == "Reasoning Thinking"

    def test_thinking_education_with_thought_process(self):
        """thought_process 있을 때 캡처 문구 포함"""
        from component.education_tips import get_thinking_education
        result = get_thinking_education(1024, "some thought")
        assert "reasoning 도구의 사고 과정이 캡처" in result["explanation"]
