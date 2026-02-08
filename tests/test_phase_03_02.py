"""Phase 03-2: Structured Output (Pydantic) 테스트

Note: Phase 03-3에서 Tool Calling 패턴으로 변경되어
      _llm_tool_selector, _llm_result_processor 관련 테스트는 제거됨.
      Pydantic 모델 자체의 유효성 테스트만 유지.
"""
import os
import uuid
import pytest
from pydantic import ValidationError


class TestLLMOutputModels:
    """domain/llm_output.py Pydantic 모델 테스트"""

    def test_tool_selector_output_import(self):
        """ToolSelectorOutput 임포트 가능해야 함"""
        from domain.llm_output import ToolSelectorOutput
        assert ToolSelectorOutput is not None

    def test_result_processor_output_import(self):
        """ResultProcessorOutput 임포트 가능해야 함"""
        from domain.llm_output import ResultProcessorOutput
        assert ResultProcessorOutput is not None

    def test_tool_selector_output_valid_tool(self):
        """ToolSelectorOutput 유효한 도구명 테스트"""
        from domain.llm_output import ToolSelectorOutput

        valid_tools = ["get_current_time", "web_search", "search_pdf_knowledge", "reasoning", "none"]

        for tool in valid_tools:
            output = ToolSelectorOutput(
                selected_tool=tool,
                reason="테스트 이유"
            )
            assert output.selected_tool == tool

    def test_tool_selector_output_invalid_tool_raises_error(self):
        """ToolSelectorOutput 유효하지 않은 도구명은 ValidationError"""
        from domain.llm_output import ToolSelectorOutput

        with pytest.raises(ValidationError):
            ToolSelectorOutput(
                selected_tool="invalid_tool",
                reason="테스트"
            )

    def test_result_processor_output_valid(self):
        """ResultProcessorOutput 정상 생성"""
        from domain.llm_output import ResultProcessorOutput

        output = ResultProcessorOutput(
            needs_more_tools=True,
            summary="정보 수집 중입니다"
        )
        assert output.needs_more_tools is True
        assert output.summary == "정보 수집 중입니다"
        assert output.next_action is None  # Optional 기본값

    def test_result_processor_output_with_next_action(self):
        """ResultProcessorOutput next_action 포함"""
        from domain.llm_output import ResultProcessorOutput

        output = ResultProcessorOutput(
            needs_more_tools=True,
            summary="추가 검색 필요",
            next_action="웹에서 최신 정보 검색"
        )
        assert output.next_action == "웹에서 최신 정보 검색"

    def test_result_processor_output_needs_more_tools_false(self):
        """ResultProcessorOutput needs_more_tools=False"""
        from domain.llm_output import ResultProcessorOutput

        output = ResultProcessorOutput(
            needs_more_tools=False,
            summary="충분한 정보 수집 완료"
        )
        assert output.needs_more_tools is False


class TestDomainExports:
    """domain/__init__.py export 테스트"""

    def test_export_tool_selector_output(self):
        """ToolSelectorOutput이 domain 패키지에서 export 되어야 함"""
        from domain import ToolSelectorOutput
        assert ToolSelectorOutput is not None

    def test_export_result_processor_output(self):
        """ResultProcessorOutput이 domain 패키지에서 export 되어야 함"""
        from domain import ResultProcessorOutput
        assert ResultProcessorOutput is not None


class TestReasoningOutput:
    """ReasoningOutput Pydantic 모델 테스트"""

    def test_reasoning_output_import(self):
        """ReasoningOutput 임포트 가능해야 함"""
        from domain.llm_output import ReasoningOutput
        assert ReasoningOutput is not None

    def test_reasoning_output_valid(self):
        """ReasoningOutput 정상 생성"""
        from domain.llm_output import ReasoningOutput

        output = ReasoningOutput(
            thinking_steps=["1단계: 문제 분석", "2단계: 해결책 제시"],
            conclusion="A가 B보다 적합합니다",
            confidence="high"
        )
        assert len(output.thinking_steps) == 2
        assert output.conclusion == "A가 B보다 적합합니다"
        assert output.confidence == "high"

    def test_reasoning_output_invalid_confidence(self):
        """ReasoningOutput 잘못된 confidence는 ValidationError"""
        from domain.llm_output import ReasoningOutput

        with pytest.raises(ValidationError):
            ReasoningOutput(
                thinking_steps=["step1"],
                conclusion="결론",
                confidence="very_high"  # Invalid
            )


class TestToolCallingPattern:
    """Phase 03-3: Tool Calling 패턴 관련 테스트"""

    def test_react_graph_has_tools_attribute(self):
        """ReactGraphBuilder가 _tools 속성을 가짐"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
        )
        builder.build()

        assert hasattr(builder, "_tools")

    def test_react_graph_has_llm_with_tools(self):
        """ReactGraphBuilder가 _llm_with_tools 속성을 가짐"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
        )
        builder.build()

        assert hasattr(builder, "_llm_with_tools")


class TestStructuredOutputIntegration:
    """통합 테스트 - 실제 LLM 호출"""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_graph_invoke_returns_valid_result(self, api_key):
        """ReactGraphBuilder가 정상적으로 동작하고 결과 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="지금 몇 시야?",
            session_id=f"test_structured_output_{uuid.uuid4().hex[:8]}",
        )

        assert "text" in result
        assert isinstance(result["tool_history"], list)
        assert result["error"] is None

    def test_graph_tool_history_is_list(self, api_key):
        """tool_history가 리스트 형태로 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="현재 시간 알려줘",
            session_id=f"test_tool_history_{uuid.uuid4().hex[:8]}",
        )

        assert isinstance(result["tool_history"], list)
        if result["tool_history"]:
            assert all(isinstance(t, str) for t in result["tool_history"])
