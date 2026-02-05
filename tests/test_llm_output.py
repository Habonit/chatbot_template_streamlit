"""Phase 03-2: LLM Structured Output 모델 단위 테스트

TDD 방식으로 Pydantic 모델 검증
"""
import pytest
from pydantic import ValidationError

from domain.llm_output import (
    ToolSelectorOutput,
    ResultProcessorOutput,
    ReasoningOutput,
)


class TestToolSelectorOutput:
    """ToolSelectorOutput Pydantic 모델 테스트"""

    def test_valid_tool_selection(self):
        """유효한 도구 선택 테스트"""
        output = ToolSelectorOutput(
            selected_tool="web_search",
            reason="최신 정보가 필요합니다"
        )
        assert output.selected_tool == "web_search"
        assert output.reason == "최신 정보가 필요합니다"

    def test_all_valid_tools(self):
        """모든 유효한 도구 이름 테스트"""
        valid_tools = [
            "get_current_time",
            "web_search",
            "search_pdf_knowledge",
            "reasoning",
            "none"
        ]
        for tool in valid_tools:
            output = ToolSelectorOutput(selected_tool=tool, reason="test")
            assert output.selected_tool == tool

    def test_invalid_tool_raises_error(self):
        """잘못된 도구 이름은 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ToolSelectorOutput(
                selected_tool="invalid_tool",
                reason="test"
            )

    def test_missing_reason_raises_error(self):
        """reason 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ToolSelectorOutput(selected_tool="web_search")

    def test_missing_selected_tool_raises_error(self):
        """selected_tool 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ToolSelectorOutput(reason="test reason")


class TestResultProcessorOutput:
    """ResultProcessorOutput Pydantic 모델 테스트"""

    def test_valid_output_with_required_fields(self):
        """필수 필드만으로 생성 테스트"""
        output = ResultProcessorOutput(
            needs_more_tools=True,
            summary="정보 수집 중"
        )
        assert output.needs_more_tools is True
        assert output.summary == "정보 수집 중"
        assert output.next_action is None  # Optional 필드

    def test_valid_output_with_all_fields(self):
        """모든 필드 포함하여 생성 테스트"""
        output = ResultProcessorOutput(
            needs_more_tools=True,
            summary="웹 검색 완료",
            next_action="reasoning 도구로 분석 필요"
        )
        assert output.needs_more_tools is True
        assert output.summary == "웹 검색 완료"
        assert output.next_action == "reasoning 도구로 분석 필요"

    def test_needs_more_tools_false(self):
        """needs_more_tools가 False인 경우"""
        output = ResultProcessorOutput(
            needs_more_tools=False,
            summary="충분한 정보 수집 완료"
        )
        assert output.needs_more_tools is False

    def test_missing_needs_more_tools_raises_error(self):
        """needs_more_tools 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ResultProcessorOutput(summary="test")

    def test_missing_summary_raises_error(self):
        """summary 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ResultProcessorOutput(needs_more_tools=True)


class TestReasoningOutput:
    """ReasoningOutput Pydantic 모델 테스트"""

    def test_valid_reasoning_output(self):
        """유효한 ReasoningOutput 생성"""
        output = ReasoningOutput(
            thinking_steps=["1단계: 문제 분석", "2단계: 정보 수집", "3단계: 결론 도출"],
            conclusion="분석 결과 A가 더 적합합니다",
            confidence="high"
        )
        assert len(output.thinking_steps) == 3
        assert output.conclusion == "분석 결과 A가 더 적합합니다"
        assert output.confidence == "high"

    def test_all_confidence_levels(self):
        """모든 유효한 confidence 레벨 테스트"""
        for level in ["high", "medium", "low"]:
            output = ReasoningOutput(
                thinking_steps=["step1"],
                conclusion="test",
                confidence=level
            )
            assert output.confidence == level

    def test_invalid_confidence_raises_error(self):
        """잘못된 confidence 값은 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ReasoningOutput(
                thinking_steps=["step1"],
                conclusion="test",
                confidence="very_high"  # Invalid
            )

    def test_empty_thinking_steps(self):
        """빈 thinking_steps 리스트도 유효"""
        output = ReasoningOutput(
            thinking_steps=[],
            conclusion="직접 결론",
            confidence="low"
        )
        assert output.thinking_steps == []

    def test_missing_thinking_steps_raises_error(self):
        """thinking_steps 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ReasoningOutput(
                conclusion="test",
                confidence="high"
            )

    def test_missing_conclusion_raises_error(self):
        """conclusion 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ReasoningOutput(
                thinking_steps=["step1"],
                confidence="high"
            )

    def test_missing_confidence_raises_error(self):
        """confidence 필드 누락 시 ValidationError 발생"""
        with pytest.raises(ValidationError):
            ReasoningOutput(
                thinking_steps=["step1"],
                conclusion="test"
            )
