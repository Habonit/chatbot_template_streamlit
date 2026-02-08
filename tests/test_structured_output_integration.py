"""Phase 03-2: Structured Output 통합 테스트

실제 LLM 호출로 with_structured_output 동작 검증
"""
import os
import uuid
import pytest
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from domain.llm_output import (
    ToolSelectorOutput,
    ResultProcessorOutput,
    ReasoningOutput,
)

# .env 파일 로드
load_dotenv()


@pytest.fixture
def gemini_api_key():
    """Gemini API 키 fixture"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return api_key


@pytest.fixture
def base_llm(gemini_api_key):
    """기본 LLM fixture"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0.1,  # 일관된 결과를 위해 낮은 temperature
    )


class TestToolSelectorStructuredOutput:
    """Tool Selector Structured Output 통합 테스트"""

    def test_tool_selector_returns_pydantic_model(self, base_llm):
        """with_structured_output이 Pydantic 모델 인스턴스를 반환"""
        structured_llm = base_llm.with_structured_output(
            ToolSelectorOutput,
            method="json_schema",
        )

        prompt = """다음 질문에 대해 어떤 도구를 사용해야 하는지 결정하세요.

질문: "지금 몇 시야?"

사용 가능한 도구:
- get_current_time: 현재 시각 조회
- web_search: 웹 검색
- search_pdf_knowledge: PDF 문서 검색
- reasoning: 복잡한 추론
- none: 도구 필요 없음"""

        response = structured_llm.invoke([HumanMessage(content=prompt)])

        assert isinstance(response, ToolSelectorOutput)
        assert response.selected_tool == "get_current_time"
        assert len(response.reason) > 0

    def test_tool_selector_selects_none_for_casual(self, base_llm):
        """일상적 대화에는 'none' 선택"""
        structured_llm = base_llm.with_structured_output(
            ToolSelectorOutput,
            method="json_schema",
        )

        prompt = """다음 질문에 대해 어떤 도구를 사용해야 하는지 결정하세요.

질문: "안녕하세요!"

사용 가능한 도구:
- get_current_time: 현재 시각 조회
- web_search: 웹 검색
- search_pdf_knowledge: PDF 문서 검색
- reasoning: 복잡한 추론
- none: 도구 필요 없음 (일상적 대화)"""

        response = structured_llm.invoke([HumanMessage(content=prompt)])

        assert isinstance(response, ToolSelectorOutput)
        assert response.selected_tool == "none"


class TestResultProcessorStructuredOutput:
    """Result Processor Structured Output 통합 테스트"""

    def test_result_processor_returns_pydantic_model(self, base_llm):
        """with_structured_output이 Pydantic 모델 인스턴스를 반환"""
        structured_llm = base_llm.with_structured_output(
            ResultProcessorOutput,
            method="json_schema",
        )

        prompt = """다음 정보가 사용자 질문에 답하기에 충분한지 판단하세요.

사용자 질문: "지금 몇 시야?"

수집된 정보:
[get_current_time]: 2024-01-15 14:30:00 (KST)

추가 도구가 필요한가요?"""

        response = structured_llm.invoke([HumanMessage(content=prompt)])

        assert isinstance(response, ResultProcessorOutput)
        assert response.needs_more_tools is False
        assert len(response.summary) > 0

    def test_result_processor_needs_more_tools(self, base_llm):
        """추가 도구가 필요한 경우 needs_more_tools=True"""
        structured_llm = base_llm.with_structured_output(
            ResultProcessorOutput,
            method="json_schema",
        )

        prompt = """다음 정보가 사용자 질문에 답하기에 충분한지 판단하세요.

사용자 질문: "오늘 날씨와 내일 날씨를 비교해서 분석해줘"

수집된 정보:
[web_search]: 오늘 서울 날씨는 맑음, 기온 15도입니다.

아직 내일 날씨 정보가 없고 분석도 필요합니다. 추가 도구가 필요한가요?"""

        response = structured_llm.invoke([HumanMessage(content=prompt)])

        assert isinstance(response, ResultProcessorOutput)
        assert response.needs_more_tools is True


class TestReasoningStructuredOutput:
    """Reasoning Structured Output 통합 테스트"""

    def test_reasoning_returns_pydantic_model(self, base_llm):
        """with_structured_output이 Pydantic 모델 인스턴스를 반환"""
        structured_llm = base_llm.with_structured_output(
            ReasoningOutput,
            method="json_schema",
        )

        prompt = """다음 문제를 단계별로 추론하세요.

문제: "5 + 3 * 2 = ?"

연산 우선순위를 고려하여 단계별로 풀이하세요."""

        response = structured_llm.invoke([HumanMessage(content=prompt)])

        assert isinstance(response, ReasoningOutput)
        assert len(response.thinking_steps) > 0
        assert "11" in response.conclusion
        assert response.confidence in ["high", "medium", "low"]


class TestReactGraphStructuredOutput:
    """ReactGraphBuilder Structured Output 통합 테스트"""

    def test_graph_invoke_returns_valid_result(self, gemini_api_key):
        """ReactGraphBuilder가 정상적으로 동작하고 결과 반환"""
        from service.react_graph import ReactGraphBuilder

        graph = ReactGraphBuilder(api_key=gemini_api_key, db_path=":memory:")
        graph.build()

        result = graph.invoke(
            user_input="지금 몇 시야?",
            session_id=f"test_structured_output_{uuid.uuid4().hex[:8]}",
        )

        assert "text" in result
        assert isinstance(result["tool_history"], list)
        assert result["error"] is None

    def test_graph_tool_history_is_list(self, gemini_api_key):
        """tool_history가 리스트 형태로 반환"""
        from service.react_graph import ReactGraphBuilder

        graph = ReactGraphBuilder(api_key=gemini_api_key, db_path=":memory:")
        graph.build()

        result = graph.invoke(
            user_input="오늘 날짜 알려줘",
            session_id=f"test_tool_history_{uuid.uuid4().hex[:8]}",
        )

        assert isinstance(result["tool_history"], list)
        # get_current_time이 호출되었을 수 있음
        if result["tool_history"]:
            assert all(isinstance(t, str) for t in result["tool_history"])
