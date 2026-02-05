"""Phase 02-5: 통합 테스트 (실제 API 사용)

실행 방법:
    uv run pytest tests/test_integration.py -v -s

환경 변수:
    .env 파일에 GEMINI_API_KEY와 TAVILY_API_KEY가 설정되어 있어야 합니다.
"""
import os
import pytest
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 확인
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@pytest.fixture
def google_api_key():
    """Google API 키 fixture"""
    if not GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not found in .env")
    return GEMINI_API_KEY


@pytest.fixture
def tavily_api_key():
    """Tavily API 키 fixture"""
    if not TAVILY_API_KEY:
        pytest.skip("TAVILY_API_KEY not found in .env")
    return TAVILY_API_KEY


class TestReactGraphIntegration:
    """ReactGraphBuilder 통합 테스트 (Phase 02-5)"""

    def test_simple_question(self, google_api_key):
        """간단한 질문 처리 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=5,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="1 + 1은 뭐야?",
            session_id="test_simple",
        )

        assert result["text"] is not None
        assert len(result["text"]) > 0
        assert result["error"] is None
        print(f"\n응답: {result['text']}")
        print(f"툴 히스토리: {result['tool_history']}")

    def test_time_tool_invocation(self, google_api_key):
        """시간 툴 호출 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=5,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="지금 몇 시야?",
            session_id="test_time",
        )

        assert result["text"] is not None
        # 시간 툴이 호출되었어야 함
        assert "get_current_time" in result["tool_history"]
        print(f"\n응답: {result['text']}")
        print(f"툴 히스토리: {result['tool_history']}")
        print(f"툴 결과: {result['tool_results']}")

    def test_web_search_tool_invocation(self, google_api_key, tavily_api_key):
        """웹 검색 툴 호출 테스트"""
        from service.react_graph import ReactGraphBuilder
        from service.search_service import SearchService

        search_service = SearchService(api_key=tavily_api_key)

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=5,
            search_service=search_service,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="파이썬 3.13 새 기능 검색해줘",
            session_id="test_search",
        )

        assert result["text"] is not None
        # 웹 검색 툴이 호출되었어야 함
        assert "web_search" in result["tool_history"]
        print(f"\n응답: {result['text'][:500]}...")
        print(f"툴 히스토리: {result['tool_history']}")

    def test_reasoning_tool_invocation(self, google_api_key):
        """추론 툴 호출 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=5,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="파이썬과 자바를 비교 분석해줘",
            session_id="test_reasoning",
        )

        assert result["text"] is not None
        # 추론 툴이 호출되었어야 함
        assert "reasoning" in result["tool_history"]
        print(f"\n응답: {result['text'][:500]}...")
        print(f"툴 히스토리: {result['tool_history']}")

    def test_multiple_tools_sequential(self, google_api_key, tavily_api_key):
        """여러 툴 순차 호출 테스트 (ReAct 루프)"""
        from service.react_graph import ReactGraphBuilder
        from service.search_service import SearchService

        search_service = SearchService(api_key=tavily_api_key)

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=5,
            search_service=search_service,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="LangGraph 최신 기능을 검색하고 분석해줘",
            session_id="test_multi",
        )

        assert result["text"] is not None
        # 여러 툴이 호출되었어야 함 (검색 + 추론)
        assert len(result["tool_history"]) >= 1
        print(f"\n응답: {result['text'][:500]}...")
        print(f"툴 히스토리: {result['tool_history']}")
        print(f"반복 횟수: {result['iteration']}")

    def test_max_iterations_respected(self, google_api_key):
        """max_iterations 설정이 적용되는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            max_iterations=2,
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="안녕",
            session_id="test_max_iter",
        )

        assert result["error"] is None
        assert result["iteration"] <= 2
        print(f"\n응답: {result['text']}")
        print(f"반복 횟수: {result['iteration']}")


class TestPhase07Integration:
    """Phase 02-7: 통합 테스트"""

    def test_casual_conversation_fast_path(self, google_api_key):
        """일상적 대화 Fast-path 통합 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            db_path=":memory:",
        )

        # 일상적 대화 테스트
        casual_inputs = ["오호", "그렇구나", "고마워", "ㅎㅎ"]

        for user_input in casual_inputs:
            result = builder.invoke(
                user_input=user_input,
                session_id=f"test_casual_{user_input}",
            )

            assert result["error"] is None
            assert result.get("is_casual") is True, f"'{user_input}'은 is_casual=True여야 함"
            assert result["tool_history"] == [], f"'{user_input}'은 툴을 사용하면 안됨"
            print(f"\n[{user_input}] → {result['text']}")

    def test_token_usage_tracking(self, google_api_key):
        """토큰 사용량 추적 통합 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            db_path=":memory:",
        )

        result = builder.invoke(
            user_input="파이썬이 뭐야?",
            session_id="test_token_tracking",
        )

        assert result["error"] is None
        assert result["input_tokens"] > 0, "input_tokens가 0보다 커야 함"
        assert result["output_tokens"] > 0, "output_tokens가 0보다 커야 함"
        assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]

        print(f"\n응답: {result['text'][:200]}...")
        print(f"입력 토큰: {result['input_tokens']}")
        print(f"출력 토큰: {result['output_tokens']}")
        print(f"총 토큰: {result['total_tokens']}")

    def test_reasoning_detection(self, google_api_key):
        """추론 모드 감지 통합 테스트"""
        from service.react_graph import ReactGraphBuilder
        from service.reasoning_detector import detect_reasoning_need

        # 추론 모드 감지 확인
        user_input = "A와 B의 차이점을 비교해서 분석해줘"
        mode = detect_reasoning_need(user_input)
        assert mode == "reasoning", f"Expected 'reasoning', got '{mode}'"

        builder = ReactGraphBuilder(
            api_key=google_api_key,
            model="gemini-2.0-flash",
            db_path=":memory:",
        )

        # 추론이 필요한 질문
        result = builder.invoke(
            user_input=user_input,
            session_id="test_reasoning_detection",
        )

        assert result["error"] is None
        assert result["text"], "응답 텍스트가 있어야 함"
        # LLM이 도구를 사용할지는 선택적 (도구 없이도 응답 가능)
        print(f"\n응답: {result['text'][:300]}...")
        print(f"툴 히스토리: {result['tool_history']}")


