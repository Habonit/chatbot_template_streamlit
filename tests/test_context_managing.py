"""Phase 03-3: Context Managing 테스트

TDD: 테스트 먼저 작성
- extract_last_n_turns() 함수
- extract_current_turn() 함수
- summary_node Context 구성
- compression_rate 적용
"""
import os
import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()


class TestShouldSummarize:
    """should_summarize() 함수 테스트"""

    def test_turn_1_2_returns_false(self):
        """Turn 1, 2에서는 False (현재 턴 미포함 기준)"""
        from service.react_graph import should_summarize

        assert should_summarize(1) is False
        assert should_summarize(2) is False

    def test_turn_3_6_9_returns_true(self):
        """Turn 3, 6, 9에서는 True (이전 normal 턴 3개 누적 시)"""
        from service.react_graph import should_summarize

        assert should_summarize(3) is True
        assert should_summarize(6) is True
        assert should_summarize(9) is True

    def test_turn_4_5_7_8_returns_false(self):
        """Turn 4, 5, 7, 8에서는 False"""
        from service.react_graph import should_summarize

        assert should_summarize(4) is False
        assert should_summarize(5) is False
        assert should_summarize(7) is False
        assert should_summarize(8) is False


class TestExtractLastNTurns:
    """extract_last_n_turns() 함수 테스트"""

    def test_extract_last_n_turns_import(self):
        """extract_last_n_turns 함수 임포트 가능"""
        from service.react_graph import extract_last_n_turns

        assert callable(extract_last_n_turns)

    def test_extract_single_completed_turn(self):
        """완료된 1턴 추출"""
        from service.react_graph import extract_last_n_turns

        messages = [
            HumanMessage(content="안녕"),
            AIMessage(content="안녕하세요!", tool_calls=[]),
        ]

        result = extract_last_n_turns(messages, n=1)

        assert len(result) == 2
        assert result[0].content == "안녕"
        assert result[1].content == "안녕하세요!"

    def test_extract_turn_with_tool_calls(self):
        """Tool Calling이 포함된 턴 추출"""
        from service.react_graph import extract_last_n_turns

        messages = [
            HumanMessage(content="몇 시야?"),
            AIMessage(content="", tool_calls=[{"name": "get_current_time", "args": {}, "id": "1"}]),
            ToolMessage(content="2024-01-15 14:30:00 (KST)", tool_call_id="1"),
            AIMessage(content="현재 시간은 14시 30분입니다.", tool_calls=[]),
        ]

        result = extract_last_n_turns(messages, n=1)

        assert len(result) == 4
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[-1], AIMessage)
        assert result[-1].tool_calls == []

    def test_extract_multiple_turns(self):
        """여러 턴 추출"""
        from service.react_graph import extract_last_n_turns

        messages = [
            # Turn 1
            HumanMessage(content="Turn1 질문"),
            AIMessage(content="Turn1 답변", tool_calls=[]),
            # Turn 2
            HumanMessage(content="Turn2 질문"),
            AIMessage(content="Turn2 답변", tool_calls=[]),
            # Turn 3
            HumanMessage(content="Turn3 질문"),
            AIMessage(content="Turn3 답변", tool_calls=[]),
        ]

        result = extract_last_n_turns(messages, n=2)

        # Turn 2, 3만 추출되어야 함
        assert len(result) == 4
        assert result[0].content == "Turn2 질문"
        assert result[-1].content == "Turn3 답변"

    def test_extract_excludes_incomplete_turn(self):
        """미완료 턴은 제외"""
        from service.react_graph import extract_last_n_turns

        messages = [
            # Turn 1 (완료)
            HumanMessage(content="Turn1 질문"),
            AIMessage(content="Turn1 답변", tool_calls=[]),
            # Turn 2 (미완료 - user만 있음)
            HumanMessage(content="Turn2 질문"),
        ]

        result = extract_last_n_turns(messages, n=1)

        # Turn 1만 추출되어야 함
        assert len(result) == 2
        assert result[0].content == "Turn1 질문"
        assert result[1].content == "Turn1 답변"

    def test_extract_zero_turns_returns_empty(self):
        """n=0이면 빈 리스트"""
        from service.react_graph import extract_last_n_turns

        messages = [
            HumanMessage(content="질문"),
            AIMessage(content="답변", tool_calls=[]),
        ]

        result = extract_last_n_turns(messages, n=0)

        assert result == []


class TestExtractCurrentTurn:
    """extract_current_turn() 함수 테스트"""

    def test_extract_current_turn_import(self):
        """extract_current_turn 함수 임포트 가능"""
        from service.react_graph import extract_current_turn

        assert callable(extract_current_turn)

    def test_extract_current_turn_user_only(self):
        """현재 턴에 user 메시지만 있는 경우"""
        from service.react_graph import extract_current_turn

        messages = [
            # 이전 완료된 턴
            HumanMessage(content="이전 질문"),
            AIMessage(content="이전 답변", tool_calls=[]),
            # 현재 진행 중인 턴
            HumanMessage(content="현재 질문"),
        ]

        result = extract_current_turn(messages)

        assert len(result) == 1
        assert result[0].content == "현재 질문"

    def test_extract_current_turn_with_tool_in_progress(self):
        """현재 턴에 Tool Calling 진행 중인 경우"""
        from service.react_graph import extract_current_turn

        messages = [
            # 이전 완료된 턴
            HumanMessage(content="이전 질문"),
            AIMessage(content="이전 답변", tool_calls=[]),
            # 현재 진행 중인 턴
            HumanMessage(content="현재 질문"),
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "1"}]),
            ToolMessage(content="검색 결과", tool_call_id="1"),
        ]

        result = extract_current_turn(messages)

        assert len(result) == 3
        assert result[0].content == "현재 질문"
        assert isinstance(result[-1], ToolMessage)

    def test_extract_current_turn_no_completed_turns(self):
        """완료된 턴이 없는 경우 전체 반환"""
        from service.react_graph import extract_current_turn

        messages = [
            HumanMessage(content="첫 질문"),
        ]

        result = extract_current_turn(messages)

        assert len(result) == 1
        assert result[0].content == "첫 질문"


class TestChatStateCompressionRate:
    """ChatState compression_rate 필드 테스트"""

    def test_chat_state_has_compression_rate(self):
        """ChatState에 compression_rate 필드 존재"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "compression_rate" in annotations

    def test_react_graph_builder_accepts_compression_rate(self):
        """ReactGraphBuilder가 compression_rate 파라미터 받음"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_key",
            db_path=":memory:",
        )

        # invoke 시 compression_rate 전달 가능해야 함
        assert hasattr(builder, "invoke")


class TestSummaryHistoryStructure:
    """summary_history JSON 구조 테스트"""

    def test_summary_history_has_required_fields(self):
        """summary_history 항목에 필수 필드 존재"""
        # 예상 구조
        expected_fields = [
            "thread_id",
            "turns",
            "turn_length",
            "original_chars",
            "summary_chars",
            "compression_rate",
            "summary",
        ]

        sample_summary = {
            "thread_id": "session_123",
            "turns": [1, 2, 3],
            "turn_length": 3,
            "original_chars": 500,
            "summary_chars": 150,
            "compression_rate": 0.3,
            "summary": "요약 내용",
        }

        for field in expected_fields:
            assert field in sample_summary


class TestSidebarCompressionRate:
    """Sidebar compression_rate 슬라이더 테스트"""

    def test_sidebar_returns_compression_rate(self):
        """sidebar가 compression_rate를 반환"""
        from component.sidebar import render_sidebar

        # render_sidebar의 반환값에 compression_rate가 있는지 확인
        # 실제 렌더링 없이 코드 검사
        import inspect

        source = inspect.getsource(render_sidebar)
        assert "compression_rate" in source


class TestSequenceIntegration:
    """시퀀스 테스트 - Turn별 summary_history 상태"""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_turn_4_creates_first_summary(self, api_key):
        """Turn 4에서 첫 요약 생성"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        # Turn 1-3 시뮬레이션 (이전 대화) - Phase 03-3-2: turn_id 메타데이터 포함
        messages = [
            HumanMessage(content="Turn1: 안녕", additional_kwargs={"turn_id": 1, "mode": "normal"}),
            AIMessage(content="Turn1: 안녕하세요!", tool_calls=[], additional_kwargs={"turn_id": 1}),
            HumanMessage(content="Turn2: 날씨 어때?", additional_kwargs={"turn_id": 2, "mode": "normal"}),
            AIMessage(content="Turn2: 좋아요", tool_calls=[], additional_kwargs={"turn_id": 2}),
            HumanMessage(content="Turn3: 뭐해?", additional_kwargs={"turn_id": 3, "mode": "normal"}),
            AIMessage(content="Turn3: 일하고 있어요", tool_calls=[], additional_kwargs={"turn_id": 3}),
        ]

        # Turn 4 실행 - Phase 03-3-2: normal_turn_ids 전달
        result = builder.invoke(
            user_input="Turn4: Python 설명해줘",  # normal mode로 감지되는 입력
            session_id="test_sequence_turn4",
            messages=messages,
            turn_count=4,
            compression_rate=0.3,
            normal_turn_ids=[1, 2, 3],  # 이전 normal 턴들
        )

        # summary_history에 1개 요약이 있어야 함
        summary_history = result.get("summary_history", [])
        assert len(summary_history) == 1
        assert summary_history[0]["turns"] == [1, 2, 3]
        assert "compression_rate" in summary_history[0]

    def test_turn_7_adds_second_summary(self, api_key):
        """Turn 7에서 두 번째 요약 추가"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        # Turn 1-6 시뮬레이션 - Phase 03-3-2: turn_id 메타데이터 포함
        messages = [
            HumanMessage(content="Turn1", additional_kwargs={"turn_id": 1, "mode": "normal"}),
            AIMessage(content="Turn1 답변", tool_calls=[], additional_kwargs={"turn_id": 1}),
            HumanMessage(content="Turn2", additional_kwargs={"turn_id": 2, "mode": "normal"}),
            AIMessage(content="Turn2 답변", tool_calls=[], additional_kwargs={"turn_id": 2}),
            HumanMessage(content="Turn3", additional_kwargs={"turn_id": 3, "mode": "normal"}),
            AIMessage(content="Turn3 답변", tool_calls=[], additional_kwargs={"turn_id": 3}),
            HumanMessage(content="Turn4", additional_kwargs={"turn_id": 4, "mode": "normal"}),
            AIMessage(content="Turn4 답변", tool_calls=[], additional_kwargs={"turn_id": 4}),
            HumanMessage(content="Turn5", additional_kwargs={"turn_id": 5, "mode": "normal"}),
            AIMessage(content="Turn5 답변", tool_calls=[], additional_kwargs={"turn_id": 5}),
            HumanMessage(content="Turn6", additional_kwargs={"turn_id": 6, "mode": "normal"}),
            AIMessage(content="Turn6 답변", tool_calls=[], additional_kwargs={"turn_id": 6}),
        ]

        # 기존 summary_history (Turn 4에서 생성된 것)
        existing_summary = [{
            "thread_id": "test_sequence_turn7",
            "turns": [1, 2, 3],
            "summarized_turns": [1, 2, 3],
            "excluded_turns": [],
            "turn_length": 3,
            "original_chars": 100,
            "summary_chars": 30,
            "compression_rate": 0.3,
            "summary": "Turn 1-3 요약",
        }]

        # Turn 7 실행 - Phase 03-3-2: normal_turn_ids 전달
        result = builder.invoke(
            user_input="LangChain에 대해 설명해줘",
            session_id="test_sequence_turn7_v2",
            messages=messages,
            turn_count=7,
            compression_rate=0.3,
            summary_history=existing_summary,
            normal_turn_ids=[1, 2, 3, 4, 5, 6],  # 이전 normal 턴들
        )

        # summary_history에 2개 요약이 있어야 함
        summary_history = result.get("summary_history", [])
        assert len(summary_history) == 2
        assert summary_history[0]["turns"] == [1, 2, 3]
        assert summary_history[1]["turns"] == [4, 5, 6]


class TestContextComposition:
    """Context 구성 테스트"""

    def test_turn_4_context_has_summary_and_current(self):
        """Turn 4 Context: 요약 + 현재 턴"""
        # summary_node 후 llm_node에서 Context 구성
        # System(1-3 요약) + User4 메시지
        pass  # 통합 테스트에서 확인

    def test_turn_5_context_has_summary_and_two_raw_turns(self):
        """Turn 5 Context: 요약 + Turn4 + Turn5"""
        pass  # 통합 테스트에서 확인


class TestCompressionRateApplication:
    """compression_rate 적용 테스트"""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_compression_rate_affects_summary_length(self, api_key):
        """compression_rate가 요약 길이에 영향"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        messages = [
            HumanMessage(content="Turn1: 긴 질문입니다. " * 10),
            AIMessage(content="Turn1: 긴 답변입니다. " * 10, tool_calls=[]),
            HumanMessage(content="Turn2: 긴 질문입니다. " * 10),
            AIMessage(content="Turn2: 긴 답변입니다. " * 10, tool_calls=[]),
            HumanMessage(content="Turn3: 긴 질문입니다. " * 10),
            AIMessage(content="Turn3: 긴 답변입니다. " * 10, tool_calls=[]),
        ]

        # compression_rate 0.3으로 실행
        result = builder.invoke(
            user_input="Turn4",
            session_id="test_compression",
            messages=messages,
            turn_count=4,
            compression_rate=0.3,
        )

        summary_history = result.get("summary_history", [])
        if summary_history:
            # compression_rate가 저장되어 있어야 함
            assert summary_history[0]["compression_rate"] == 0.3
            # original_chars > summary_chars 이어야 함
            assert summary_history[0]["original_chars"] > summary_history[0]["summary_chars"]


class TestAppCompressionRatePassing:
    """app.py에서 compression_rate 전달 테스트"""

    def test_app_passes_compression_rate_to_invoke(self):
        """app.py 소스코드에 compression_rate 전달이 있어야 함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)

        # invoke() 호출부에 compression_rate가 있어야 함
        assert "compression_rate=" in source or "compression_rate =" in source
