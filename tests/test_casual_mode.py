"""Phase 03-3-2: Casual Mode 테스트

TDD: 테스트 먼저 작성
- casual 모드 턴 카운트 제외
- normal_turn_ids 관리
- tool_history 현재 턴만 추출
- should_summarize Fallback
"""
import os
import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


class TestMessageTurnIdMetadata:
    """메시지에 turn_id 메타데이터 추가 테스트"""

    def test_human_message_has_turn_id(self):
        """HumanMessage에 turn_id 포함"""
        msg = HumanMessage(
            content="테스트",
            additional_kwargs={"turn_id": 1, "mode": "normal"}
        )
        assert msg.additional_kwargs["turn_id"] == 1
        assert msg.additional_kwargs["mode"] == "normal"

    def test_ai_message_can_have_turn_id(self):
        """AIMessage에도 turn_id 추가 가능"""
        msg = AIMessage(
            content="응답",
            additional_kwargs={"turn_id": 1},
            tool_calls=[]
        )
        assert msg.additional_kwargs["turn_id"] == 1


class TestExtractMessagesByTurnIds:
    """turn_id 기반 메시지 추출 테스트"""

    def test_function_exists(self):
        """extract_messages_by_turn_ids 함수 존재"""
        from service.react_graph import extract_messages_by_turn_ids
        assert callable(extract_messages_by_turn_ids)

    def test_extract_single_turn(self):
        """단일 턴 추출"""
        from service.react_graph import extract_messages_by_turn_ids

        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(content="A1", additional_kwargs={"turn_id": 1}),
            HumanMessage(content="Q2", additional_kwargs={"turn_id": 2}),
            AIMessage(content="A2", additional_kwargs={"turn_id": 2}),
        ]
        result = extract_messages_by_turn_ids(messages, [1])
        assert len(result) == 2
        assert result[0].content == "Q1"
        assert result[1].content == "A1"

    def test_extract_non_consecutive_turns(self):
        """비연속 턴 추출 [1, 3]"""
        from service.react_graph import extract_messages_by_turn_ids

        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(content="A1", additional_kwargs={"turn_id": 1}),
            HumanMessage(content="Q2", additional_kwargs={"turn_id": 2}),
            AIMessage(content="A2", additional_kwargs={"turn_id": 2}),
            HumanMessage(content="Q3", additional_kwargs={"turn_id": 3}),
            AIMessage(content="A3", additional_kwargs={"turn_id": 3}),
        ]
        result = extract_messages_by_turn_ids(messages, [1, 3])
        assert len(result) == 4
        assert result[0].content == "Q1"
        assert result[2].content == "Q3"

    def test_extract_empty_turn_ids(self):
        """빈 turn_ids는 빈 리스트 반환"""
        from service.react_graph import extract_messages_by_turn_ids

        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
        ]
        result = extract_messages_by_turn_ids(messages, [])
        assert result == []

    def test_extract_message_without_turn_id(self):
        """turn_id가 없는 메시지는 제외"""
        from service.react_graph import extract_messages_by_turn_ids

        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(content="A1"),  # turn_id 없음
            HumanMessage(content="Q2", additional_kwargs={"turn_id": 2}),
        ]
        result = extract_messages_by_turn_ids(messages, [1, 2])
        assert len(result) == 2
        assert result[0].content == "Q1"
        assert result[1].content == "Q2"


class TestShouldSummarizeWithFallback:
    """요약 트리거 조건 테스트 (Fallback 포함)"""

    def test_function_signature_changed(self):
        """should_summarize가 2개 파라미터 받음"""
        from service.react_graph import should_summarize
        import inspect

        sig = inspect.signature(should_summarize)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, "should_summarize는 normal_turn_count, total_turn_count 필요"

    def test_normal_trigger_at_4(self):
        """normal_count=4에서 트리거"""
        from service.react_graph import should_summarize
        assert should_summarize(4, 4) is True

    def test_normal_trigger_at_7(self):
        """normal_count=7에서 트리거"""
        from service.react_graph import should_summarize
        assert should_summarize(7, 7) is True

    def test_no_trigger_at_3(self):
        """normal_count=3에서 트리거 안함"""
        from service.react_graph import should_summarize
        assert should_summarize(3, 3) is False

    def test_no_trigger_at_5(self):
        """normal_count=5에서 트리거 안함"""
        from service.react_graph import should_summarize
        assert should_summarize(5, 5) is False

    def test_fallback_trigger_at_total_10(self):
        """total=10에서 Fallback 트리거 (normal이 적어도)"""
        from service.react_graph import should_summarize
        assert should_summarize(1, 10) is True

    def test_fallback_trigger_at_total_20(self):
        """total=20에서 Fallback 트리거"""
        from service.react_graph import should_summarize
        assert should_summarize(2, 20) is True

    def test_no_fallback_at_total_9(self):
        """total=9에서 Fallback 트리거 안함"""
        from service.react_graph import should_summarize
        # normal_count=1이므로 기본 조건도 안됨
        assert should_summarize(1, 9) is False


class TestChatStateNormalTurnIds:
    """ChatState normal_turn_ids 필드 테스트"""

    def test_chat_state_has_normal_turn_ids(self):
        """ChatState에 normal_turn_ids 필드 존재"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "normal_turn_ids" in annotations

    def test_chat_state_has_normal_turn_count(self):
        """ChatState에 normal_turn_count 필드 존재"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "normal_turn_count" in annotations


class TestInvokeNormalTurnIds:
    """invoke() normal_turn_ids 파라미터 테스트"""

    def test_invoke_accepts_normal_turn_ids(self):
        """invoke가 normal_turn_ids 파라미터 받음"""
        from service.react_graph import ReactGraphBuilder
        import inspect

        sig = inspect.signature(ReactGraphBuilder.invoke)
        params = list(sig.parameters.keys())
        assert "normal_turn_ids" in params

    def test_invoke_returns_normal_turn_ids(self):
        """invoke 반환값에 normal_turn_ids 포함"""
        from service.react_graph import ReactGraphBuilder

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="LangChain에 대해 설명해줘",  # normal
            session_id="test_normal_turn_ids",
            turn_count=1,
            normal_turn_ids=[],
        )

        assert "normal_turn_ids" in result


class TestCasualModeIntegration:
    """Casual 모드 통합 테스트"""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_casual_does_not_change_normal_turn_ids(self, api_key):
        """casual 입력이 normal_turn_ids를 변경하지 않음"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="안녕",  # casual
            session_id="test_casual_unchanged",
            turn_count=2,
            normal_turn_ids=[1],
        )

        assert result["normal_turn_ids"] == [1]
        assert result["is_casual"] is True

    def test_normal_appends_to_normal_turn_ids(self, api_key):
        """normal 입력이 normal_turn_ids에 추가됨"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="Docker 컨테이너를 만드는 방법을 설명해줘",  # normal
            session_id="test_normal_appends",
            turn_count=3,
            normal_turn_ids=[1],
        )

        assert result["normal_turn_ids"] == [1, 3]

    def test_casual_returns_normal_turn_count(self, api_key):
        """casual 반환값에 normal_turn_count 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="고마워",  # casual
            session_id="test_casual_count",
            turn_count=3,
            normal_turn_ids=[1, 2],
        )

        assert result["normal_turn_count"] == 2  # 변화 없음


class TestToolHistoryCurrentTurnOnly:
    """tool_history 현재 턴만 추출 테스트"""

    @pytest.fixture
    def api_key(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_tool_history_empty_when_no_tools(self, api_key):
        """도구 미사용 시 tool_history 빈 배열"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        # casual은 도구 사용 안함
        result = builder.invoke(
            user_input="안녕",
            session_id="test_no_tools",
            turn_count=1,
        )

        assert result["tool_history"] == []

    def test_tool_history_not_accumulated(self, api_key):
        """이전 턴 도구가 현재 턴에 누적되지 않음"""
        from service.react_graph import ReactGraphBuilder
        from langchain_core.messages import HumanMessage, AIMessage

        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        # Turn 1: 도구 사용했다고 가정하는 메시지
        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": "1"}],
                additional_kwargs={"turn_id": 1}
            ),
        ]

        # Turn 2: 도구 사용 안함
        result = builder.invoke(
            user_input="고마워",  # casual, 도구 사용 안함
            session_id="test_not_accumulated",
            messages=messages,
            turn_count=2,
        )

        # Turn 2에서는 도구 사용 안했으므로 빈 배열
        assert result["tool_history"] == []


class TestSummaryHistoryStructure:
    """summary_history 구조 테스트"""

    def test_summary_has_summarized_turns(self):
        """summary_history에 summarized_turns 필드 존재"""
        # 예상 구조
        expected_fields = [
            "turns",
            "summarized_turns",
            "excluded_turns",
        ]

        sample_summary = {
            "thread_id": "session_123",
            "turns": [1, 2, 3],
            "summarized_turns": [1, 3],
            "excluded_turns": [2],
            "turn_length": 2,
            "original_chars": 500,
            "summary_chars": 150,
            "compression_rate": 0.3,
            "summary": "요약 내용",
        }

        for field in expected_fields:
            assert field in sample_summary


class TestAppNormalTurnIds:
    """app.py normal_turn_ids 관리 테스트"""

    def test_app_has_normal_turn_ids_initialization(self):
        """app.py에 normal_turn_ids 초기화 코드 존재"""
        import inspect
        import app

        source = inspect.getsource(app.init_session_state)
        assert "normal_turn_ids" in source

    def test_app_passes_normal_turn_ids_to_invoke(self):
        """app.py에서 invoke 호출 시 normal_turn_ids 전달"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)
        assert "normal_turn_ids" in source
