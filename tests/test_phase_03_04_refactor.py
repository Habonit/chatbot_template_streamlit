"""Phase 03-4: 리팩토링 테스트

invoke()에서 추출한 공통 메서드 테스트:
- _prepare_invocation: 공통 전처리
- _extract_current_turn_messages: 현재 턴 추출
- _parse_result: 공통 결과 파싱
- _invoke_casual: casual 모드 분리
- _create_graph_builder: 팩토리 함수
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TestPrepareInvocation:
    """_prepare_invocation 메서드 테스트"""

    def test_prepare_invocation_casual(self):
        """casual 입력 시 None 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation("안녕!", session_id="test")
        assert result is None

    def test_prepare_invocation_normal(self):
        """normal 입력 시 (mode, state, config, turn_ids) 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation(
            "지금 몇 시야?", session_id="test", turn_count=1
        )
        assert result is not None
        mode, state, config, turn_ids = result
        assert mode == "normal"
        assert isinstance(state, dict)
        assert isinstance(config, dict)
        assert turn_ids == [1]

    def test_prepare_invocation_normal_turn_ids(self):
        """normal 모드에서 turn_ids 업데이트 확인"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation(
            "검색해줘", session_id="test",
            turn_count=3, normal_turn_ids=[1, 2]
        )
        assert result is not None
        _, _, _, turn_ids = result
        assert turn_ids == [1, 2, 3]

    def test_prepare_invocation_builds_state(self):
        """normal 모드에서 initial_state 구성 확인"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation(
            "파이썬 설명해줘", session_id="test-session",
            turn_count=2, summary="이전 요약",
            pdf_description="PDF 설명",
        )
        _, state, config, _ = result
        assert state["session_id"] == "test-session"
        assert state["summary"] == "이전 요약"
        assert state["pdf_description"] == "PDF 설명"
        assert state["turn_count"] == 2
        assert config == {"configurable": {"thread_id": "test-session"}}

    def test_prepare_invocation_converts_messages(self):
        """domain.message.Message → LangChain BaseMessage 변환 확인"""
        from service.react_graph import ReactGraphBuilder
        from domain.message import Message

        builder = ReactGraphBuilder(api_key="test-key")
        messages = [
            Message(turn_id=1, role="user", content="Hello"),
            Message(turn_id=1, role="assistant", content="Hi there"),
        ]
        result = builder._prepare_invocation(
            "설명해줘", session_id="test",
            messages=messages, turn_count=2
        )
        _, state, _, _ = result
        # converted messages + user_message
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], HumanMessage)

    def test_prepare_invocation_builds_graph_if_needed(self):
        """그래프가 None이면 build() 호출"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        assert builder._graph is None
        builder._prepare_invocation("설명해줘", session_id="test")
        assert builder._graph is not None


class TestExtractCurrentTurnMessages:
    """_extract_current_turn_messages 메서드 테스트"""

    def test_extract_current_turn_messages(self):
        """turn_id 기반 추출 정확성"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        messages = [
            HumanMessage(content="이전 질문", additional_kwargs={"turn_id": 1}),
            AIMessage(content="이전 답변"),
            HumanMessage(content="현재 질문", additional_kwargs={"turn_id": 2}),
            AIMessage(content="현재 답변"),
        ]
        result = builder._extract_current_turn_messages(messages, turn_count=2)
        assert len(result) == 2
        assert result[0].content == "현재 질문"
        assert result[1].content == "현재 답변"

    def test_extract_current_turn_messages_fallback(self):
        """turn_id 없을 때 전체 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        messages = [
            HumanMessage(content="질문1"),
            AIMessage(content="답변1"),
        ]
        result = builder._extract_current_turn_messages(messages, turn_count=99)
        assert len(result) == 2  # 전체 반환


class TestParseResult:
    """_parse_result 메서드 테스트"""

    def test_parse_result_final_text(self):
        """최종 텍스트 추출"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        messages = [
            HumanMessage(content="질문", additional_kwargs={"turn_id": 1}),
            AIMessage(content="최종 답변"),
        ]
        result = builder._parse_result(messages, turn_count=1)
        assert result["text"] == "최종 답변"

    def test_parse_result_tool_history(self):
        """tool_history 추출"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "get_current_time", "args": {}, "id": "1"}]

        messages = [
            HumanMessage(content="몇 시야?", additional_kwargs={"turn_id": 1}),
            ai_msg,
            ToolMessage(content="14:30", name="get_current_time", tool_call_id="1"),
            AIMessage(content="현재 시간은 14:30입니다."),
        ]
        result = builder._parse_result(messages, turn_count=1)
        assert "get_current_time" in result["tool_history"]

    def test_parse_result_tool_results(self):
        """tool_results 추출"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"name": "web_search", "args": {}, "id": "1"}]

        messages = [
            HumanMessage(content="검색해줘", additional_kwargs={"turn_id": 1}),
            ai_msg,
            ToolMessage(content="검색 결과입니다", name="web_search", tool_call_id="1"),
            AIMessage(content="결과를 찾았습니다"),
        ]
        result = builder._parse_result(messages, turn_count=1)
        assert "web_search" in result["tool_results"]
        assert result["tool_results"]["web_search"] == "검색 결과입니다"

    def test_parse_result_empty_messages(self):
        """빈 메시지 처리"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._parse_result([], turn_count=1)
        assert result["text"] == ""
        assert result["tool_history"] == []
        assert result["tool_results"] == {}


class TestInvokeCasual:
    """_invoke_casual 메서드 테스트"""

    def test_invoke_casual_exists(self):
        """_invoke_casual 메서드 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        assert hasattr(builder, "_invoke_casual")
        assert callable(builder._invoke_casual)

    def test_invoke_casual_returns_dict(self):
        """casual 결과 반환 형태"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("안녕하세요!", 10, 5)
        ):
            result = builder._invoke_casual("안녕", "", [], [])
        assert result["text"] == "안녕하세요!"
        assert result["is_casual"] is True
        assert result["tool_history"] == []
        assert result["error"] is None

    def test_invoke_casual_preserves_normal_turn_ids(self):
        """casual은 normal_turn_ids 변경 없음"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("네!", 5, 3)
        ):
            result = builder._invoke_casual("네", "", [], [1, 2])
        assert result["normal_turn_ids"] == [1, 2]
        assert result["normal_turn_count"] == 2


class TestInvokeRefactored:
    """리팩토링된 invoke() 테스트"""

    def test_invoke_refactored_casual(self):
        """리팩토링 후 casual 동작 동일"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("반가워요!", 10, 5)
        ):
            result = builder.invoke("오호", session_id="test", turn_count=1)

        assert result["text"] == "반가워요!"
        assert result["is_casual"] is True
        assert result["error"] is None

    def test_invoke_refactored_normal(self):
        """리팩토링 후 normal 동작 동일 (mock)"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        mock_result = {
            "messages": [
                HumanMessage(content="설명해줘", additional_kwargs={"turn_id": 1}),
                AIMessage(content="설명입니다"),
            ],
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "파이썬에 대해 설명해줘", session_id="test", turn_count=1
            )

        assert "text" in result
        assert "tool_history" in result
        assert "normal_turn_ids" in result
        assert result["normal_turn_ids"] == [1]
        assert result["error"] is None


class TestCreateGraphBuilder:
    """_create_graph_builder 팩토리 테스트"""

    def test_create_graph_builder_exists(self):
        """_create_graph_builder 함수 존재"""
        from app import _create_graph_builder
        assert callable(_create_graph_builder)

    @patch("app.st")
    def test_create_graph_builder_returns_builder(self, mock_st):
        """팩토리가 ReactGraphBuilder 반환"""
        from app import _create_graph_builder
        from service.react_graph import ReactGraphBuilder
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.chunks = []
        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {"gemini_api_key": "test-key"}

        builder = _create_graph_builder(settings, embed_repo)
        assert isinstance(builder, ReactGraphBuilder)

    @patch("app.st")
    def test_create_graph_builder_with_search(self, mock_st):
        """tavily_api_key가 있으면 search_service 포함"""
        from app import _create_graph_builder
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.chunks = []
        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {
            "gemini_api_key": "test-key",
            "tavily_api_key": "tavily-test-key",
        }

        builder = _create_graph_builder(settings, embed_repo)
        assert builder.search_service is not None
