"""Phase 03-3: ReAct 그래프 테스트 (Tool Calling 패턴)"""
import pytest
from typing import Any


class TestChatState:
    """ChatState 스키마 테스트 (Phase 03-3: 단순화된 구조)"""

    def test_chat_state_import(self):
        """ChatState 임포트 테스트"""
        from service.react_graph import ChatState

        assert ChatState is not None

    def test_chat_state_has_messages_field(self):
        """ChatState에 messages 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "messages" in annotations

    def test_chat_state_has_session_id(self):
        """ChatState에 session_id 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "session_id" in annotations

    def test_chat_state_has_context_fields(self):
        """ChatState에 컨텍스트 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "summary" in annotations
        assert "pdf_description" in annotations

    def test_chat_state_has_summary_fields(self):
        """ChatState에 요약 관련 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "summary_history" in annotations
        assert "turn_count" in annotations

    def test_chat_state_has_token_fields(self):
        """ChatState에 토큰 관련 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "input_tokens" in annotations
        assert "output_tokens" in annotations


class TestReactGraphBuilder:
    """ReactGraphBuilder 클래스 테스트"""

    def test_import_react_graph_builder(self):
        """ReactGraphBuilder 임포트 테스트"""
        from service.react_graph import ReactGraphBuilder

        assert ReactGraphBuilder is not None

    def test_create_react_graph_builder(self):
        """ReactGraphBuilder 생성 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-api-key",
            model="gemini-2.0-flash",
        )

        assert builder is not None
        assert builder.api_key == "test-api-key"
        assert builder.model_name == "gemini-2.0-flash"

    def test_builder_has_max_iterations(self):
        """max_iterations 파라미터 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-api-key",
            max_iterations=10,
        )

        assert builder.max_iterations == 10

    def test_builder_default_max_iterations(self):
        """기본 max_iterations 값 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert builder.max_iterations == 5

    def test_build_returns_compiled_graph(self):
        """build()가 컴파일된 그래프를 반환하는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_builder_has_invoke_method(self):
        """invoke 메서드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert hasattr(builder, "invoke")
        assert callable(builder.invoke)


class TestGraphNodesPhase03:
    """Phase 03-3: 그래프 노드 테스트 (Tool Calling 패턴)"""

    def test_graph_has_summary_node(self):
        """summary_node가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "summary_node" in graph.nodes

    def test_graph_has_llm_node(self):
        """llm_node가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "llm_node" in graph.nodes

    def test_graph_has_tool_node(self):
        """tool_node가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "tool_node" in graph.nodes


class TestPhase05SqliteSaver:
    """Phase 02-5: SqliteSaver 테스트"""

    def test_builder_uses_sqlite_saver(self):
        """ReactGraphBuilder가 SqliteSaver를 사용하는지 테스트"""
        from service.react_graph import ReactGraphBuilder
        from langgraph.checkpoint.sqlite import SqliteSaver

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert isinstance(builder._checkpointer, SqliteSaver)

    def test_builder_db_path_parameter(self):
        """db_path 파라미터가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-api-key",
            db_path=":memory:",
        )

        assert builder.db_path == ":memory:"

    def test_builder_default_db_path(self):
        """기본 db_path가 설정되는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert builder.db_path is not None
        assert "langgraph" in builder.db_path or "data" in builder.db_path


class TestPhase05SummaryNode:
    """Phase 02-5: Summary Node 테스트"""

    def test_graph_has_summary_node(self):
        """그래프에 summary_node가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "summary_node" in graph.nodes

    def test_should_summarize_function_import(self):
        """should_summarize 함수 임포트 테스트"""
        from service.react_graph import should_summarize

        assert callable(should_summarize)

    def test_should_summarize_returns_false_for_early_turns(self):
        """초기 턴에서는 요약 안함 테스트"""
        from service.react_graph import should_summarize

        assert should_summarize(1) is False
        assert should_summarize(2) is False
        assert should_summarize(3) is False

    def test_should_summarize_returns_true_at_turn_4(self):
        """Turn 4에서 요약 트리거 테스트"""
        from service.react_graph import should_summarize

        assert should_summarize(4) is True

    def test_should_summarize_returns_true_at_turn_7(self):
        """Turn 7에서 요약 트리거 테스트"""
        from service.react_graph import should_summarize

        assert should_summarize(7) is True

    def test_should_summarize_returns_true_at_turn_10(self):
        """Turn 10에서 요약 트리거 테스트"""
        from service.react_graph import should_summarize

        assert should_summarize(10) is True

    def test_should_summarize_returns_false_at_turn_5(self):
        """Turn 5에서는 요약 안함 테스트"""
        from service.react_graph import should_summarize

        assert should_summarize(5) is False


class TestPhase05GraphStructure:
    """Phase 03-3: 그래프 구조 테스트"""

    def test_graph_starts_with_summary_node(self):
        """그래프가 summary_node로 시작하는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        start_edges = graph.get_graph().edges
        start_connections = [e for e in start_edges if e[0] == "__start__"]

        assert len(start_connections) > 0
        assert start_connections[0][1] == "summary_node"

    def test_summary_node_connects_to_llm_node(self):
        """summary_node가 llm_node로 연결되는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        edges = graph.get_graph().edges
        summary_edges = [e for e in edges if e[0] == "summary_node"]

        assert len(summary_edges) > 0
        assert summary_edges[0][1] == "llm_node"


class TestPhase07TokenUsage:
    """Phase 02-7: Token Usage 추적 테스트"""

    def test_chat_state_has_token_fields(self):
        """ChatState에 토큰 관련 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "input_tokens" in annotations
        assert "output_tokens" in annotations

    def test_builder_has_token_tracking_method(self):
        """ReactGraphBuilder에 토큰 추적 메서드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert hasattr(builder, "_invoke_llm_with_token_tracking")
        assert callable(builder._invoke_llm_with_token_tracking)

    def test_invoke_returns_token_info(self):
        """invoke() 반환값에 토큰 정보가 있는지 테스트 (Mock)"""
        from unittest.mock import patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        mock_result = {
            "messages": [],
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            # normal 모드로 감지되는 입력 사용 (casual은 그래프 스킵)
            result = builder.invoke(
                user_input="LangChain에 대해 설명해줘",
                session_id="test-session",
            )

            assert "input_tokens" in result
            assert "output_tokens" in result
            assert result["input_tokens"] >= 0
            assert result["output_tokens"] >= 0


class TestPhase07FastPath:
    """Phase 02-7: Fast-path 최적화 테스트"""

    def test_casual_conversation_returns_is_casual_flag(self):
        """일상적 대화에서 is_casual 플래그가 반환되는지 테스트"""
        from unittest.mock import patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        with patch.object(
            builder,
            "_invoke_llm_with_token_tracking",
            return_value=("네, 궁금한 점 있으시면 말씀해주세요!", 10, 5)
        ):
            result = builder.invoke(
                user_input="오호",
                session_id="test-session",
            )

            assert result.get("is_casual") is True
            assert result.get("tool_history") == []

    def test_normal_question_does_not_use_fast_path(self):
        """일반 질문에서는 fast-path를 사용하지 않는지 테스트"""
        from unittest.mock import patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        mock_result = {
            "messages": [],
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                user_input="파이썬 3.13 새 기능 검색해줘",
                session_id="test-session",
            )

            assert result.get("is_casual") is not True
