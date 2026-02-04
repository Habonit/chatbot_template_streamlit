"""Phase 02-4: ReAct 그래프 테스트 (TDD)"""
import pytest
from typing import Any


class TestChatState:
    """ChatState 스키마 테스트"""

    def test_chat_state_import(self):
        """ChatState 임포트 테스트"""
        from service.react_graph import ChatState

        assert ChatState is not None

    def test_chat_state_has_input_fields(self):
        """ChatState에 입력 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "user_input" in annotations
        assert "session_id" in annotations

    def test_chat_state_has_context_fields(self):
        """ChatState에 컨텍스트 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "messages" in annotations
        assert "summary" in annotations
        assert "pdf_description" in annotations

    def test_chat_state_has_react_loop_fields(self):
        """ChatState에 ReAct 루프 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "current_tool" in annotations
        assert "tool_history" in annotations
        assert "iteration" in annotations
        assert "max_iterations" in annotations

    def test_chat_state_has_tool_result_fields(self):
        """ChatState에 툴 결과 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "tool_results" in annotations
        assert "needs_more_tools" in annotations
        assert "processor_summary" in annotations

    def test_chat_state_has_output_fields(self):
        """ChatState에 출력 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "final_response" in annotations


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
        # CompiledGraph 타입 체크
        assert hasattr(graph, "invoke")

    def test_builder_has_invoke_method(self):
        """invoke 메서드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")

        assert hasattr(builder, "invoke")
        assert callable(builder.invoke)


class TestGraphNodes:
    """그래프 노드 테스트"""

    def test_graph_has_tool_selector_node(self):
        """tool_selector 노드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        # 노드 이름 확인
        assert "tool_selector" in graph.nodes

    def test_graph_has_tool_nodes(self):
        """툴 노드들이 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "time_tool" in graph.nodes
        assert "reasoning_tool" in graph.nodes

    def test_graph_has_result_processor_node(self):
        """result_processor 노드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "result_processor" in graph.nodes

    def test_graph_has_response_generator_node(self):
        """response_generator 노드가 있는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        assert "response_generator" in graph.nodes


class TestRoutingFunctions:
    """라우팅 함수 테스트"""

    def test_route_to_selected_tool_import(self):
        """route_to_selected_tool 함수 임포트 테스트"""
        from service.react_graph import route_to_selected_tool

        assert callable(route_to_selected_tool)

    def test_route_to_selected_tool_returns_tool_name(self):
        """route_to_selected_tool이 툴 이름을 반환하는지 테스트"""
        from service.react_graph import route_to_selected_tool

        state = {"current_tool": "web_search"}
        result = route_to_selected_tool(state)

        assert result == "web_search"

    def test_route_to_selected_tool_returns_none(self):
        """route_to_selected_tool이 none을 반환하는지 테스트"""
        from service.react_graph import route_to_selected_tool

        state = {"current_tool": "none"}
        result = route_to_selected_tool(state)

        assert result == "none"

    def test_should_continue_loop_import(self):
        """should_continue_loop 함수 임포트 테스트"""
        from service.react_graph import should_continue_loop

        assert callable(should_continue_loop)

    def test_should_continue_loop_returns_continue(self):
        """should_continue_loop이 continue를 반환하는지 테스트"""
        from service.react_graph import should_continue_loop

        state = {
            "needs_more_tools": True,
            "iteration": 1,
            "max_iterations": 5,
        }
        result = should_continue_loop(state)

        assert result == "continue"

    def test_should_continue_loop_returns_finish(self):
        """should_continue_loop이 finish를 반환하는지 테스트"""
        from service.react_graph import should_continue_loop

        state = {
            "needs_more_tools": False,
            "iteration": 2,
            "max_iterations": 5,
        }
        result = should_continue_loop(state)

        assert result == "finish"

    def test_should_continue_loop_respects_max_iterations(self):
        """should_continue_loop이 max_iterations를 존중하는지 테스트"""
        from service.react_graph import should_continue_loop

        state = {
            "needs_more_tools": True,
            "iteration": 5,
            "max_iterations": 5,
        }
        result = should_continue_loop(state)

        assert result == "finish"


class TestPhase05ChatStateExtension:
    """Phase 02-5: ChatState 확장 테스트"""

    def test_chat_state_has_turn_count(self):
        """ChatState에 turn_count 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "turn_count" in annotations

    def test_chat_state_has_summary_history(self):
        """ChatState에 summary_history 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "summary_history" in annotations

    def test_chat_state_has_messages_for_context(self):
        """ChatState에 messages_for_context 필드가 있는지 테스트"""
        from service.react_graph import ChatState

        annotations = ChatState.__annotations__
        assert "messages_for_context" in annotations


class TestPhase05SqliteSaver:
    """Phase 02-5: SqliteSaver 테스트"""

    def test_builder_uses_sqlite_saver(self):
        """ReactGraphBuilder가 SqliteSaver를 사용하는지 테스트"""
        from service.react_graph import ReactGraphBuilder
        from langgraph.checkpoint.sqlite import SqliteSaver

        builder = ReactGraphBuilder(api_key="test-api-key")

        # _checkpointer가 SqliteSaver 타입인지 확인
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
    """Phase 02-5: 그래프 구조 테스트"""

    def test_graph_starts_with_summary_node(self):
        """그래프가 summary_node로 시작하는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        # START에서 첫 노드가 summary_node인지 확인
        # graph.nodes.__start__의 연결 확인
        start_edges = graph.get_graph().edges
        start_connections = [e for e in start_edges if e[0] == "__start__"]

        assert len(start_connections) > 0
        assert start_connections[0][1] == "summary_node"

    def test_summary_node_connects_to_tool_selector(self):
        """summary_node가 tool_selector로 연결되는지 테스트"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        graph = builder.build()

        edges = graph.get_graph().edges
        summary_edges = [e for e in edges if e[0] == "summary_node"]

        assert len(summary_edges) > 0
        assert summary_edges[0][1] == "tool_selector"


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
        from unittest.mock import MagicMock, patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        # Mock the graph invoke to return expected structure
        mock_result = {
            "final_response": "테스트 응답",
            "tool_history": [],
            "tool_results": {},
            "iteration": 0,
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                user_input="테스트 질문",
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
        from unittest.mock import MagicMock, patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        # Mock _invoke_llm_with_token_tracking
        with patch.object(
            builder,
            "_invoke_llm_with_token_tracking",
            return_value=("네, 궁금한 점 있으시면 말씀해주세요!", 10, 5)
        ):
            result = builder.invoke(
                user_input="오호",
                session_id="test-session",
            )

            # is_casual 플래그가 True여야 함
            assert result.get("is_casual") is True
            # tool_history가 비어있어야 함
            assert result.get("tool_history") == []

    def test_normal_question_does_not_use_fast_path(self):
        """일반 질문에서는 fast-path를 사용하지 않는지 테스트"""
        from unittest.mock import MagicMock, patch
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-api-key")
        builder.build()

        # Mock the graph invoke
        mock_result = {
            "final_response": "응답 내용",
            "tool_history": ["web_search"],
            "tool_results": {},
            "iteration": 1,
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

            # is_casual 플래그가 없거나 False여야 함
            assert result.get("is_casual") is not True
