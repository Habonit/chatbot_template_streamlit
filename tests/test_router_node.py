"""Router Node 통합 테스트

router_node, casual_node, _route_by_mode, ChatState 필드,
그래프 구조 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestChatStateRouterFields:
    """ChatState에 mode/is_casual 필드 존재 확인"""

    def test_chat_state_has_mode(self):
        """ChatState에 mode 필드 존재"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert "mode" in hints

    def test_chat_state_has_is_casual(self):
        """ChatState에 is_casual 필드 존재"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert "is_casual" in hints

    def test_chat_state_mode_type(self):
        """mode 타입은 str"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert hints["mode"] is str

    def test_chat_state_is_casual_type(self):
        """is_casual 타입은 bool"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert hints["is_casual"] is bool


class TestRouterNode:
    """_router_node 메서드 테스트"""

    def _make_state(self, user_input="안녕", turn_count=1, normal_turn_ids=None):
        """테스트용 state 생성"""
        return {
            "messages": [HumanMessage(content=user_input)],
            "turn_count": turn_count,
            "normal_turn_ids": normal_turn_ids or [],
            "normal_turn_count": len(normal_turn_ids or []),
        }

    def test_router_node_casual(self):
        """casual 입력 → mode='casual', is_casual=True"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = self._make_state("안녕!", turn_count=1)

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="casual"):
            result = builder._router_node(state)

        assert result["mode"] == "casual"
        assert result["is_casual"] is True

    def test_router_node_normal(self):
        """normal 입력 → mode='normal', is_casual=False"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = self._make_state("지금 몇 시야?", turn_count=1)

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="normal"):
            result = builder._router_node(state)

        assert result["mode"] == "normal"
        assert result["is_casual"] is False

    def test_router_node_graph_path(self):
        """router_node 결과에 graph_path=['router_node'] 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = self._make_state("안녕", turn_count=1)

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="casual"):
            result = builder._router_node(state)

        assert result["graph_path"] == ["router_node"]

    def test_router_node_normal_updates_turn_ids(self):
        """normal 모드에서 normal_turn_ids에 turn_count 추가"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = self._make_state("설명해줘", turn_count=3, normal_turn_ids=[1, 2])

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="normal"):
            result = builder._router_node(state)

        assert result["normal_turn_ids"] == [1, 2, 3]

    def test_router_node_casual_preserves_turn_ids(self):
        """casual 모드에서 normal_turn_ids 변경 없음"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = self._make_state("안녕", turn_count=3, normal_turn_ids=[1, 2])

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="casual"):
            result = builder._router_node(state)

        assert result["normal_turn_ids"] == [1, 2]


class TestCasualNode:
    """_casual_node 메서드 테스트"""

    def test_casual_node_returns_ai_message(self):
        """casual_node가 AIMessage를 반환"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "안녕하세요!"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        state = {
            "messages": [HumanMessage(content="안녕")],
            "summary_history": [],
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        with patch.object(ChatGoogleGenerativeAI, "invoke", return_value=mock_response):
            result = builder._casual_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response

    def test_casual_node_graph_path(self):
        """casual_node가 graph_path에 casual_node 추가"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "안녕하세요!"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        state = {
            "messages": [HumanMessage(content="안녕")],
            "summary_history": [],
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        with patch.object(ChatGoogleGenerativeAI, "invoke", return_value=mock_response):
            result = builder._casual_node(state)

        assert result["graph_path"] == ["router_node", "casual_node"]

    def test_casual_node_token_tracking(self):
        """casual_node가 토큰 사용량 추적"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "네!"
        mock_response.usage_metadata = {"input_tokens": 15, "output_tokens": 8}

        state = {
            "messages": [HumanMessage(content="고마워")],
            "summary_history": [],
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        with patch.object(ChatGoogleGenerativeAI, "invoke", return_value=mock_response):
            result = builder._casual_node(state)

        assert result["input_tokens"] == 15
        assert result["output_tokens"] == 8

    def test_casual_node_uses_summary_history(self):
        """casual_node가 summary_history를 컨텍스트에 포함"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "네!"
        mock_response.usage_metadata = None

        summary_history = [
            {"turns": [1, 2, 3], "summary": "파이썬 대화 요약"}
        ]

        state = {
            "messages": [HumanMessage(content="고마워")],
            "summary_history": summary_history,
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        captured_args = {}

        def capture_invoke(self_llm, msgs, **kwargs):
            captured_args["messages"] = msgs
            return mock_response

        with patch.object(ChatGoogleGenerativeAI, "invoke", capture_invoke):
            builder._casual_node(state)

        sent_messages = captured_args["messages"]
        system_msgs = [m for m in sent_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 1
        assert "파이썬 대화 요약" in system_msgs[0].content


class TestRouteByMode:
    """_route_by_mode 메서드 테스트"""

    def test_route_casual(self):
        """is_casual=True → 'casual_node'"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = {"is_casual": True}
        assert builder._route_by_mode(state) == "casual_node"

    def test_route_normal(self):
        """is_casual=False → 'llm_node'"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = {"is_casual": False}
        assert builder._route_by_mode(state) == "llm_node"


class TestGraphStructure:
    """빌드된 그래프 구조 테스트"""

    def test_graph_has_router_node(self):
        """그래프에 router_node 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        node_names = list(builder._graph.get_graph().nodes.keys())
        assert "router_node" in node_names

    def test_graph_has_casual_node(self):
        """그래프에 casual_node 존재"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        node_names = list(builder._graph.get_graph().nodes.keys())
        assert "casual_node" in node_names

    def test_casual_node_uses_recent_3_turns_only(self):
        """casual_node가 전체 히스토리 대신 최근 3턴만 사용"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "안녕!"
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        # 5턴 분량의 히스토리 + 현재 입력
        messages = [
            HumanMessage(content="턴1 질문"), AIMessage(content="턴1 답변"),
            HumanMessage(content="턴2 질문"), AIMessage(content="턴2 답변"),
            HumanMessage(content="턴3 질문"), AIMessage(content="턴3 답변"),
            HumanMessage(content="턴4 질문"), AIMessage(content="턴4 답변"),
            HumanMessage(content="턴5 질문"), AIMessage(content="턴5 답변"),
            HumanMessage(content="안녕"),  # 현재 입력
        ]

        state = {
            "messages": messages,
            "summary_history": [],
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        captured_args = {}

        def capture_invoke(self_llm, msgs, **kwargs):
            captured_args["messages"] = msgs
            return mock_response

        with patch.object(ChatGoogleGenerativeAI, "invoke", capture_invoke):
            builder._casual_node(state)

        sent_messages = captured_args["messages"]
        # 최근 3턴(6개 메시지) + casual 프롬프트(1개) = 7개 이하
        human_msgs = [m for m in sent_messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in sent_messages if isinstance(m, AIMessage)]

        # 턴1, 턴2는 포함되지 않아야 함
        all_content = " ".join(m.content for m in sent_messages if hasattr(m, "content"))
        assert "턴1 질문" not in all_content
        assert "턴2 질문" not in all_content
        # 최근 3턴(턴3~5)은 포함되어야 함
        assert "턴3 질문" in all_content
        assert "턴5 답변" in all_content

    def test_casual_node_windowing_with_summary(self):
        """casual_node가 요약문 + 최근 3턴으로 컨텍스트 구성"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "네!"
        mock_response.usage_metadata = None

        summary_history = [
            {"turns": [1, 2, 3], "summary": "이전 3턴 요약 내용"}
        ]

        # 4턴 히스토리 + 현재 입력
        messages = [
            HumanMessage(content="턴1"), AIMessage(content="답1"),
            HumanMessage(content="턴2"), AIMessage(content="답2"),
            HumanMessage(content="턴3"), AIMessage(content="답3"),
            HumanMessage(content="턴4"), AIMessage(content="답4"),
            HumanMessage(content="지금 입력"),
        ]

        state = {
            "messages": messages,
            "summary_history": summary_history,
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        captured_args = {}

        def capture_invoke(self_llm, msgs, **kwargs):
            captured_args["messages"] = msgs
            return mock_response

        with patch.object(ChatGoogleGenerativeAI, "invoke", capture_invoke):
            builder._casual_node(state)

        sent_messages = captured_args["messages"]

        # SystemMessage에 요약문 포함
        system_msgs = [m for m in sent_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 1
        assert "이전 3턴 요약 내용" in system_msgs[0].content

        # 턴1은 제외, 최근 3턴(턴2~4)만 포함
        all_content = " ".join(m.content for m in sent_messages if hasattr(m, "content"))
        assert "턴1" not in all_content or "턴1" in system_msgs[0].content  # 요약에는 있을 수 있음
        assert "턴2" in all_content
        assert "턴4" in all_content

    def test_casual_node_short_history_no_truncation(self):
        """히스토리가 3턴 이하면 전부 포함"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = "좋아요!"
        mock_response.usage_metadata = None

        # 2턴 히스토리 + 현재 입력
        messages = [
            HumanMessage(content="첫번째"), AIMessage(content="답변1"),
            HumanMessage(content="두번째"), AIMessage(content="답변2"),
            HumanMessage(content="안녕"),
        ]

        state = {
            "messages": messages,
            "summary_history": [],
            "graph_path": ["router_node"],
            "input_tokens": 0,
            "output_tokens": 0,
        }

        captured_args = {}

        def capture_invoke(self_llm, msgs, **kwargs):
            captured_args["messages"] = msgs
            return mock_response

        with patch.object(ChatGoogleGenerativeAI, "invoke", capture_invoke):
            builder._casual_node(state)

        sent_messages = captured_args["messages"]
        all_content = " ".join(m.content for m in sent_messages if hasattr(m, "content"))
        # 2턴밖에 없으므로 전부 포함
        assert "첫번째" in all_content
        assert "답변2" in all_content

    def test_graph_start_goes_to_summary(self):
        """START → summary_node 연결 확인"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        graph = builder.build()

        # 그래프의 엣지를 통해 START → summary_node 확인
        graph_data = graph.get_graph()
        edges = [(e.source, e.target) for e in graph_data.edges]
        assert ("__start__", "summary_node") in edges
