"""Phase 04: 교육용 메타데이터 테스트

react_graph.py, app.py의 메타데이터 추적 테스트:
- ChatState에 graph_path, summary_triggered 추가
- _summary_node, _llm_node에서 메타데이터 기록
- invoke, stream에서 메타데이터 반환
- app.py에서 Message에 메타데이터 포함
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestChatState:
    """ChatState에 메타데이터 필드 존재 확인"""

    def test_chat_state_has_graph_path(self):
        """ChatState에 graph_path 필드 존재"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert "graph_path" in hints

    def test_chat_state_has_summary_triggered(self):
        """ChatState에 summary_triggered 필드 존재"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert "summary_triggered" in hints

    def test_chat_state_graph_path_type(self):
        """graph_path 타입은 list[str]"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert hints["graph_path"] == list[str]

    def test_chat_state_summary_triggered_type(self):
        """summary_triggered 타입은 bool"""
        from service.react_graph import ChatState
        import typing
        hints = typing.get_type_hints(ChatState)
        assert hints["summary_triggered"] is bool


class TestSummaryNodeMetadata:
    """_summary_node 메타데이터 추적 테스트"""

    def test_summary_node_no_summary_returns_metadata(self):
        """요약 불필요 시 summary_triggered=False, graph_path 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = {
            "turn_count": 1,
            "normal_turn_count": 1,
            "normal_turn_ids": [1],
            "messages": [],
            "session_id": "test",
            "compression_rate": 0.3,
            "summary_history": [],
            "graph_path": [],
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result = builder._summary_node(state)
        assert result["summary_triggered"] is False
        assert "summary_node" in result["graph_path"]

    def test_summary_node_graph_path_appended(self):
        """graph_path에 summary_node 추가"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        state = {
            "turn_count": 2,
            "normal_turn_count": 2,
            "normal_turn_ids": [1, 2],
            "messages": [],
            "session_id": "test",
            "compression_rate": 0.3,
            "summary_history": [],
            "graph_path": [],
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result = builder._summary_node(state)
        assert result["graph_path"] == ["summary_node"]

    def test_summary_node_triggered_returns_true(self):
        """요약 실행 시 summary_triggered=True"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")

        # normal_turn_count=4 -> should_summarize = True
        messages = []
        for i in range(1, 5):
            messages.append(
                HumanMessage(content=f"질문 {i}", additional_kwargs={"turn_id": i})
            )
            messages.append(AIMessage(content=f"답변 {i}"))

        state = {
            "turn_count": 4,
            "normal_turn_count": 4,
            "normal_turn_ids": [1, 2, 3, 4],
            "messages": messages,
            "session_id": "test",
            "compression_rate": 0.3,
            "summary_history": [],
            "graph_path": [],
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("요약 내용", 50, 20)
        ):
            result = builder._summary_node(state)

        assert result["summary_triggered"] is True
        assert "summary_node" in result["graph_path"]

    def test_summary_node_no_turns_to_summarize(self):
        """요약할 턴이 없으면 summary_triggered=False"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        # should_summarize(4, 4) = True, but normal_turn_ids has only 1 entry
        state = {
            "turn_count": 4,
            "normal_turn_count": 4,
            "normal_turn_ids": [4],  # 하나만 있으면 turns_to_summarize = []
            "messages": [],
            "session_id": "test",
            "compression_rate": 0.3,
            "summary_history": [],
            "graph_path": [],
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result = builder._summary_node(state)
        assert result["summary_triggered"] is False
        assert "summary_node" in result["graph_path"]


class TestLlmNodeMetadata:
    """_llm_node 메타데이터 추적 테스트"""

    def test_llm_node_graph_path_includes_llm_node(self):
        """_llm_node가 graph_path에 llm_node 추가"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        mock_response = MagicMock()
        mock_response.content = "응답입니다"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        state = {
            "turn_count": 1,
            "summary_history": [],
            "messages": [HumanMessage(content="테스트")],
            "graph_path": ["summary_node"],
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "pdf_description": "",
        }

        with patch.object(ChatGoogleGenerativeAI, "invoke", return_value=mock_response):
            result = builder._llm_node(state)

        assert "llm_node" in result["graph_path"]
        assert result["graph_path"] == ["summary_node", "llm_node"]

    def test_llm_node_appends_to_existing_path(self):
        """기존 graph_path에 llm_node가 추가됨"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        mock_response = MagicMock()
        mock_response.content = "결과"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        state = {
            "turn_count": 1,
            "summary_history": [],
            "messages": [HumanMessage(content="테스트")],
            "graph_path": ["summary_node", "llm_node"],  # 이미 한번 호출된 상태
            "summary_triggered": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "pdf_description": "",
        }

        with patch.object(ChatGoogleGenerativeAI, "invoke", return_value=mock_response):
            result = builder._llm_node(state)

        assert result["graph_path"] == ["summary_node", "llm_node", "llm_node"]


class TestPrepareInvocationMetadata:
    """_prepare_invocation 메타데이터 포함 테스트"""

    def test_prepare_invocation_includes_graph_path(self):
        """initial_state에 graph_path 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation(
            "파이썬 설명해줘", session_id="test", turn_count=1
        )
        _, state, _, _ = result
        assert "graph_path" in state
        assert state["graph_path"] == []

    def test_prepare_invocation_includes_summary_triggered(self):
        """initial_state에 summary_triggered 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        result = builder._prepare_invocation(
            "설명해줘", session_id="test", turn_count=1
        )
        _, state, _, _ = result
        assert "summary_triggered" in state
        assert state["summary_triggered"] is False


class TestInvokeMetadata:
    """invoke() 메타데이터 반환 테스트"""

    def test_invoke_normal_returns_mode(self):
        """normal invoke는 mode 반환"""
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
            "graph_path": ["summary_node", "llm_node"],
            "summary_triggered": False,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "파이썬에 대해 설명해줘", session_id="test", turn_count=1
            )

        assert result["mode"] == "normal"
        assert result["is_casual"] is False

    def test_invoke_normal_returns_graph_path(self):
        """normal invoke는 graph_path 반환"""
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
            "graph_path": ["summary_node", "llm_node"],
            "summary_triggered": False,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "파이썬에 대해 설명해줘", session_id="test", turn_count=1
            )

        assert "graph_path" in result
        assert "summary_node" in result["graph_path"]
        assert "llm_node" in result["graph_path"]

    def test_invoke_normal_returns_summary_triggered(self):
        """normal invoke는 summary_triggered 반환"""
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
            "graph_path": ["summary_node", "llm_node"],
            "summary_triggered": False,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "파이썬에 대해 설명해줘", session_id="test", turn_count=1
            )

        assert result["summary_triggered"] is False

    def test_invoke_casual_returns_casual_metadata(self):
        """casual invoke는 casual 메타데이터 반환"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("반가워요!", 10, 5)
        ):
            result = builder.invoke("안녕!", session_id="test", turn_count=1)

        assert result["mode"] == "casual"
        assert result["graph_path"] == ["casual_bypass"]
        assert result["summary_triggered"] is False
        assert result["is_casual"] is True

    def test_invoke_graph_path_enhanced_with_tool_node(self):
        """tool 사용 시 graph_path에 tool_node 추론 삽입"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        builder.build()

        ai_msg_with_tool = AIMessage(content="")
        ai_msg_with_tool.tool_calls = [{"name": "web_search", "args": {}, "id": "1"}]

        from langchain_core.messages import ToolMessage
        mock_result = {
            "messages": [
                HumanMessage(content="검색해줘", additional_kwargs={"turn_id": 1}),
                ai_msg_with_tool,
                ToolMessage(content="검색 결과", name="web_search", tool_call_id="1"),
                AIMessage(content="결과입니다"),
            ],
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
            # summary_node -> llm_node(1st) -> tool -> llm_node(2nd)
            "graph_path": ["summary_node", "llm_node", "llm_node"],
            "summary_triggered": False,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "최신 뉴스 검색해줘", session_id="test", turn_count=1
            )

        # tool_node가 2번째 llm_node 앞에 삽입되어야 함
        assert result["graph_path"] == [
            "summary_node", "llm_node", "tool_node", "llm_node"
        ]

    def test_invoke_no_tool_no_enhancement(self):
        """tool 미사용 시 graph_path 변환 없음"""
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
            "graph_path": ["summary_node", "llm_node"],
            "summary_triggered": False,
        }

        with patch.object(builder._graph, "invoke", return_value=mock_result):
            result = builder.invoke(
                "파이썬 설명해줘", session_id="test", turn_count=1
            )

        assert result["graph_path"] == ["summary_node", "llm_node"]


class TestInvokeCasualMetadata:
    """_invoke_casual 메타데이터 테스트"""

    def test_invoke_casual_has_mode(self):
        """_invoke_casual 반환에 mode='casual' 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("네!", 5, 3)
        ):
            result = builder._invoke_casual("안녕", "", [], [])

        assert result["mode"] == "casual"

    def test_invoke_casual_has_graph_path(self):
        """_invoke_casual 반환에 graph_path=['casual_bypass'] 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("네!", 5, 3)
        ):
            result = builder._invoke_casual("안녕", "", [], [])

        assert result["graph_path"] == ["casual_bypass"]

    def test_invoke_casual_has_summary_triggered_false(self):
        """_invoke_casual 반환에 summary_triggered=False 포함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(api_key="test-key")
        with patch.object(
            builder, "_invoke_llm_with_token_tracking",
            return_value=("네!", 5, 3)
        ):
            result = builder._invoke_casual("안녕", "", [], [])

        assert result["summary_triggered"] is False


class TestStreamMetadata:
    """stream() 메타데이터 테스트"""

    def test_stream_casual_done_has_mode(self):
        """casual 스트리밍 done 이벤트에 mode='casual' 포함"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk = MagicMock()
        mock_chunk.content = "네!"
        mock_chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk])):
            chunks = list(builder.stream("안녕", session_id="test"))

        done = chunks[-1]
        assert done["type"] == "done"
        meta = done["metadata"]
        assert meta["mode"] == "casual"
        assert meta["graph_path"] == ["casual_bypass"]
        assert meta["summary_triggered"] is False

    def test_stream_casual_done_has_graph_path(self):
        """casual stream done에 graph_path 포함"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk = MagicMock()
        mock_chunk.content = "안녕하세요"
        mock_chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk])):
            chunks = list(builder._stream_casual("안녕", "", [], []))

        done = chunks[-1]
        assert done["metadata"]["graph_path"] == ["casual_bypass"]

    def test_stream_done_metadata_has_required_keys(self):
        """stream done 이벤트에 Phase 04 메타데이터 키가 있는지"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(api_key="test-key")

        mock_chunk = MagicMock()
        mock_chunk.content = "네!"
        mock_chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3}

        with patch.object(ChatGoogleGenerativeAI, "stream", return_value=iter([mock_chunk])):
            chunks = list(builder.stream("안녕", session_id="test"))

        done = chunks[-1]
        metadata = done["metadata"]

        phase_04_keys = ["mode", "graph_path", "summary_triggered"]
        for key in phase_04_keys:
            assert key in metadata, f"Missing key: {key}"


class TestAppMetadata:
    """app.py Message 생성 메타데이터 포함 테스트"""

    @patch("app.st")
    def test_handle_chat_message_includes_metadata(self, mock_st):
        """handle_chat_message에서 Message에 메타데이터 포함"""
        from app import handle_chat_message
        from domain.message import Message
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.current_session = "test-session"
        mock_st.session_state.messages = []
        mock_st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_st.session_state.summary = ""
        mock_st.session_state.summary_history = []
        mock_st.session_state.pdf_description = ""
        mock_st.session_state.chunks = []
        mock_st.session_state.normal_turn_ids = []

        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {"gemini_api_key": "test-key", "thinking_budget": 1024}

        mock_result = {
            "text": "응답입니다",
            "tool_history": [],
            "tool_results": {},
            "iteration": 0,
            "model_used": "gemini-2.0-flash",
            "summary": "",
            "summary_history": [],
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "normal_turn_ids": [1],
            "normal_turn_count": 1,
            "mode": "normal",
            "graph_path": ["summary_node", "llm_node"],
            "summary_triggered": False,
            "is_casual": False,
            "error": None,
        }

        with patch("app._create_graph_builder") as mock_builder:
            mock_builder.return_value.invoke.return_value = mock_result
            handle_chat_message("설명해줘", settings, embed_repo)

        # messages에 추가된 마지막 메시지 확인
        assert len(mock_st.session_state.messages) == 2  # user + assistant
        assistant_msg = mock_st.session_state.messages[-1]
        assert isinstance(assistant_msg, Message)
        assert assistant_msg.mode == "normal"
        assert assistant_msg.graph_path == ["summary_node", "llm_node"]
        assert assistant_msg.summary_triggered is False
        assert assistant_msg.thinking_budget == 1024
        assert assistant_msg.is_casual is False

    @patch("app.st")
    def test_handle_chat_message_casual_metadata(self, mock_st):
        """casual 응답 시 Message에 casual 메타데이터 포함"""
        from app import handle_chat_message
        from domain.message import Message
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.current_session = "test-session"
        mock_st.session_state.messages = []
        mock_st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_st.session_state.summary = ""
        mock_st.session_state.summary_history = []
        mock_st.session_state.pdf_description = ""
        mock_st.session_state.chunks = []
        mock_st.session_state.normal_turn_ids = []

        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {"gemini_api_key": "test-key"}

        mock_result = {
            "text": "안녕하세요!",
            "tool_history": [],
            "tool_results": {},
            "iteration": 0,
            "model_used": "gemini-2.0-flash",
            "summary": "",
            "summary_history": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "normal_turn_ids": [],
            "normal_turn_count": 0,
            "mode": "casual",
            "graph_path": ["casual_bypass"],
            "summary_triggered": False,
            "is_casual": True,
            "error": None,
        }

        with patch("app._create_graph_builder") as mock_builder:
            mock_builder.return_value.invoke.return_value = mock_result
            handle_chat_message("안녕!", settings, embed_repo)

        assistant_msg = mock_st.session_state.messages[-1]
        assert assistant_msg.mode == "casual"
        assert assistant_msg.graph_path == ["casual_bypass"]
        assert assistant_msg.is_casual is True

    @patch("app.st")
    def test_handle_stream_message_includes_metadata(self, mock_st):
        """handle_stream_message에서 Message에 메타데이터 포함"""
        from app import handle_stream_message
        from domain.message import Message
        from repository.embedding_repo import EmbeddingRepository
        from pathlib import Path

        mock_st.session_state.current_session = "test-session"
        mock_st.session_state.messages = []
        mock_st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        mock_st.session_state.summary = ""
        mock_st.session_state.summary_history = []
        mock_st.session_state.pdf_description = ""
        mock_st.session_state.chunks = []
        mock_st.session_state.normal_turn_ids = []

        embed_repo = EmbeddingRepository(base_path=Path("data/sessions"))
        settings = {"gemini_api_key": "test-key", "thinking_budget": 2048}

        stream_chunks = [
            {"type": "token", "content": "응답"},
            {"type": "done", "metadata": {
                "text": "응답",
                "tool_history": [],
                "tool_results": {},
                "model_used": "gemini-2.0-flash",
                "summary": "",
                "summary_history": [],
                "input_tokens": 50,
                "output_tokens": 20,
                "normal_turn_ids": [1],
                "normal_turn_count": 1,
                "mode": "normal",
                "graph_path": ["summary_node", "llm_node"],
                "summary_triggered": False,
                "is_casual": False,
            }},
        ]

        with patch("app._create_graph_builder") as mock_builder:
            mock_builder.return_value.stream.return_value = iter(stream_chunks)
            chunks = list(handle_stream_message("설명해줘", settings, embed_repo))

        # messages에 추가된 마지막 메시지 확인
        assert len(mock_st.session_state.messages) == 2  # user + assistant
        assistant_msg = mock_st.session_state.messages[-1]
        assert isinstance(assistant_msg, Message)
        assert assistant_msg.mode == "normal"
        assert assistant_msg.graph_path == ["summary_node", "llm_node"]
        assert assistant_msg.summary_triggered is False
        assert assistant_msg.thinking_budget == 2048
        assert assistant_msg.is_casual is False

    def test_app_handle_chat_message_has_metadata_in_source(self):
        """handle_chat_message 소스코드에 Phase 04 메타데이터 필드 포함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)
        assert "mode=" in source
        assert "graph_path=" in source
        assert "summary_triggered=" in source
        assert "thought_process=" in source
        assert "thinking_budget=" in source
        assert "is_casual=" in source

    def test_app_handle_stream_message_has_metadata_in_source(self):
        """handle_stream_message 소스코드에 Phase 04 메타데이터 필드 포함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_stream_message)
        assert "mode=" in source
        assert "graph_path=" in source
        assert "summary_triggered=" in source
        assert "thought_process=" in source
        assert "thinking_budget=" in source
        assert "is_casual=" in source
