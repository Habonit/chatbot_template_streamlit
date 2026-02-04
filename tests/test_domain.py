import pytest
from datetime import datetime
import numpy as np


class TestMessage:
    def test_create_user_message(self):
        from domain.message import Message

        msg = Message(
            turn_id=1,
            role="user",
            content="Hello, world!",
        )

        assert msg.turn_id == 1
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.timestamp is not None
        assert msg.input_tokens == 0
        assert msg.output_tokens == 0
        assert msg.model_used is None

    def test_create_assistant_message_with_tokens(self):
        from domain.message import Message

        msg = Message(
            turn_id=1,
            role="assistant",
            content="Hi there!",
            input_tokens=10,
            output_tokens=5,
            model_used="gemini-2.5-flash",
        )

        assert msg.role == "assistant"
        assert msg.input_tokens == 10
        assert msg.output_tokens == 5
        assert msg.model_used == "gemini-2.5-flash"

    # === 항목 5: function_calls 필드 테스트 ===

    def test_message_default_function_calls(self):
        """function_calls 기본값 테스트 (빈 리스트)"""
        from domain.message import Message

        msg = Message(turn_id=1, role="user", content="Hello")

        assert msg.function_calls == []

    def test_message_with_function_calls(self):
        """function_calls가 포함된 메시지 생성 테스트"""
        from domain.message import Message

        function_calls = [
            {"name": "web_search", "args": {"query": "Python tutorial"}},
            {"name": "search_pdf_knowledge", "args": {"query": "AI", "top_k": 5}},
        ]
        msg = Message(
            turn_id=1,
            role="assistant",
            content="검색 결과입니다.",
            function_calls=function_calls,
        )

        assert len(msg.function_calls) == 2
        assert msg.function_calls[0]["name"] == "web_search"
        assert msg.function_calls[1]["args"]["top_k"] == 5

    # === Phase 02: tool_results 필드 테스트 ===

    def test_message_default_tool_results(self):
        """tool_results 기본값 테스트 (빈 딕셔너리)"""
        from domain.message import Message

        msg = Message(turn_id=1, role="user", content="Hello")

        assert msg.tool_results == {}

    def test_message_with_tool_results(self):
        """tool_results가 포함된 메시지 생성 테스트"""
        from domain.message import Message

        tool_results = {
            "web_search": "Python 3.13이 최신 버전입니다.",
            "get_current_time": "2026-02-04 15:30:45 (KST)",
        }
        msg = Message(
            turn_id=1,
            role="assistant",
            content="검색 결과입니다.",
            tool_results=tool_results,
        )

        assert len(msg.tool_results) == 2
        assert msg.tool_results["web_search"] == "Python 3.13이 최신 버전입니다."
        assert "(KST)" in msg.tool_results["get_current_time"]

class TestChunk:
    def test_create_chunk(self):
        from domain.chunk import Chunk

        chunk = Chunk(
            chunk_index=0,
            original_text="Original text here",
            normalized_text="Normalized text here",
            source_file="document.pdf",
            source_page=1,
            start_char=0,
            end_char=100,
        )

        assert chunk.chunk_index == 0
        assert chunk.original_text == "Original text here"
        assert chunk.normalized_text == "Normalized text here"
        assert chunk.source_file == "document.pdf"
        assert chunk.source_page == 1
        assert chunk.embedding is None

    def test_chunk_with_embedding(self):
        from domain.chunk import Chunk

        embedding = np.random.rand(768).astype(np.float32)
        chunk = Chunk(
            chunk_index=0,
            original_text="Text",
            normalized_text="Text",
            source_file="doc.pdf",
            source_page=1,
            start_char=0,
            end_char=10,
            embedding=embedding,
        )

        assert chunk.embedding is not None
        assert len(chunk.embedding) == 768

    def test_chunk_to_dict(self):
        from domain.chunk import Chunk

        chunk = Chunk(
            chunk_index=0,
            original_text="Text",
            normalized_text="Text",
            source_file="doc.pdf",
            source_page=1,
            start_char=0,
            end_char=10,
        )
        data = chunk.to_dict()

        assert "chunk_index" in data
        assert "metadata" in data
        assert data["metadata"]["source_file"] == "doc.pdf"

    def test_chunk_from_dict(self):
        from domain.chunk import Chunk

        data = {
            "chunk_index": 0,
            "original_text": "Text",
            "normalized_text": "Text",
            "embedding": None,
            "metadata": {
                "source_file": "doc.pdf",
                "source_page": 1,
                "start_char": 0,
                "end_char": 10,
            },
        }
        chunk = Chunk.from_dict(data)

        assert chunk.chunk_index == 0
        assert chunk.source_file == "doc.pdf"


class TestSession:
    def test_create_session(self):
        from domain.session import Session

        session = Session(session_id="202601151430")

        assert session.session_id == "202601151430"
        assert session.created_at is not None
        assert session.total_turns == 0
        assert session.current_summary == ""
        assert session.pdf_files == []

    def test_session_with_settings(self):
        from domain.session import Session

        settings = {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "top_p": 0.9,
        }
        session = Session(session_id="202601151430", settings=settings)

        assert session.settings["model"] == "gemini-2.5-flash"
        assert session.settings["temperature"] == 0.7

    def test_session_from_dict(self):
        from domain.session import Session

        data = {
            "session_id": "202601151430",
            "created_at": "2026-01-15T14:30:00",
            "last_updated": "2026-01-15T14:30:00",
            "total_turns": 5,
            "current_summary": "Summary",
            "pdf_files": ["doc.pdf"],
            "settings": {"model": "gemini-2.5-flash"},
        }
        session = Session.from_dict(data)

        assert session.session_id == "202601151430"
        assert session.total_turns == 5
        assert "doc.pdf" in session.pdf_files

    def test_generate_session_id(self):
        from domain.session import Session

        session_id = Session.generate_id()

        assert len(session_id) == 12
        assert session_id.isdigit()

    # === 항목 7: Session 도메인 확장 테스트 ===

    def test_session_with_token_usage(self):
        """token_usage 필드가 포함된 세션 생성 테스트"""
        from domain.session import Session

        token_usage = {"input": 1000, "output": 500, "total": 1500}
        session = Session(session_id="202601151430", token_usage=token_usage)

        assert session.token_usage["input"] == 1000
        assert session.token_usage["output"] == 500
        assert session.token_usage["total"] == 1500

    def test_session_default_token_usage(self):
        """token_usage 기본값 테스트"""
        from domain.session import Session

        session = Session(session_id="202601151430")

        assert session.token_usage == {"input": 0, "output": 0, "total": 0}

    def test_session_with_pdf_description(self):
        """pdf_description 필드 테스트"""
        from domain.session import Session

        session = Session(
            session_id="202601151430",
            pdf_description="AI 기술 동향에 대한 보고서",
        )

        assert session.pdf_description == "AI 기술 동향에 대한 보고서"

    def test_session_default_pdf_description(self):
        """pdf_description 기본값 테스트"""
        from domain.session import Session

        session = Session(session_id="202601151430")

        assert session.pdf_description == ""

    def test_session_from_dict_with_new_fields(self):
        """from_dict()에서 token_usage, pdf_description 파싱 테스트"""
        from domain.session import Session

        data = {
            "session_id": "202601151430",
            "created_at": "2026-01-15T14:30:00",
            "last_updated": "2026-01-15T14:30:00",
            "total_turns": 5,
            "current_summary": "Summary",
            "pdf_files": ["doc.pdf"],
            "settings": {"model": "gemini-2.5-flash"},
            "token_usage": {"input": 1000, "output": 500, "total": 1500},
            "pdf_description": "AI 기술 동향 보고서",
        }
        session = Session.from_dict(data)

        assert session.token_usage["input"] == 1000
        assert session.token_usage["total"] == 1500
        assert session.pdf_description == "AI 기술 동향 보고서"

    def test_session_from_dict_missing_new_fields(self):
        """from_dict()에서 새 필드 없을 때 기본값 테스트 (하위 호환성)"""
        from domain.session import Session

        data = {
            "session_id": "202601151430",
            "total_turns": 5,
        }
        session = Session.from_dict(data)

        assert session.token_usage == {"input": 0, "output": 0, "total": 0}
        assert session.pdf_description == ""

    # === Phase 02 항목 6: summary_history 필드 테스트 ===

    def test_session_default_summary_history(self):
        """summary_history 기본값 테스트 (빈 리스트)"""
        from domain.session import Session

        session = Session(session_id="202601151430")

        assert session.summary_history == []

    def test_session_with_summary_history(self):
        """summary_history가 포함된 세션 생성 테스트"""
        from domain.session import Session

        summary_history = [
            {"created_at_turn": 4, "covers_turns": "1-3", "summary": "사용자가 A를 물어봤고..."},
            {"created_at_turn": 7, "covers_turns": "4-6", "summary": "이후 B에 대해 논의..."},
        ]
        session = Session(session_id="202601151430", summary_history=summary_history)

        assert len(session.summary_history) == 2
        assert session.summary_history[0]["created_at_turn"] == 4
        assert session.summary_history[1]["covers_turns"] == "4-6"

    def test_session_from_dict_with_summary_history(self):
        """from_dict()에서 summary_history 파싱 테스트"""
        from domain.session import Session

        data = {
            "session_id": "202601151430",
            "created_at": "2026-01-15T14:30:00",
            "summary_history": [
                {"created_at_turn": 4, "covers_turns": "1-3", "summary": "요약 1"},
                {"created_at_turn": 7, "covers_turns": "4-6", "summary": "요약 2"},
            ],
        }
        session = Session.from_dict(data)

        assert len(session.summary_history) == 2
        assert session.summary_history[0]["summary"] == "요약 1"

    def test_session_from_dict_missing_summary_history(self):
        """from_dict()에서 summary_history 없을 때 기본값 테스트 (하위 호환성)"""
        from domain.session import Session

        data = {
            "session_id": "202601151430",
            "total_turns": 5,
        }
        session = Session.from_dict(data)

        assert session.summary_history == []
