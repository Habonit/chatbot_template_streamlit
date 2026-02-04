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

    def test_message_to_dict(self):
        from domain.message import Message

        msg = Message(turn_id=1, role="user", content="Test")
        data = msg.to_dict()

        assert "turn_id" in data
        assert "role" in data
        assert "content" in data
        assert "timestamp" in data

    def test_message_from_dict(self):
        from domain.message import Message

        data = {
            "turn_id": 1,
            "role": "user",
            "content": "Test",
            "timestamp": "2026-01-15T14:30:00",
            "input_tokens": 0,
            "output_tokens": 0,
            "model_used": None,
        }
        msg = Message.from_dict(data)

        assert msg.turn_id == 1
        assert msg.content == "Test"


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

    def test_session_add_turn(self):
        from domain.session import Session

        session = Session(session_id="202601151430")
        session.add_turn()
        session.add_turn()

        assert session.total_turns == 2

    def test_session_update_summary(self):
        from domain.session import Session

        session = Session(session_id="202601151430")
        session.update_summary("User asked about AI chatbots.")

        assert session.current_summary == "User asked about AI chatbots."

    def test_session_add_pdf(self):
        from domain.session import Session

        session = Session(session_id="202601151430")
        session.add_pdf("document.pdf")

        assert "document.pdf" in session.pdf_files

    def test_session_to_dict(self):
        from domain.session import Session

        session = Session(session_id="202601151430")
        data = session.to_dict()

        assert "session_id" in data
        assert "created_at" in data
        assert "settings" in data

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

    def test_session_update_token_usage(self):
        """update_token_usage 메서드 테스트"""
        from domain.session import Session

        session = Session(session_id="202601151430")
        session.update_token_usage(input_tokens=100, output_tokens=50)

        assert session.token_usage["input"] == 100
        assert session.token_usage["output"] == 50
        assert session.token_usage["total"] == 150

    def test_session_update_token_usage_accumulate(self):
        """update_token_usage 누적 테스트"""
        from domain.session import Session

        session = Session(session_id="202601151430")
        session.update_token_usage(input_tokens=100, output_tokens=50)
        session.update_token_usage(input_tokens=200, output_tokens=100)

        assert session.token_usage["input"] == 300
        assert session.token_usage["output"] == 150
        assert session.token_usage["total"] == 450

    def test_session_to_dict_with_new_fields(self):
        """to_dict()에 token_usage, pdf_description 포함 테스트"""
        from domain.session import Session

        session = Session(
            session_id="202601151430",
            token_usage={"input": 1000, "output": 500, "total": 1500},
            pdf_description="테스트 문서",
        )
        data = session.to_dict()

        assert "token_usage" in data
        assert data["token_usage"]["input"] == 1000
        assert "pdf_description" in data
        assert data["pdf_description"] == "테스트 문서"

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
