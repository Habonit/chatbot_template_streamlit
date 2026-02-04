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
