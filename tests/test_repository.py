import pytest
import tempfile
import os
from pathlib import Path
import numpy as np


class TestConversationRepository:
    def test_save_and_load_messages(self, tmp_path):
        from repository.conversation_repo import ConversationRepository
        from domain.message import Message

        repo = ConversationRepository(base_path=tmp_path)
        session_id = "202601151430"

        messages = [
            Message(turn_id=1, role="user", content="Hello"),
            Message(turn_id=1, role="assistant", content="Hi there!", input_tokens=5, output_tokens=3),
        ]

        repo.save_messages(session_id, messages)
        loaded = repo.load_messages(session_id)

        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert loaded[1].role == "assistant"

    def test_append_message(self, tmp_path):
        from repository.conversation_repo import ConversationRepository
        from domain.message import Message

        repo = ConversationRepository(base_path=tmp_path)
        session_id = "202601151430"

        msg1 = Message(turn_id=1, role="user", content="First")
        msg2 = Message(turn_id=1, role="assistant", content="Response")

        repo.append_message(session_id, msg1)
        repo.append_message(session_id, msg2)

        loaded = repo.load_messages(session_id)
        assert len(loaded) == 2

    def test_load_empty_returns_empty_list(self, tmp_path):
        from repository.conversation_repo import ConversationRepository

        repo = ConversationRepository(base_path=tmp_path)
        loaded = repo.load_messages("nonexistent")

        assert loaded == []

    def test_clear_messages(self, tmp_path):
        from repository.conversation_repo import ConversationRepository
        from domain.message import Message

        repo = ConversationRepository(base_path=tmp_path)
        session_id = "202601151430"

        repo.append_message(session_id, Message(turn_id=1, role="user", content="Test"))
        repo.clear_messages(session_id)

        loaded = repo.load_messages(session_id)
        assert loaded == []


class TestEmbeddingRepository:
    def test_save_and_load_chunks(self, tmp_path):
        from repository.embedding_repo import EmbeddingRepository
        from domain.chunk import Chunk

        repo = EmbeddingRepository(base_path=tmp_path)
        session_id = "202601151430"

        chunks = [
            Chunk(
                chunk_index=0,
                original_text="Original",
                normalized_text="Normalized",
                source_file="doc.pdf",
                source_page=1,
                start_char=0,
                end_char=100,
                embedding=np.random.rand(768).astype(np.float32),
            ),
        ]

        repo.save_chunks(session_id, chunks, embedding_model="gemini-embedding-001", embedding_dim=768)
        loaded_chunks, config = repo.load_chunks(session_id)

        assert len(loaded_chunks) == 1
        assert loaded_chunks[0].original_text == "Original"
        assert config["embedding_model"] == "gemini-embedding-001"

    def test_search_similar(self, tmp_path):
        from repository.embedding_repo import EmbeddingRepository
        from domain.chunk import Chunk

        repo = EmbeddingRepository(base_path=tmp_path)
        session_id = "202601151430"

        embeddings = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]

        chunks = [
            Chunk(
                chunk_index=i,
                original_text=f"Text {i}",
                normalized_text=f"Text {i}",
                source_file="doc.pdf",
                source_page=1,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                embedding=embeddings[i],
            )
            for i in range(3)
        ]

        repo.save_chunks(session_id, chunks, embedding_model="test", embedding_dim=3)

        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = repo.search_similar(session_id, query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0]["chunk"].chunk_index == 0

    def test_delete_chunks(self, tmp_path):
        from repository.embedding_repo import EmbeddingRepository
        from domain.chunk import Chunk

        repo = EmbeddingRepository(base_path=tmp_path)
        session_id = "202601151430"

        chunks = [
            Chunk(
                chunk_index=0,
                original_text="Text",
                normalized_text="Text",
                source_file="doc.pdf",
                source_page=1,
                start_char=0,
                end_char=10,
                embedding=np.random.rand(768).astype(np.float32),
            ),
        ]

        repo.save_chunks(session_id, chunks, embedding_model="test", embedding_dim=768)
        repo.delete_chunks(session_id)

        loaded, _ = repo.load_chunks(session_id)
        assert loaded == []

    def test_load_nonexistent_returns_empty(self, tmp_path):
        from repository.embedding_repo import EmbeddingRepository

        repo = EmbeddingRepository(base_path=tmp_path)
        chunks, config = repo.load_chunks("nonexistent")

        assert chunks == []
        assert config == {}


class TestSessionRepository:
    """항목 7: SessionRepository 테스트 - 세션 메타데이터 저장/로드"""

    def test_save_and_load_session(self, tmp_path):
        """세션 저장 및 로드 테스트"""
        from repository.session_repo import SessionRepository
        from domain.session import Session

        repo = SessionRepository(base_path=tmp_path)
        session = Session(
            session_id="202601151430",
            total_turns=5,
            current_summary="테스트 요약",
            token_usage={"input": 1000, "output": 500, "total": 1500},
            pdf_description="테스트 PDF 설명",
        )

        repo.save_session(session)
        loaded = repo.load_session("202601151430")

        assert loaded is not None
        assert loaded.session_id == "202601151430"
        assert loaded.total_turns == 5
        assert loaded.current_summary == "테스트 요약"
        assert loaded.token_usage["input"] == 1000
        assert loaded.pdf_description == "테스트 PDF 설명"

    def test_load_nonexistent_session(self, tmp_path):
        """존재하지 않는 세션 로드 시 None 반환"""
        from repository.session_repo import SessionRepository

        repo = SessionRepository(base_path=tmp_path)
        loaded = repo.load_session("nonexistent")

        assert loaded is None

    def test_list_sessions(self, tmp_path):
        """세션 목록 조회 테스트"""
        from repository.session_repo import SessionRepository
        from domain.session import Session

        repo = SessionRepository(base_path=tmp_path)

        repo.save_session(Session(session_id="session1"))
        repo.save_session(Session(session_id="session2"))
        repo.save_session(Session(session_id="session3"))

        sessions = repo.list_sessions()

        assert len(sessions) == 3
        assert "session1" in sessions
        assert "session2" in sessions
        assert "session3" in sessions

    def test_list_sessions_empty(self, tmp_path):
        """세션이 없을 때 빈 리스트 반환"""
        from repository.session_repo import SessionRepository

        repo = SessionRepository(base_path=tmp_path)
        sessions = repo.list_sessions()

        assert sessions == []

    def test_delete_session(self, tmp_path):
        """세션 삭제 테스트"""
        from repository.session_repo import SessionRepository
        from domain.session import Session

        repo = SessionRepository(base_path=tmp_path)
        repo.save_session(Session(session_id="202601151430"))

        repo.delete_session("202601151430")
        loaded = repo.load_session("202601151430")

        assert loaded is None

    def test_session_exists(self, tmp_path):
        """세션 존재 여부 확인 테스트"""
        from repository.session_repo import SessionRepository
        from domain.session import Session

        repo = SessionRepository(base_path=tmp_path)
        repo.save_session(Session(session_id="202601151430"))

        assert repo.exists("202601151430") is True
        assert repo.exists("nonexistent") is False

    def test_update_session(self, tmp_path):
        """세션 업데이트 테스트"""
        from repository.session_repo import SessionRepository
        from domain.session import Session

        repo = SessionRepository(base_path=tmp_path)

        session = Session(session_id="202601151430", total_turns=0)
        repo.save_session(session)

        session.total_turns = 10
        session.current_summary = "업데이트된 요약"
        repo.save_session(session)

        loaded = repo.load_session("202601151430")

        assert loaded.total_turns == 10
        assert loaded.current_summary == "업데이트된 요약"


class TestPDFExtractor:
    def test_extract_text_from_pdf(self, tmp_path):
        from repository.pdf_extractor import PDFExtractor

        extractor = PDFExtractor()

        test_pdf = tmp_path / "test.pdf"
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(test_pdf), pagesize=letter)
        c.drawString(100, 750, "Hello, this is a test PDF document.")
        c.save()

        text, pages = extractor.extract_text(test_pdf)

        assert "Hello" in text
        assert pages == 1

    def test_validate_pdf_size(self):
        from repository.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(max_size_mb=20)

        assert extractor.validate_size(1024 * 1024 * 10) is True
        assert extractor.validate_size(1024 * 1024 * 25) is False

    def test_validate_pdf_extension(self):
        from repository.pdf_extractor import PDFExtractor

        extractor = PDFExtractor()

        assert extractor.validate_extension("document.pdf") is True
        assert extractor.validate_extension("document.PDF") is True
        assert extractor.validate_extension("document.txt") is False
