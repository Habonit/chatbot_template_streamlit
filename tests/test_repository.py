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
