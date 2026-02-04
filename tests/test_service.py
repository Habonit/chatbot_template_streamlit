"""Phase 02-6: Service 테스트 (레거시 제거됨)"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestRAGService:
    def test_recursive_chunking(self):
        from service.rag_service import RAGService

        service = RAGService()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = service.chunk_text(text, chunk_size=1024, overlap=256)

        assert len(chunks) >= 1
        assert all(hasattr(c, "original_text") for c in chunks)

    def test_chunk_with_long_text(self):
        from service.rag_service import RAGService

        service = RAGService()
        text = "A" * 3000

        chunks = service.chunk_text(text, chunk_size=1024, overlap=256)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.original_text) <= 1024

    def test_chunk_metadata(self):
        from service.rag_service import RAGService

        service = RAGService()
        text = "Hello world. This is a test."

        chunks = service.chunk_text(text, source_file="test.pdf", source_page=1)

        assert chunks[0].source_file == "test.pdf"
        assert chunks[0].source_page == 1
        assert chunks[0].chunk_index == 0


class TestSearchService:
    @patch("service.search_service.TavilyClient")
    def test_search_basic(self, mock_client_class):
        from service.search_service import SearchService

        mock_client = Mock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Test", "url": "http://test.com", "content": "Test content"}
            ]
        }
        mock_client_class.return_value = mock_client

        service = SearchService(api_key="test_key")
        results = service.search("test query", search_depth="basic", max_results=5)

        assert len(results) == 1
        assert results[0]["title"] == "Test"

    @patch("service.search_service.TavilyClient")
    def test_format_results_for_llm(self, mock_client_class):
        from service.search_service import SearchService

        mock_client_class.return_value = Mock()

        service = SearchService(api_key="test_key")
        results = [
            {"title": "Title 1", "url": "http://url1.com", "content": "Content 1"},
            {"title": "Title 2", "url": "http://url2.com", "content": "Content 2"},
        ]

        formatted = service.format_for_llm(results)

        assert "[웹 검색 결과]" in formatted
        assert "Title 1" in formatted
        assert "http://url1.com" in formatted


class TestEmbeddingService:
    @patch("service.embedding_service.genai")
    def test_create_embedding(self, mock_genai):
        from service.embedding_service import EmbeddingService

        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(values=[0.1] * 768)]
        mock_client.models.embed_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        service = EmbeddingService(api_key="test_key")
        embedding = service.create_embedding("test text")

        assert len(embedding) == 768

    @patch("service.embedding_service.genai")
    def test_create_embeddings_batch(self, mock_genai):
        from service.embedding_service import EmbeddingService

        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(values=[0.1] * 768), Mock(values=[0.2] * 768)]
        mock_client.models.embed_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        service = EmbeddingService(api_key="test_key")
        embeddings = service.create_embeddings(["text1", "text2"])

        assert len(embeddings) == 2


class TestLLMService:
    @patch("service.llm_service.genai")
    def test_generate_content(self, mock_genai):
        from service.llm_service import LLMService

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Hello!"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        mock_response.candidates = [Mock(finish_reason="STOP")]
        mock_response.function_calls = None
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        service = LLMService(api_key="test_key")
        result = service.generate("Hello", model="gemini-2.5-flash")

        assert result["text"] == "Hello!"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5

    @patch("service.llm_service.genai")
    def test_generate_with_tools(self, mock_genai):
        from service.llm_service import LLMService

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Using tool"
        mock_response.function_calls = []
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        mock_response.candidates = [Mock(finish_reason="STOP")]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        service = LLMService(api_key="test_key")

        def sample_tool(query: str) -> str:
            return "result"

        result = service.generate("Test", tools=[sample_tool])

        assert "text" in result

    @patch("service.llm_service.genai")
    def test_generate_with_max_output_tokens(self, mock_genai):
        """max_output_tokens 파라미터가 config에 전달되는지 테스트"""
        from service.llm_service import LLMService

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Hello!"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        )
        mock_response.function_calls = None
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        service = LLMService(api_key="test_key")
        result = service.generate(
            "Hello",
            model="gemini-2.5-flash",
            max_output_tokens=4096,
        )

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")

        assert config.max_output_tokens == 4096
        assert result["text"] == "Hello!"
