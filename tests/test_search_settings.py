"""Step 2-1: search_depth, max_results 설정 전달 테스트"""
import inspect
from unittest.mock import MagicMock, patch


class TestToolsSearchParams:
    """create_tools_with_services가 search params를 올바르게 전달하는지 테스트"""

    def test_web_search_passes_search_depth(self):
        """web_search 도구가 search_depth를 SearchService에 전달"""
        from service.tools import create_tools_with_services

        mock_search = MagicMock()
        mock_search.search.return_value = []
        mock_search.format_for_llm.return_value = "결과"

        tools = create_tools_with_services(
            search_service=mock_search,
            search_depth="advanced",
            max_results=3,
        )

        web_search_tool = next(t for t in tools if t.name == "web_search")
        web_search_tool.invoke({"query": "test query"})

        mock_search.search.assert_called_once_with(
            "test query", search_depth="advanced", max_results=3
        )

    def test_web_search_uses_default_params(self):
        """기본값으로 search_depth=basic, max_results=5 사용"""
        from service.tools import create_tools_with_services

        mock_search = MagicMock()
        mock_search.search.return_value = []
        mock_search.format_for_llm.return_value = "결과"

        tools = create_tools_with_services(search_service=mock_search)

        web_search_tool = next(t for t in tools if t.name == "web_search")
        web_search_tool.invoke({"query": "test"})

        mock_search.search.assert_called_once_with(
            "test", search_depth="basic", max_results=5
        )

    def test_create_tools_accepts_search_params(self):
        """create_tools_with_services가 search_depth, max_results 파라미터를 받는지"""
        from service.tools import create_tools_with_services

        source = inspect.getsource(create_tools_with_services)
        assert "search_depth" in source
        assert "max_results" in source


class TestReactGraphBuilderSearchParams:
    """ReactGraphBuilder가 search params를 tools에 전달하는지 테스트"""

    def test_builder_stores_search_params(self):
        """ReactGraphBuilder가 search_depth, max_results를 저장"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            search_depth="advanced",
            max_results=3,
            db_path=":memory:",
        )

        assert builder.search_depth == "advanced"
        assert builder.max_results == 3

    def test_builder_default_search_params(self):
        """기본값 확인"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            db_path=":memory:",
        )

        assert builder.search_depth == "basic"
        assert builder.max_results == 5

    def test_build_passes_search_params_to_tools(self):
        """build()가 create_tools_with_services에 search params 전달"""
        from service.react_graph import ReactGraphBuilder
        from langchain_google_genai import ChatGoogleGenerativeAI

        builder = ReactGraphBuilder(
            api_key="test-key",
            search_depth="advanced",
            max_results=8,
            db_path=":memory:",
        )

        with patch("service.react_graph.create_tools_with_services") as mock_create:
            mock_create.return_value = []
            with patch.object(ChatGoogleGenerativeAI, "bind_tools", return_value=MagicMock()):
                builder.build()

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert "search_depth" in str(call_kwargs)
            assert "max_results" in str(call_kwargs)


class TestAppSearchSettingsPassing:
    """app.py가 settings에서 search params를 ReactGraphBuilder에 전달하는지 테스트"""

    def test_app_passes_search_depth(self):
        """app.py 소스코드에 search_depth 전달이 있어야 함"""
        import app

        source = inspect.getsource(app._create_graph_builder)
        assert "search_depth=" in source

    def test_app_passes_max_results(self):
        """app.py 소스코드에 max_results 전달이 있어야 함"""
        import app

        source = inspect.getsource(app._create_graph_builder)
        assert "max_results=" in source
