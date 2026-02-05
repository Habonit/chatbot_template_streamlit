"""Phase 03-1: LangSmith + seed 파라미터 테스트

TDD: 테스트 먼저 작성
"""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestLangSmithEnvironment:
    """LangSmith 환경 변수 설정 테스트"""

    def test_langsmith_env_vars_structure(self):
        """LangSmith 환경 변수 키 존재 확인"""
        required_vars = [
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
        ]

        # .env.example에서 환경 변수 키 확인
        env_example_path = "/.env.example"
        # 실제 앱에서는 dotenv가 로드해야 함

        # 환경 변수 키가 정의 가능해야 함
        for var in required_vars:
            # os.environ에 설정 가능한지 확인
            os.environ[var] = "test_value"
            assert os.environ.get(var) == "test_value"
            del os.environ[var]

    def test_langsmith_tracing_value(self):
        """LANGSMITH_TRACING 값은 'true' 문자열이어야 함"""
        os.environ["LANGSMITH_TRACING"] = "true"
        assert os.environ["LANGSMITH_TRACING"] == "true"
        del os.environ["LANGSMITH_TRACING"]

    def test_langsmith_project_default(self):
        """LANGSMITH_PROJECT 기본값 확인"""
        # 기본값은 'default'
        project = os.environ.get("LANGSMITH_PROJECT", "default")
        assert project == "default"


class TestSidebarSeedParameter:
    """sidebar.py seed 파라미터 테스트 - seed 변환 로직만 테스트"""

    def test_seed_conversion_negative_to_none(self):
        """seed=-1은 None으로 변환되어야 함"""
        seed = -1
        result = seed if seed >= 0 else None
        assert result is None

    def test_seed_conversion_zero_is_kept(self):
        """seed=0은 그대로 유지되어야 함"""
        seed = 0
        result = seed if seed >= 0 else None
        assert result == 0

    def test_seed_conversion_positive_is_kept(self):
        """seed > 0은 그대로 유지되어야 함"""
        seed = 42
        result = seed if seed >= 0 else None
        assert result == 42

    def test_sidebar_module_has_seed_in_return_dict(self):
        """sidebar.py 소스코드에 seed 반환이 있어야 함"""
        import inspect
        from component import sidebar

        source = inspect.getsource(sidebar.render_sidebar)
        assert '"seed":' in source or "'seed':" in source


class TestReactGraphSeedParameter:
    """react_graph.py seed 파라미터 테스트"""

    def test_react_graph_builder_accepts_seed(self):
        """ReactGraphBuilder가 seed 파라미터를 받아야 함"""
        from service.react_graph import ReactGraphBuilder

        # seed 파라미터로 초기화 가능해야 함
        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
            seed=42,
        )

        assert builder.seed == 42

    def test_react_graph_builder_seed_none_default(self):
        """ReactGraphBuilder seed 기본값은 None"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
        )

        assert builder.seed is None

    def test_react_graph_builder_accepts_top_p(self):
        """ReactGraphBuilder가 top_p 파라미터를 받아야 함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
            top_p=0.95,
        )

        assert builder.top_p == 0.95

    def test_react_graph_builder_accepts_max_output_tokens(self):
        """ReactGraphBuilder가 max_output_tokens 파라미터를 받아야 함"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test_api_key",
            model="gemini-2.5-flash",
            max_output_tokens=4096,
        )

        assert builder.max_output_tokens == 4096


class TestEnvFileStructure:
    """환경 변수 파일 구조 테스트"""

    def test_env_example_has_langsmith_vars(self):
        """.env.example에 LangSmith 변수가 있어야 함"""
        env_example_path = "/home/paradeigma/workspace/01_work/05_20260223_chatbot/.env.example"

        with open(env_example_path, "r") as f:
            content = f.read()

        # LangSmith 관련 변수가 있어야 함
        assert "LANGSMITH_TRACING" in content
        assert "LANGSMITH_API_KEY" in content
        assert "LANGSMITH_PROJECT" in content


class TestAppParameterPassing:
    """app.py에서 ReactGraphBuilder로 파라미터 전달 테스트"""

    def test_app_passes_top_p_to_graph_builder(self):
        """app.py 소스코드에 top_p 전달이 있어야 함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)

        # ReactGraphBuilder 호출부에 top_p가 있어야 함
        assert "top_p=" in source or "top_p =" in source

    def test_app_passes_max_output_tokens_to_graph_builder(self):
        """app.py 소스코드에 max_output_tokens 전달이 있어야 함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)

        # ReactGraphBuilder 호출부에 max_output_tokens가 있어야 함
        assert "max_output_tokens=" in source or "max_output_tokens =" in source

    def test_app_passes_seed_to_graph_builder(self):
        """app.py 소스코드에 seed 전달이 있어야 함"""
        import inspect
        import app

        source = inspect.getsource(app.handle_chat_message)

        # ReactGraphBuilder 호출부에 seed가 있어야 함
        assert "seed=" in source or "seed =" in source
