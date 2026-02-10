"""Phase 05 Step 6: Sidebar 파라미터 help 텍스트 테스트

sidebar.py에서 get_parameter_help 사용 확인 및
각 파라미터 help 텍스트가 비어있지 않은지 검증.
"""
import pytest
import inspect


class TestSidebarImport:
    """sidebar.py에 get_parameter_help 임포트 확인"""

    def test_sidebar_imports_get_parameter_help(self):
        """sidebar.py 소스에 get_parameter_help 임포트 존재"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert "get_parameter_help" in source


class TestSidebarParameterHelps:
    """각 파라미터의 help 텍스트 적용 확인"""

    def test_temperature_help_in_source(self):
        """sidebar.py에 temperature help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("temperature")' in source

    def test_top_p_help_in_source(self):
        """sidebar.py에 top_p help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("top_p")' in source

    def test_max_output_tokens_help_in_source(self):
        """sidebar.py에 max_output_tokens help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("max_output_tokens")' in source

    def test_seed_help_in_source(self):
        """sidebar.py에 seed help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("seed")' in source

    def test_thinking_budget_help_in_source(self):
        """sidebar.py에 thinking_budget help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("thinking_budget")' in source

    def test_compression_rate_help_in_source(self):
        """sidebar.py에 compression_rate help 적용"""
        import component.sidebar as sidebar_module
        source = inspect.getsource(sidebar_module)
        assert 'get_parameter_help("compression_rate")' in source
