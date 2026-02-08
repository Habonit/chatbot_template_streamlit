"""Step 3: 커스텀 CSS 스타일 테스트"""
from unittest.mock import patch, MagicMock


class TestGetCustomCss:
    """get_custom_css() 반환값 검증"""

    def test_contains_mode_badge_styles(self):
        """모드 뱃지 CSS 클래스 포함"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert ".mode-badge-casual" in css
        assert ".mode-badge-normal" in css
        assert ".mode-badge-reasoning" in css

    def test_contains_metadata_panel_style(self):
        """메타데이터 패널 CSS 포함"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert ".metadata-panel" in css

    def test_contains_concept_card_style(self):
        """컨셉 카드 CSS 포함"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert ".concept-card" in css

    def test_contains_welcome_card_style(self):
        """Welcome 카드 CSS 포함"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert ".welcome-card" in css

    def test_contains_graph_path_style(self):
        """그래프 경로 CSS 포함"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert ".graph-path" in css

    def test_css_wrapped_in_style_tag(self):
        """<style> 태그로 감싸져 있는지"""
        from component.styles import get_custom_css

        css = get_custom_css()
        assert "<style>" in css
        assert "</style>" in css


class TestInjectCustomCss:
    """inject_custom_css() 호출 테스트"""

    @patch("component.styles.st")
    def test_inject_calls_st_markdown(self, mock_st):
        """st.markdown이 unsafe_allow_html=True로 호출"""
        from component.styles import inject_custom_css

        result = inject_custom_css()
        mock_st.markdown.assert_called_once()
        call_args = mock_st.markdown.call_args
        assert call_args.kwargs.get("unsafe_allow_html") is True
        assert "<style>" in call_args.args[0]

    @patch("component.styles.st")
    def test_inject_returns_css_string(self, mock_st):
        """CSS 문자열을 반환"""
        from component.styles import inject_custom_css

        result = inject_custom_css()
        assert isinstance(result, str)
        assert "<style>" in result


class TestAppCssIntegration:
    """app.py에서 CSS 주입 확인"""

    def test_app_imports_inject_custom_css(self):
        """app.py가 inject_custom_css를 import"""
        import inspect
        import app

        source = inspect.getsource(app.main)
        assert "inject_custom_css" in source
