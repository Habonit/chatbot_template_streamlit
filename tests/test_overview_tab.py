import pytest


class TestOverviewTab:
    def test_render_overview_tab_exists(self):
        """Overview 탭 렌더링 함수 존재 확인"""
        from component.overview_tab import render_overview_tab
        assert callable(render_overview_tab)

    def test_get_overview_content_structure(self):
        """Overview 콘텐츠 구조 확인"""
        from component.overview_tab import get_overview_content

        content = get_overview_content()

        # 필수 섹션 확인
        assert "introduction" in content
        assert "quick_start" in content
        assert "features" in content
        assert "settings" in content
        assert "faq" in content

    def test_overview_content_not_empty(self):
        """각 섹션 내용이 비어있지 않음 확인"""
        from component.overview_tab import get_overview_content

        content = get_overview_content()

        for section, text in content.items():
            assert len(text) > 0, f"Section '{section}' should not be empty"
