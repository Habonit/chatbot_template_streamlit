import pytest


class TestPromptsTab:
    def test_render_prompts_tab_exists(self):
        """프롬프트 탭 렌더링 함수 존재 확인"""
        from component.prompts_tab import render_prompts_tab
        assert callable(render_prompts_tab)

    def test_get_prompt_info_structure(self):
        """프롬프트 정보 구조 확인"""
        from component.prompts_tab import get_prompt_info

        info = get_prompt_info()

        # 필수 프롬프트 섹션 확인
        assert "system_prompt" in info
        assert "summary_prompt" in info
        assert "normalization_prompt" in info
        assert "description_prompt" in info

    def test_prompt_info_has_required_fields(self):
        """각 프롬프트 정보에 필수 필드 확인"""
        from component.prompts_tab import get_prompt_info

        info = get_prompt_info()

        for key, prompt_data in info.items():
            assert "title" in prompt_data, f"'{key}' should have 'title'"
            assert "description" in prompt_data, f"'{key}' should have 'description'"
            assert "content" in prompt_data, f"'{key}' should have 'content'"
            assert "usage" in prompt_data, f"'{key}' should have 'usage'"

    def test_prompt_content_not_empty(self):
        """프롬프트 내용이 비어있지 않음 확인"""
        from component.prompts_tab import get_prompt_info

        info = get_prompt_info()

        for key, prompt_data in info.items():
            assert len(prompt_data["content"]) > 0, f"'{key}' content should not be empty"
