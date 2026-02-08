"""Phase 04 Step 6: Prompts Tab 테스트

get_prompt_info(), get_system_prompt_builder_info(), get_prompt_flow_diagram(),
render_prompts_tab() 테스트
"""
import pytest
from unittest.mock import patch, MagicMock

from service.prompt_loader import PromptLoader


class TestGetPromptInfoGraphNode:
    """get_prompt_info()의 graph_node 필드 테스트"""

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_get_prompt_info_has_graph_node(self, mock_load):
        """모든 프롬프트에 graph_node 필드가 있어야 한다"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        for key, info in prompt_info.items():
            assert "graph_node" in info, f"{key}에 graph_node 필드가 없습니다"

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_get_prompt_info_has_status(self, mock_load):
        """모든 프롬프트에 status 필드가 있어야 한다"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        for key, info in prompt_info.items():
            assert "status" in info, f"{key}에 status 필드가 없습니다"

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_active_prompts_count(self, mock_load):
        """활성 프롬프트는 5개여야 한다"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        active = {k: v for k, v in prompt_info.items() if v["status"] == "active"}

        assert len(active) == 5

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_legacy_prompts_count(self, mock_load):
        """레거시 프롬프트는 2개여야 한다"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        legacy = {k: v for k, v in prompt_info.items() if v["status"] == "legacy"}

        assert len(legacy) == 2

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_legacy_prompts_identified(self, mock_load):
        """result_processor와 response_generator가 레거시로 식별되어야 한다"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert prompt_info["result_processor"]["status"] == "legacy"
        assert prompt_info["response_generator"]["status"] == "legacy"

    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    def test_prompt_info_graph_node_values(self, mock_load):
        """주요 프롬프트의 graph_node 값이 올바른지 확인"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert prompt_info["tool_selector"]["graph_node"] == "llm_node (시스템 프롬프트)"
        assert prompt_info["reasoning_prompt"]["graph_node"] == "tool_node (reasoning tool)"
        assert prompt_info["summary_prompt"]["graph_node"] == "summary_node"
        assert prompt_info["result_processor"]["graph_node"] == "N/A (레거시)"
        assert prompt_info["response_generator"]["graph_node"] == "N/A (레거시)"
        assert prompt_info["normalization_prompt"]["graph_node"] == "N/A (PDF 전처리)"
        assert prompt_info["description_prompt"]["graph_node"] == "N/A (PDF 전처리)"


class TestGetSystemPromptBuilderInfo:
    """get_system_prompt_builder_info() 테스트"""

    def test_get_system_prompt_builder_info(self):
        """반환값에 필요한 키가 모두 있어야 한다"""
        from component.prompts_tab import get_system_prompt_builder_info

        info = get_system_prompt_builder_info()

        assert "title" in info
        assert "description" in info
        assert "components" in info
        assert "flow" in info

    def test_system_prompt_builder_components_count(self):
        """컴포넌트가 3개여야 한다"""
        from component.prompts_tab import get_system_prompt_builder_info

        info = get_system_prompt_builder_info()

        assert len(info["components"]) == 3

    def test_system_prompt_builder_flow(self):
        """flow 문자열에 핵심 요소가 포함되어야 한다"""
        from component.prompts_tab import get_system_prompt_builder_info

        info = get_system_prompt_builder_info()

        assert "기본 지침" in info["flow"]
        assert "이전 대화 요약" in info["flow"]
        assert "PDF 설명" in info["flow"]


class TestGetPromptFlowDiagram:
    """get_prompt_flow_diagram() 테스트"""

    def test_get_prompt_flow_diagram(self):
        """다이어그램 문자열에 핵심 요소가 포함되어야 한다"""
        from component.prompts_tab import get_prompt_flow_diagram

        diagram = get_prompt_flow_diagram()

        assert "ReasoningDetector" in diagram
        assert "casual" in diagram
        assert "System Prompt Builder" in diagram
        assert "summary" in diagram.lower()


class TestRenderPromptsTab:
    """render_prompts_tab() 렌더링 테스트"""

    @patch("component.prompts_tab.st")
    @patch.object(PromptLoader, "load", return_value="mock prompt content")
    @patch("streamlit_mermaid.st_mermaid")
    def test_render_prompts_tab(self, mock_mermaid, mock_load, mock_st):
        """render_prompts_tab()이 오류 없이 실행되어야 한다"""
        # st.expander를 context manager로 동작하도록 설정
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=False)
        mock_st.expander.return_value = mock_expander

        # st.container를 context manager로 동작하도록 설정
        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_st.container.return_value = mock_container

        from component.prompts_tab import render_prompts_tab

        render_prompts_tab()

        mock_st.title.assert_called_once_with("Prompts")
