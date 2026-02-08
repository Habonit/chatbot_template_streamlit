"""Phase 04 Step 5: Overview Tab redesign tests"""
from unittest.mock import MagicMock, patch

import pytest


class TestGetOverviewContent:
    """get_overview_content() 테스트"""

    def test_get_overview_content_has_sections(self):
        """모든 필수 섹션 키가 존재하는지 확인"""
        from component.overview_tab import get_overview_content

        content = get_overview_content()

        expected_keys = ["introduction", "quick_start", "features", "settings", "faq"]
        for key in expected_keys:
            assert key in content, f"Missing key: {key}"

    def test_introduction_contains_key_concepts(self):
        """introduction에 핵심 개념이 언급되는지 확인"""
        from component.overview_tab import get_overview_content

        content = get_overview_content()
        intro = content["introduction"]

        expected_concepts = [
            "ReAct",
            "Tool Calling",
            "Streaming",
            "Thinking Mode",
            "Context Compression",
            "Casual Detection",
            "Session Checkpointing",
        ]
        for concept in expected_concepts:
            assert concept in intro, f"Introduction should mention '{concept}'"


class TestGetLanggraphDiagram:
    """get_langgraph_diagram() 테스트"""

    def test_get_langgraph_diagram_has_react_nodes(self):
        """마크다운에 ReAct 노드들이 포함되는지 확인"""
        from component.overview_tab import get_langgraph_diagram

        markdown = get_langgraph_diagram()

        assert "summary_node" in markdown
        assert "llm_node" in markdown
        assert "tool_node" in markdown
        assert "tools_condition" in markdown

    def test_get_langgraph_diagram_is_markdown_table(self):
        """마크다운 테이블 형식인지 확인"""
        from component.overview_tab import get_langgraph_diagram

        markdown = get_langgraph_diagram()

        assert "|" in markdown
        assert "순서" in markdown
        assert "노드" in markdown


class TestGetArchitectureDiagram:
    """get_architecture_diagram() 테스트"""

    def test_get_architecture_diagram(self):
        """아키텍처 다이어그램에 UI, Service, LangGraph 섹션이 포함되는지 확인"""
        from component.overview_tab import get_architecture_diagram

        diagram = get_architecture_diagram()

        # UI layer
        assert "Streamlit UI" in diagram
        assert "Overview Tab" in diagram
        assert "Chat Tab" in diagram
        assert "Sidebar Settings" in diagram

        # Service layer
        assert "Service Layer" in diagram
        assert "ReactGraphBuilder" in diagram
        assert "ReasoningDetector" in diagram
        assert "SessionManager" in diagram
        assert "RAGService" in diagram

        # LangGraph layer
        assert "LangGraph ReAct Graph" in diagram
        assert "summary_node" in diagram
        assert "llm_node" in diagram
        assert "tool_node" in diagram


class TestGetConceptCards:
    """get_concept_cards() 테스트"""

    def test_get_concept_cards_count(self):
        """7개의 카드가 반환되는지 확인"""
        from component.overview_tab import get_concept_cards

        cards = get_concept_cards()
        assert len(cards) == 7

    def test_get_concept_cards_titles(self):
        """모든 카드 제목이 올바른지 확인"""
        from component.overview_tab import get_concept_cards

        cards = get_concept_cards()
        titles = [card["title"] for card in cards]

        expected_titles = [
            "ReAct 패턴",
            "Tool Calling",
            "Context Compression",
            "Streaming",
            "Thinking Mode",
            "Casual Detection",
            "Session & Checkpointing",
        ]
        for title in expected_titles:
            assert title in titles, f"Missing card title: {title}"

    def test_get_concept_cards_fields(self):
        """각 카드에 필수 필드(title, emoji, description, detail)가 있는지 확인"""
        from component.overview_tab import get_concept_cards

        cards = get_concept_cards()
        required_fields = ["title", "emoji", "description", "detail"]

        for card in cards:
            for field in required_fields:
                assert field in card, f"Card '{card.get('title', '?')}' missing field: {field}"
                assert len(card[field]) > 0, f"Card '{card.get('title', '?')}' has empty field: {field}"


class TestGetToolInfo:
    """get_tool_info() 테스트"""

    def test_get_tool_info(self):
        """4개 도구가 반환되고 bind_method 필드가 포함되는지 확인"""
        from component.overview_tab import get_tool_info

        tools = get_tool_info()

        assert len(tools) == 4

        expected_names = [
            "get_current_time",
            "switch_to_reasoning",
            "web_search",
            "search_pdf_knowledge",
        ]
        for name in expected_names:
            found = any(t["name"] == name for t in tools)
            assert found, f"Missing tool: {name}"

        for tool in tools:
            assert "bind_method" in tool, f"Tool '{tool['name']}' missing 'bind_method' field"
            assert tool["bind_method"] == "bind_tools()"


class TestRenderOverviewTab:
    """render_overview_tab() 테스트"""

    @patch("component.overview_tab.st")
    @patch("component.overview_tab.st_mermaid", create=True)
    def test_render_overview_tab(self, mock_st_mermaid, mock_st):
        """render_overview_tab이 에러 없이 실행되는지 확인"""
        # streamlit_mermaid 모듈을 모킹
        mock_mermaid_module = MagicMock()
        mock_mermaid_module.st_mermaid = MagicMock()

        with patch.dict("sys.modules", {"streamlit_mermaid": mock_mermaid_module}):
            # st의 context manager 메서드 설정
            mock_expander = MagicMock()
            mock_expander.__enter__ = MagicMock(return_value=mock_expander)
            mock_expander.__exit__ = MagicMock(return_value=False)
            mock_st.expander.return_value = mock_expander

            mock_container = MagicMock()
            mock_container.__enter__ = MagicMock(return_value=mock_container)
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_st.container.return_value = mock_container

            from component.overview_tab import render_overview_tab

            render_overview_tab()

            # 기본 호출 확인
            mock_st.title.assert_called_once_with("Gemini Hybrid Chatbot")
            mock_st.caption.assert_any_call("AI 챗봇 핵심 개념 교육 데모")
            mock_st.divider.assert_called()


class TestGetResponseLengthDiagram:
    """get_response_length_diagram() 테스트 (하위 호환성)"""

    def test_get_response_length_diagram_still_exists(self):
        """get_response_length_diagram 함수가 여전히 존재하고 유효한 Mermaid를 반환하는지 확인"""
        from component.overview_tab import get_response_length_diagram

        diagram = get_response_length_diagram()

        assert "graph" in diagram.lower() or "flowchart" in diagram.lower()
        assert "상세" in diagram or "detailed" in diagram.lower()
        assert "간결" in diagram or "brief" in diagram.lower()
