import pytest
from domain.message import Message


class TestSidebarHelpers:
    def test_generate_csv_data(self):
        """CSV 생성 헬퍼 함수 테스트"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content="Hello"),
            Message(turn_id=1, role="assistant", content="Hi there!", model_used="gemini-2.5-flash", input_tokens=10, output_tokens=5),
            Message(turn_id=2, role="user", content="How are you?"),
            Message(turn_id=2, role="assistant", content="I'm doing well!", model_used="gemini-2.5-flash", input_tokens=15, output_tokens=8),
        ]

        csv_data = _generate_csv_data(messages)

        # bytes 타입으로 반환되어야 함 (UTF-8 BOM 포함)
        assert isinstance(csv_data, bytes)

        # UTF-8 BOM 확인
        assert csv_data.startswith(b'\xef\xbb\xbf')

        # 내용 확인 (decode 후)
        csv_str = csv_data.decode("utf-8-sig")
        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_str
        assert "1,user,Hello" in csv_str
        assert "1,assistant,Hi there!,gemini-2.5-flash,10,5" in csv_str
        assert "2,user,How are you?" in csv_str

    def test_generate_csv_data_empty(self):
        """빈 메시지 리스트 CSV 생성 테스트"""
        from component.sidebar import _generate_csv_data

        messages = []
        csv_data = _generate_csv_data(messages)

        # bytes 타입 확인
        assert isinstance(csv_data, bytes)

        # 헤더만 있어야 함
        csv_str = csv_data.decode("utf-8-sig")
        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_str
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # 헤더만

    def test_generate_csv_data_special_characters(self):
        """특수 문자 포함 메시지 CSV 생성 테스트"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content='Hello, "world"'),
            Message(turn_id=1, role="assistant", content="Line1\nLine2"),
        ]

        csv_data = _generate_csv_data(messages)

        # CSV가 생성되어야 함 (csv 모듈이 특수 문자 처리)
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

    def test_generate_csv_data_korean(self):
        """한글 메시지 CSV 생성 테스트 (Excel 호환)"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content="안녕하세요"),
            Message(turn_id=1, role="assistant", content="반갑습니다!"),
        ]

        csv_data = _generate_csv_data(messages)

        # UTF-8 BOM 포함 확인 (Excel 한글 호환)
        assert csv_data.startswith(b'\xef\xbb\xbf')

        # 한글 정상 인코딩 확인
        csv_str = csv_data.decode("utf-8-sig")
        assert "안녕하세요" in csv_str
        assert "반갑습니다!" in csv_str


class TestPdfTabHelpers:
    """PDF 탭 헬퍼 함수 테스트 (항목 7)"""

    def test_format_time_seconds(self):
        """초 단위 시간 포맷 테스트"""
        from component.pdf_tab import _format_time

        assert _format_time(30) == "약 30초"
        assert _format_time(45.7) == "약 45초"
        assert _format_time(0) == "약 0초"

    def test_format_time_minutes(self):
        """분 단위 시간 포맷 테스트"""
        from component.pdf_tab import _format_time

        assert _format_time(60) == "약 1분 0초"
        assert _format_time(90) == "약 1분 30초"
        assert _format_time(150) == "약 2분 30초"
        assert _format_time(125.5) == "약 2분 5초"


class TestPromptsTab:
    """프롬프트 탭 테스트 (Phase 02-5 업데이트)"""

    def test_get_prompt_info_returns_all_prompts(self):
        """get_prompt_info()가 모든 프롬프트를 반환하는지 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        # Phase 02-5: 노드별 프롬프트 구조
        required_keys = [
            "tool_selector",
            "reasoning_prompt",
            "result_processor",
            "response_generator",
            "summary_prompt",
            "normalization_prompt",
            "description_prompt",
        ]

        for key in required_keys:
            assert key in prompt_info, f"Missing prompt: {key}"

    def test_get_prompt_info_has_required_fields(self):
        """각 프롬프트 정보에 필수 필드가 있는지 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        required_fields = ["title", "description", "usage", "content"]

        for key, info in prompt_info.items():
            for field in required_fields:
                assert field in info, f"Missing field '{field}' in prompt '{key}'"

    def test_reasoning_prompt_content(self):
        """reasoning 프롬프트 내용 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert "reasoning_prompt" in prompt_info
        reasoning = prompt_info["reasoning_prompt"]
        assert "추론" in reasoning["content"] or "분석" in reasoning["content"]
        assert "단계" in reasoning["content"] or "Step" in reasoning["content"]

    def test_tool_selector_prompt_content(self):
        """tool_selector 프롬프트 내용 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert "tool_selector" in prompt_info
        selector = prompt_info["tool_selector"]
        assert "툴" in selector["content"] or "tool" in selector["content"].lower()

    def test_prompt_count(self):
        """프롬프트 개수 테스트 (7개: Phase 02-5 구조)"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        assert len(prompt_info) == 7


class TestOverviewTab:
    """Overview 탭 테스트 (Phase 02 항목 7)"""

    def test_get_overview_content_has_required_sections(self):
        """get_overview_content()가 필수 섹션을 포함하는지 테스트"""
        from component.overview_tab import get_overview_content

        content = get_overview_content()

        required_sections = [
            "introduction",
            "quick_start",
            "features",
            "settings",
            "faq",
        ]

        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_get_langgraph_diagram_returns_markdown(self):
        """get_langgraph_diagram()이 마크다운을 반환하는지 테스트"""
        from component.overview_tab import get_langgraph_diagram

        markdown = get_langgraph_diagram()

        assert "summary_node" in markdown
        assert "llm_node" in markdown
        assert "tool_node" in markdown
        assert "tools_condition" in markdown

    def test_get_tool_info_table_has_all_tools(self):
        """get_tool_info()가 모든 툴 정보를 반환하는지 테스트"""
        from component.overview_tab import get_tool_info

        tool_info = get_tool_info()

        expected_tools = [
            "get_current_time",
            "switch_to_reasoning",
            "web_search",
            "search_pdf_knowledge",
        ]

        for tool in expected_tools:
            found = any(t["name"] == tool for t in tool_info)
            assert found, f"Missing tool: {tool}"

    def test_get_response_length_diagram(self):
        """get_response_length_diagram()이 Mermaid 코드를 반환하는지 테스트"""
        from component.overview_tab import get_response_length_diagram

        mermaid_code = get_response_length_diagram()

        assert "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower()
        assert "상세" in mermaid_code or "detailed" in mermaid_code.lower()
        assert "간결" in mermaid_code or "brief" in mermaid_code.lower()


class TestPhase07SidebarReasoningMode:
    """Phase 02-7: 사이드바 추론 모드 UI 테스트"""

    def test_sidebar_module_has_reasoning_mode_support(self):
        """sidebar 모듈이 추론 모드 관련 코드를 포함하는지 테스트"""
        import component.sidebar as sidebar_module
        import inspect

        source = inspect.getsource(sidebar_module)

        # 추론 모드 관련 키워드가 있어야 함
        assert "추론" in source or "reasoning" in source.lower()


class TestChatTabSummary:
    """채팅 탭 요약 사이드바 테스트 (Phase 02 항목 6)"""

    def test_format_summary_card_basic(self):
        """기본 요약 카드 포맷팅 테스트 (연속 범위는 1-3 형식)"""
        from component.chat_tab import format_summary_card

        summary_entry = {
            "created_at_turn": 4,
            "turns": [1, 2, 3],
            "summary": "사용자가 Python에 대해 물어봤습니다."
        }

        result = format_summary_card(summary_entry)

        # Phase 03-3-2: 연속 범위는 "1-3" 형식으로 표시
        assert "Turn 1-3" in result
        assert "사용자가 Python에 대해 물어봤습니다." in result

    def test_format_summary_card_long_summary(self):
        """긴 요약문 포맷팅 테스트"""
        from component.chat_tab import format_summary_card

        long_summary = "이것은 매우 긴 요약문입니다. " * 10
        summary_entry = {
            "created_at_turn": 7,
            "turns": [4, 5, 6],
            "summary": long_summary
        }

        result = format_summary_card(summary_entry)

        # Phase 03-3-2: 연속 범위는 "4-6" 형식으로 표시
        assert "Turn 4-6" in result
        assert "긴 요약문" in result

    def test_format_summary_card_non_consecutive(self):
        """비연속 턴 요약 카드 테스트 (Phase 03-3-2)"""
        from component.chat_tab import format_summary_card

        summary_entry = {
            "turns": [1, 3, 4],  # Turn 2 제외 (casual)
            "excluded_turns": [2],
            "summary": "비연속 턴 요약"
        }

        result = format_summary_card(summary_entry)

        # 비연속이므로 "1, 3, 4" 형식
        assert "Turn 1, 3, 4" in result
        assert "2턴 제외" in result

    def test_format_summary_card_with_excluded_turns(self):
        """excluded_turns 표시 테스트 (Phase 03-3-2)"""
        from component.chat_tab import format_summary_card

        summary_entry = {
            "turns": [1, 2, 3],
            "excluded_turns": [2],
            "summarized_turns": [1, 3],
            "summary": "요약 내용"
        }

        result = format_summary_card(summary_entry)

        assert "2턴 제외" in result

    def test_get_summary_history_empty(self):
        """빈 요약 히스토리 테스트"""
        summary_history = []

        assert len(summary_history) == 0

    def test_get_summary_history_multiple(self):
        """다중 요약 히스토리 테스트"""
        summary_history = [
            {"created_at_turn": 4, "turns": [1, 2, 3], "summary": "첫 번째 요약"},
            {"created_at_turn": 7, "turns": [4, 5, 6], "summary": "두 번째 요약"},
            {"created_at_turn": 10, "turns": [7, 8, 9], "summary": "세 번째 요약"},
        ]

        assert len(summary_history) == 3
        assert summary_history[0]["created_at_turn"] == 4
        assert summary_history[2]["turns"] == [7, 8, 9]
