"""Phase 03-3-3: 버그 수정 테스트

발견된 문제점 수정 검증:
1. extras/signature 노출 (content가 list인 경우)
2. casual 감지 일관성
3. 요약 품질 개선
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage


class TestExtractTextFromContent:
    """AIMessage.content 텍스트 추출 테스트"""

    def test_string_content(self):
        """문자열 content는 그대로 반환"""
        from service.react_graph import extract_text_from_content

        result = extract_text_from_content("Hello, world!")
        assert result == "Hello, world!"

    def test_list_with_text_dict(self):
        """리스트 content에서 text 타입 dict 추출"""
        from service.react_graph import extract_text_from_content

        content = [
            {"type": "text", "text": "첫 번째 텍스트"},
            {"type": "text", "text": " 두 번째 텍스트"},
        ]
        result = extract_text_from_content(content)
        assert result == "첫 번째 텍스트 두 번째 텍스트"

    def test_list_with_extras_signature(self):
        """extras/signature가 포함된 dict는 무시"""
        from service.react_graph import extract_text_from_content

        content = [
            {
                "type": "text",
                "text": "실제 응답 텍스트",
                "extras": {"signature": "CoENAb4+9vtjQ1ku..."},
            }
        ]
        result = extract_text_from_content(content)
        assert result == "실제 응답 텍스트"
        assert "extras" not in result
        assert "signature" not in result

    def test_list_with_mixed_types(self):
        """문자열과 dict가 섞인 리스트"""
        from service.react_graph import extract_text_from_content

        content = [
            "문자열 요소",
            {"type": "text", "text": "dict 요소"},
            {"type": "image_url", "image_url": {"url": "http://..."}},
        ]
        result = extract_text_from_content(content)
        assert result == "문자열 요소dict 요소"

    def test_none_content(self):
        """None은 빈 문자열 반환"""
        from service.react_graph import extract_text_from_content

        result = extract_text_from_content(None)
        assert result == ""

    def test_empty_list(self):
        """빈 리스트는 빈 문자열 반환"""
        from service.react_graph import extract_text_from_content

        result = extract_text_from_content([])
        assert result == ""


class TestCasualDetectionConsistency:
    """Casual 감지 일관성 테스트"""

    def test_tool_request_is_not_casual(self):
        """'툴을 사용해'는 casual이 아님"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need("툴을 사용해")
        assert result != "casual", "'툴을 사용해'는 명시적 요청이므로 casual이 아님"

    def test_date_question_is_not_casual(self):
        """날짜 질문은 casual이 아님"""
        from service.reasoning_detector import detect_reasoning_need

        test_cases = [
            "오늘 몇월 며칠이야",
            "지금 몇 시야",
            "오늘 몇년도야",
            "현재 몇 분이야",
        ]
        for case in test_cases:
            result = detect_reasoning_need(case)
            assert result != "casual", f"'{case}'는 시간/날짜 질문이므로 casual이 아님"

    def test_question_words_not_casual(self):
        """의문사가 있는 문장은 casual이 아님"""
        from service.reasoning_detector import detect_reasoning_need

        test_cases = [
            "언제 발매됐어",
            "어디서 샀어",
            "누구 노래야",
            "무엇을 검색해야 해",
            "얼마야",
        ]
        for case in test_cases:
            result = detect_reasoning_need(case)
            assert result != "casual", f"'{case}'는 질문이므로 casual이 아님"

    def test_true_casual_is_casual(self):
        """진짜 casual은 casual로 감지"""
        from service.reasoning_detector import detect_reasoning_need

        test_cases = [
            "안녕",
            "반가워",
            "고마워",
            "ㅎㅎ",
            "오호",
            "그렇구나",
        ]
        for case in test_cases:
            result = detect_reasoning_need(case)
            assert result == "casual", f"'{case}'는 casual이어야 함"


class TestSummaryPromptImprovement:
    """요약 프롬프트 개선 테스트"""

    def test_prompt_has_preserve_rules(self):
        """프롬프트에 보존 규칙 포함"""
        from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT

        assert "고유명사" in SUMMARY_GENERATOR_PROMPT
        assert "사람 이름" in SUMMARY_GENERATOR_PROMPT
        assert "생략 금지" in SUMMARY_GENERATOR_PROMPT

    def test_prompt_has_minimum_length(self):
        """프롬프트에 최소 길이 규칙 포함"""
        from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT

        assert "최소" in SUMMARY_GENERATOR_PROMPT
        assert "150자" in SUMMARY_GENERATOR_PROMPT  # Phase 03-3-3: 100자 → 150자

    def test_prompt_has_examples(self):
        """프롬프트에 예시 포함"""
        from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT

        assert "김상민" in SUMMARY_GENERATOR_PROMPT or "LangChain" in SUMMARY_GENERATOR_PROMPT


class TestInvokeLLMWithTokenTracking:
    """_invoke_llm_with_token_tracking 반환값 테스트"""

    def test_returns_clean_text(self):
        """list content도 깨끗한 텍스트로 반환"""
        from service.react_graph import extract_text_from_content

        gemini_response = [
            {
                "type": "text",
                "text": "AI 응답입니다.",
                "extras": {"signature": "abc123..."},
            }
        ]
        result = extract_text_from_content(gemini_response)
        assert result == "AI 응답입니다."
        assert "signature" not in result


class TestRequestPatterns:
    """REQUEST_PATTERNS 테스트"""

    def test_imperative_forms(self):
        """명령형 패턴 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        assert detect_reasoning_need("한번 해봐") != "casual"
        assert detect_reasoning_need("검색 도구 사용해") != "casual"

    def test_time_patterns(self):
        """시간/날짜 패턴 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        assert detect_reasoning_need("몇 시야") != "casual"
        assert detect_reasoning_need("몇월이야") != "casual"
        assert detect_reasoning_need("몇년도야") != "casual"
        assert detect_reasoning_need("몇 개야") != "casual"


class TestExcludedTurnsCalculation:
    """Phase 03-3-3: excluded_turns 계산 테스트"""

    def test_turn1_included_in_excluded_when_casual(self):
        """Turn 1이 casual일 때 excluded_turns에 포함"""
        from component.chat_tab import format_summary_card

        # 시뮬레이션: Turn 1이 casual, Turn 2-4가 normal
        # 요약 범위는 Turn 1-4, 실제 요약된 턴은 [2, 3, 4]
        summary_entry = {
            "turns": [1, 2, 3, 4],
            "summarized_turns": [2, 3, 4],
            "excluded_turns": [1],  # Turn 1이 casual
            "summary": "테스트 요약입니다.",
        }
        result = format_summary_card(summary_entry)
        assert "Turn 1-4" in result
        assert "1턴 casual" in result

    def test_multiple_excluded_turns(self):
        """여러 턴이 제외된 경우"""
        from component.chat_tab import format_summary_card

        summary_entry = {
            "turns": [1, 2, 3, 4, 5],
            "summarized_turns": [2, 4, 5],
            "excluded_turns": [1, 3],  # Turn 1, 3이 casual
            "summary": "테스트 요약입니다.",
        }
        result = format_summary_card(summary_entry)
        assert "Turn 1-5" in result
        assert "1, 3턴 casual" in result

    def test_no_excluded_turns(self):
        """제외된 턴이 없는 경우"""
        from component.chat_tab import format_summary_card

        summary_entry = {
            "turns": [1, 2, 3],
            "summarized_turns": [1, 2, 3],
            "excluded_turns": [],
            "summary": "테스트 요약입니다.",
        }
        result = format_summary_card(summary_entry)
        assert "Turn 1-3" in result
        assert "casual" not in result


class TestTurnNumberDisplay:
    """Phase 03-3-3: 턴 번호 표시 테스트"""

    def test_render_chat_tab_has_turn_count_param(self):
        """render_chat_tab이 turn_count 파라미터를 받음"""
        from component.chat_tab import render_chat_tab
        import inspect

        sig = inspect.signature(render_chat_tab)
        params = list(sig.parameters.keys())
        assert "turn_count" in params
