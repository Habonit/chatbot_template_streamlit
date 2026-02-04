"""Phase 02-4: 프롬프트 패키지 테스트 (TDD)"""
import pytest


class TestToolSelectorPrompt:
    """툴 선택기 프롬프트 테스트"""

    def test_import_tool_selector_prompt(self):
        """툴 선택기 프롬프트 모듈 임포트 테스트"""
        from prompt.selector.tool_selector import TOOL_SELECTOR_PROMPT, get_prompt

        assert TOOL_SELECTOR_PROMPT is not None
        assert callable(get_prompt)

    def test_tool_selector_prompt_contains_tools(self):
        """프롬프트에 사용 가능한 도구 목록이 포함되어 있는지 테스트"""
        from prompt.selector.tool_selector import TOOL_SELECTOR_PROMPT

        assert "get_current_time" in TOOL_SELECTOR_PROMPT
        assert "web_search" in TOOL_SELECTOR_PROMPT
        assert "search_pdf_knowledge" in TOOL_SELECTOR_PROMPT
        assert "reasoning" in TOOL_SELECTOR_PROMPT
        assert "none" in TOOL_SELECTOR_PROMPT

    def test_tool_selector_prompt_contains_casual_rules(self):
        """Phase 02-7: 프롬프트에 일상적 대화 규칙이 포함되어 있는지 테스트"""
        from prompt.selector.tool_selector import TOOL_SELECTOR_PROMPT

        # 일상적 대화 관련 키워드가 있어야 함
        assert "일상" in TOOL_SELECTOR_PROMPT or "대화" in TOOL_SELECTOR_PROMPT
        assert "인사" in TOOL_SELECTOR_PROMPT or "안녕" in TOOL_SELECTOR_PROMPT
        assert "감탄사" in TOOL_SELECTOR_PROMPT or "그렇구나" in TOOL_SELECTOR_PROMPT

    def test_tool_selector_prompt_specifies_none_for_casual(self):
        """Phase 02-7: 일상적 대화에서 none을 선택하라는 지시가 있는지 테스트"""
        from prompt.selector.tool_selector import TOOL_SELECTOR_PROMPT

        # none 선택 관련 지시가 있어야 함
        prompt_lower = TOOL_SELECTOR_PROMPT.lower()
        assert "none" in prompt_lower
        # 일상적 대화와 none의 연관성
        assert "툴을 사용하지 않" in TOOL_SELECTOR_PROMPT or "반드시" in TOOL_SELECTOR_PROMPT

    def test_get_prompt_returns_formatted_string(self):
        """get_prompt 함수가 포맷된 문자열을 반환하는지 테스트"""
        from prompt.selector.tool_selector import get_prompt

        result = get_prompt(
            user_input="파이썬 3.13 새 기능 검색해줘",
            tool_history=["web_search"],
            tool_results_summary="검색 결과: GIL 제거...",
            pdf_description="",
        )

        assert isinstance(result, str)
        assert "파이썬 3.13 새 기능 검색해줘" in result
        assert "web_search" in result

    def test_get_prompt_with_empty_history(self):
        """빈 히스토리로 get_prompt 호출 테스트"""
        from prompt.selector.tool_selector import get_prompt

        result = get_prompt(
            user_input="오늘 날짜 알려줘",
            tool_history=[],
            tool_results_summary="",
            pdf_description="",
        )

        assert "오늘 날짜 알려줘" in result
        assert "[]" in result or "없음" in result.lower() or len(result) > 0


class TestReasoningPrompt:
    """추론 프롬프트 테스트"""

    def test_import_reasoning_prompt(self):
        """추론 프롬프트 모듈 임포트 테스트"""
        from prompt.tools.reasoning import REASONING_PROMPT, get_prompt

        assert REASONING_PROMPT is not None
        assert callable(get_prompt)

    def test_reasoning_prompt_contains_steps(self):
        """프롬프트에 추론 단계가 포함되어 있는지 테스트"""
        from prompt.tools.reasoning import REASONING_PROMPT

        assert "1단계" in REASONING_PROMPT or "문제 분석" in REASONING_PROMPT
        assert "결론" in REASONING_PROMPT

    def test_get_prompt_returns_formatted_string(self):
        """get_prompt 함수가 포맷된 문자열을 반환하는지 테스트"""
        from prompt.tools.reasoning import get_prompt

        result = get_prompt(
            user_input="파이썬과 자바 비교해줘",
            context="파이썬은 동적 타입 언어이고...",
        )

        assert isinstance(result, str)
        assert "파이썬과 자바 비교해줘" in result
        assert "파이썬은 동적 타입 언어이고" in result


class TestResultProcessorPrompt:
    """결과 처리기 프롬프트 테스트"""

    def test_import_result_processor_prompt(self):
        """결과 처리기 프롬프트 모듈 임포트 테스트"""
        from prompt.processor.result_processor import RESULT_PROCESSOR_PROMPT, get_prompt

        assert RESULT_PROCESSOR_PROMPT is not None
        assert callable(get_prompt)

    def test_result_processor_prompt_contains_criteria(self):
        """프롬프트에 판단 기준이 포함되어 있는지 테스트"""
        from prompt.processor.result_processor import RESULT_PROCESSOR_PROMPT

        assert "충분" in RESULT_PROCESSOR_PROMPT or "needs_more_tools" in RESULT_PROCESSOR_PROMPT

    def test_get_prompt_returns_formatted_string(self):
        """get_prompt 함수가 포맷된 문자열을 반환하는지 테스트"""
        from prompt.processor.result_processor import get_prompt

        result = get_prompt(
            user_input="검색해서 분석해줘",
            tool_history=["web_search"],
            tool_results={"web_search": "검색 결과..."},
            iteration=1,
            max_iterations=5,
        )

        assert isinstance(result, str)
        assert "검색해서 분석해줘" in result
        assert "web_search" in result
        assert "1" in result and "5" in result


class TestResponseGeneratorPrompt:
    """최종 응답 생성기 프롬프트 테스트"""

    def test_import_response_generator_prompt(self):
        """응답 생성기 프롬프트 모듈 임포트 테스트"""
        from prompt.response.response_generator import RESPONSE_GENERATOR_PROMPT, get_prompt

        assert RESPONSE_GENERATOR_PROMPT is not None
        assert callable(get_prompt)

    def test_response_generator_prompt_contains_principles(self):
        """프롬프트에 응답 원칙이 포함되어 있는지 테스트"""
        from prompt.response.response_generator import RESPONSE_GENERATOR_PROMPT

        assert "구조화" in RESPONSE_GENERATOR_PROMPT or "한국어" in RESPONSE_GENERATOR_PROMPT

    def test_get_prompt_returns_formatted_string(self):
        """get_prompt 함수가 포맷된 문자열을 반환하는지 테스트"""
        from prompt.response.response_generator import get_prompt

        result = get_prompt(
            user_input="파이썬 3.13 새 기능 알려줘",
            collected_info="GIL 제거, 타입 힌트 개선...",
            processor_summary="검색과 분석 완료",
        )

        assert isinstance(result, str)
        assert "파이썬 3.13 새 기능 알려줘" in result
        assert "GIL 제거" in result

    def test_response_generator_prompt_contains_casual_handling(self):
        """Phase 02-7: 프롬프트에 일상적 대화 처리 원칙이 포함되어 있는지 테스트"""
        from prompt.response.response_generator import RESPONSE_GENERATOR_PROMPT

        # 일상적 대화 관련 키워드가 있어야 함
        assert "일상" in RESPONSE_GENERATOR_PROMPT or "대화" in RESPONSE_GENERATOR_PROMPT
        assert "친근" in RESPONSE_GENERATOR_PROMPT or "자연스럽" in RESPONSE_GENERATOR_PROMPT

    def test_response_generator_prompt_has_casual_examples(self):
        """Phase 02-7: 프롬프트에 일상적 대화 예시가 있는지 테스트"""
        from prompt.response.response_generator import RESPONSE_GENERATOR_PROMPT

        # 감탄사 예시가 포함되어 있어야 함
        assert "안녕" in RESPONSE_GENERATOR_PROMPT or "고마워" in RESPONSE_GENERATOR_PROMPT or "그렇구나" in RESPONSE_GENERATOR_PROMPT


class TestPromptPackageStructure:
    """프롬프트 패키지 구조 테스트"""

    def test_prompt_package_exists(self):
        """prompt 패키지가 존재하는지 테스트"""
        import prompt

        assert prompt is not None

    def test_selector_subpackage_exists(self):
        """prompt.selector 서브패키지가 존재하는지 테스트"""
        import prompt.selector

        assert prompt.selector is not None

    def test_tools_subpackage_exists(self):
        """prompt.tools 서브패키지가 존재하는지 테스트"""
        import prompt.tools

        assert prompt.tools is not None

    def test_processor_subpackage_exists(self):
        """prompt.processor 서브패키지가 존재하는지 테스트"""
        import prompt.processor

        assert prompt.processor is not None

    def test_response_subpackage_exists(self):
        """prompt.response 서브패키지가 존재하는지 테스트"""
        import prompt.response

        assert prompt.response is not None

    def test_summary_subpackage_exists(self):
        """prompt.summary 서브패키지가 존재하는지 테스트"""
        import prompt.summary

        assert prompt.summary is not None


class TestSummaryGeneratorPrompt:
    """Phase 02-5: 요약 생성기 프롬프트 테스트"""

    def test_import_summary_generator_prompt(self):
        """요약 생성기 프롬프트 모듈 임포트 테스트"""
        from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT, get_prompt

        assert SUMMARY_GENERATOR_PROMPT is not None
        assert callable(get_prompt)

    def test_summary_prompt_contains_rules(self):
        """프롬프트에 요약 규칙이 포함되어 있는지 테스트"""
        from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT

        assert "요약" in SUMMARY_GENERATOR_PROMPT
        assert "핵심" in SUMMARY_GENERATOR_PROMPT or "중요" in SUMMARY_GENERATOR_PROMPT

    def test_get_prompt_with_no_previous_summary(self):
        """이전 요약 없이 get_prompt 호출 테스트"""
        from prompt.summary.summary_generator import get_prompt

        result = get_prompt(
            previous_summary="",
            conversation="사용자: 파이썬 알려줘\n어시스턴트: 파이썬은 프로그래밍 언어입니다.",
        )

        assert isinstance(result, str)
        assert "파이썬 알려줘" in result
        assert "없음" in result  # 이전 요약이 없으면 "없음" 표시

    def test_get_prompt_with_previous_summary(self):
        """이전 요약 있을 때 get_prompt 호출 테스트"""
        from prompt.summary.summary_generator import get_prompt

        result = get_prompt(
            previous_summary="사용자가 파이썬에 대해 질문함",
            conversation="사용자: 자바도 알려줘\n어시스턴트: 자바는 객체지향 언어입니다.",
        )

        assert isinstance(result, str)
        assert "파이썬에 대해 질문함" in result
        assert "자바도 알려줘" in result

    def test_get_prompt_format(self):
        """프롬프트 포맷이 올바른지 테스트"""
        from prompt.summary.summary_generator import get_prompt

        result = get_prompt(
            previous_summary="이전 요약 내용",
            conversation="대화 내용",
        )

        # 섹션 구분자 확인
        assert "이전 요약" in result
        assert "대화" in result or "추가" in result
