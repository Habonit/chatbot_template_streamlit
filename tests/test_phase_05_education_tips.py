"""Phase 05 Step 4: education_tips.py 순수 함수 테스트

6개 교육 팁 함수의 반환 구조, 빈 입력 처리, 내용 검증.
"""
import pytest
from component.education_tips import (
    get_prompt_education,
    get_streaming_education,
    get_summary_education,
    get_thinking_education,
    get_tool_education,
    get_parameter_help,
)


class TestGetPromptEducation:
    """get_prompt_education 테스트"""

    def test_returns_dict_with_expected_keys(self):
        """정상 입력 시 title, system_prompt_preview, explanation 반환"""
        result = get_prompt_education({
            "system_prompt": "You are helpful.",
            "user_messages_count": 2,
            "context_turns": 3,
        })
        assert "title" in result
        assert "system_prompt_preview" in result
        assert "explanation" in result

    def test_explanation_contains_context_info(self):
        """explanation에 컨텍스트 턴 수 포함"""
        result = get_prompt_education({
            "system_prompt": "test",
            "user_messages_count": 1,
            "context_turns": 5,
        })
        assert "5턴" in result["explanation"]

    def test_empty_input_returns_empty_dict(self):
        """빈 dict 입력 시 빈 dict 반환"""
        assert get_prompt_education({}) == {}

    def test_none_like_input_returns_empty_dict(self):
        """None 입력 시 빈 dict 반환"""
        assert get_prompt_education(None) == {}


class TestGetStreamingEducation:
    """get_streaming_education 테스트"""

    def test_returns_dict_with_expected_keys(self):
        """title, explanation, terms 반환"""
        result = get_streaming_education()
        assert "title" in result
        assert "explanation" in result
        assert "terms" in result

    def test_terms_contain_ttft(self):
        """terms에 TTFT 포함"""
        result = get_streaming_education()
        term_names = [t["term"] for t in result["terms"]]
        assert "TTFT" in term_names

    def test_terms_contain_chunk(self):
        """terms에 Chunk 포함"""
        result = get_streaming_education()
        term_names = [t["term"] for t in result["terms"]]
        assert "Chunk" in term_names

    def test_terms_contain_sse(self):
        """terms에 SSE 포함"""
        result = get_streaming_education()
        term_names = [t["term"] for t in result["terms"]]
        assert "SSE" in term_names


class TestGetSummaryEducation:
    """get_summary_education 테스트"""

    def test_summary_triggered_returns_info(self):
        """summary_triggered=True 시 교육 정보 반환"""
        result = get_summary_education(True, [])
        assert "title" in result
        assert "explanation" in result
        assert "memory_diagram" in result

    def test_summary_history_exists_returns_info(self):
        """summary_history 있을 때 교육 정보 반환"""
        result = get_summary_education(False, [{"summary": "test"}])
        assert "title" in result
        assert "explanation" in result

    def test_false_and_empty_returns_empty_dict(self):
        """summary_triggered=False + 빈 히스토리 → 빈 dict"""
        assert get_summary_education(False, []) == {}

    def test_triggered_explanation_mentions_summary(self):
        """요약 트리거 시 explanation에 요약 관련 설명 포함"""
        result = get_summary_education(True, [])
        assert "요약" in result["explanation"]


class TestGetThinkingEducation:
    """get_thinking_education 테스트"""

    def test_positive_budget_returns_info(self):
        """thinking_budget > 0 시 교육 정보 반환"""
        result = get_thinking_education(1024, "")
        assert "title" in result
        assert "explanation" in result

    def test_zero_budget_returns_empty_dict(self):
        """thinking_budget == 0 → 빈 dict"""
        assert get_thinking_education(0, "") == {}

    def test_negative_budget_returns_empty_dict(self):
        """thinking_budget < 0 → 빈 dict"""
        assert get_thinking_education(-1, "") == {}

    def test_with_thought_process_adds_info(self):
        """thought_process 있을 때 추가 설명 포함"""
        result = get_thinking_education(1024, "step by step reasoning")
        assert "캡처" in result["explanation"]


class TestGetToolEducation:
    """get_tool_education 테스트"""

    def test_web_search_returns_info(self):
        """web_search 도구 → 교육 정보 반환"""
        result = get_tool_education(["web_search"])
        assert "title" in result
        assert "explanations" in result
        assert len(result["explanations"]) == 1

    def test_search_pdf_knowledge_mentions_rag(self):
        """search_pdf_knowledge → RAG 키워드 포함"""
        result = get_tool_education(["search_pdf_knowledge"])
        desc = result["explanations"][0]["desc"]
        assert "RAG" in desc

    def test_reasoning_mentions_chain_of_thought(self):
        """reasoning 도구 → 추론 체인 설명 포함"""
        result = get_tool_education(["reasoning"])
        desc = result["explanations"][0]["desc"]
        assert "추론" in desc

    def test_get_current_time_returns_info(self):
        """get_current_time 도구 → 교육 정보 반환"""
        result = get_tool_education(["get_current_time"])
        assert len(result["explanations"]) == 1

    def test_empty_list_returns_empty_dict(self):
        """빈 리스트 → 빈 dict"""
        assert get_tool_education([]) == {}

    def test_multiple_tools(self):
        """복수 도구 → 각각의 설명 포함"""
        result = get_tool_education(["web_search", "get_current_time"])
        assert len(result["explanations"]) == 2

    def test_unknown_tool_gets_default_desc(self):
        """알 수 없는 도구 → 기본 설명"""
        result = get_tool_education(["unknown_tool"])
        assert len(result["explanations"]) == 1
        assert "unknown_tool" in result["explanations"][0]["desc"]

    def test_duplicate_tools_deduplicated(self):
        """중복 도구 → 중복 제거"""
        result = get_tool_education(["web_search", "web_search"])
        assert len(result["explanations"]) == 1


class TestGetParameterHelp:
    """get_parameter_help 테스트"""

    def test_temperature_help(self):
        """temperature help 텍스트 비어있지 않음"""
        result = get_parameter_help("temperature")
        assert len(result) > 0
        assert "온도" in result or "확률" in result

    def test_top_p_help(self):
        """top_p help 텍스트 비어있지 않음"""
        result = get_parameter_help("top_p")
        assert len(result) > 0
        assert "Nucleus" in result or "누적" in result

    def test_max_output_tokens_help(self):
        """max_output_tokens help 텍스트 비어있지 않음"""
        result = get_parameter_help("max_output_tokens")
        assert len(result) > 0
        assert "토큰" in result

    def test_seed_help(self):
        """seed help 텍스트 비어있지 않음"""
        result = get_parameter_help("seed")
        assert len(result) > 0
        assert "재현" in result

    def test_thinking_budget_help(self):
        """thinking_budget help 텍스트 비어있지 않음"""
        result = get_parameter_help("thinking_budget")
        assert len(result) > 0
        assert "사고" in result or "추론" in result

    def test_compression_rate_help(self):
        """compression_rate help 텍스트 비어있지 않음"""
        result = get_parameter_help("compression_rate")
        assert len(result) > 0
        assert "압축" in result

    def test_unknown_param_returns_empty(self):
        """알 수 없는 파라미터 → 빈 문자열"""
        assert get_parameter_help("unknown_param") == ""
