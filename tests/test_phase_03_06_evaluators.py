"""Phase 03-6: 평가자(evaluators) 단위 테스트"""
import pytest


class TestToolUsageCorrect:
    """tool_usage_correct 평가자 테스트"""

    def test_correct_tool_used(self):
        """기대 도구가 사용되었으면 score=1"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={"question": "지금 몇 시야?"},
            outputs={"tool_history": ["get_current_time"]},
            reference_outputs={"expected_tool": "get_current_time"},
        )
        assert result["score"] == 1
        assert "key" in result
        assert result["key"] == "tool_usage_correct"

    def test_no_tool_expected_none_used(self):
        """도구 미사용이 기대되고 실제로 미사용이면 score=1"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={"question": "안녕하세요"},
            outputs={"tool_history": []},
            reference_outputs={"expected_tool": None},
        )
        assert result["score"] == 1

    def test_no_tool_expected_but_used(self):
        """도구 미사용이 기대되었지만 사용했으면 score=0"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={"question": "안녕"},
            outputs={"tool_history": ["web_search"]},
            reference_outputs={"expected_tool": None},
        )
        assert result["score"] == 0

    def test_expected_tool_missing(self):
        """기대 도구가 사용되지 않았으면 score=0"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={"question": "최신 뉴스"},
            outputs={"tool_history": []},
            reference_outputs={"expected_tool": "web_search"},
        )
        assert result["score"] == 0

    def test_unexpected_tool_penalty(self):
        """기대 도구 사용 + unexpected 도구도 사용하면 score=0.5"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={"question": "지금 몇 시야?"},
            outputs={"tool_history": ["get_current_time", "web_search"]},
            reference_outputs={
                "expected_tool": "get_current_time",
                "unexpected_tools": ["web_search"],
            },
        )
        assert result["score"] == 0.5

    def test_result_has_comment(self):
        """결과에 comment 필드가 포함되는지 확인"""
        from evaluation.evaluators import tool_usage_correct

        result = tool_usage_correct(
            inputs={},
            outputs={"tool_history": ["get_current_time"]},
            reference_outputs={"expected_tool": "get_current_time"},
        )
        assert "comment" in result


class TestAnswerContainsKeywords:
    """answer_contains_keywords 평가자 테스트"""

    def test_all_keywords_found(self):
        """모든 키워드가 포함되면 score=1.0"""
        from evaluation.evaluators import answer_contains_keywords

        result = answer_contains_keywords(
            inputs={},
            outputs={"text": "지금 시간은 14시 30분입니다."},
            reference_outputs={"answer_contains": ["시", "분"]},
        )
        assert result["score"] == 1.0
        assert result["key"] == "answer_contains_keywords"

    def test_partial_keywords(self):
        """일부 키워드만 포함되면 부분 점수"""
        from evaluation.evaluators import answer_contains_keywords

        result = answer_contains_keywords(
            inputs={},
            outputs={"text": "현재 14시입니다."},
            reference_outputs={"answer_contains": ["시", "분"]},
        )
        assert result["score"] == 0.5

    def test_no_keywords_expected(self):
        """키워드 검사가 불필요하면 score=1.0"""
        from evaluation.evaluators import answer_contains_keywords

        result = answer_contains_keywords(
            inputs={},
            outputs={"text": "안녕하세요!"},
            reference_outputs={"answer_contains": []},
        )
        assert result["score"] == 1.0

    def test_no_keywords_found(self):
        """키워드가 하나도 없으면 score=0.0"""
        from evaluation.evaluators import answer_contains_keywords

        result = answer_contains_keywords(
            inputs={},
            outputs={"text": "모르겠습니다."},
            reference_outputs={"answer_contains": ["시", "분"]},
        )
        assert result["score"] == 0.0


class TestResponseNotEmpty:
    """response_not_empty 평가자 테스트"""

    def test_non_empty_response(self):
        """응답이 있으면 score=1"""
        from evaluation.evaluators import response_not_empty

        result = response_not_empty(
            inputs={},
            outputs={"text": "답변입니다."},
            reference_outputs={},
        )
        assert result["score"] == 1
        assert result["key"] == "response_not_empty"

    def test_empty_response(self):
        """빈 응답이면 score=0"""
        from evaluation.evaluators import response_not_empty

        result = response_not_empty(
            inputs={},
            outputs={"text": ""},
            reference_outputs={},
        )
        assert result["score"] == 0

    def test_whitespace_only_response(self):
        """공백만 있는 응답이면 score=0"""
        from evaluation.evaluators import response_not_empty

        result = response_not_empty(
            inputs={},
            outputs={"text": "   "},
            reference_outputs={},
        )
        assert result["score"] == 0

    def test_missing_text_field(self):
        """text 필드가 없으면 score=0"""
        from evaluation.evaluators import response_not_empty

        result = response_not_empty(
            inputs={},
            outputs={},
            reference_outputs={},
        )
        assert result["score"] == 0


class TestNoError:
    """no_error 평가자 테스트"""

    def test_no_error(self):
        """에러 없으면 score=1"""
        from evaluation.evaluators import no_error

        result = no_error(
            inputs={},
            outputs={"error": None},
            reference_outputs={},
        )
        assert result["score"] == 1
        assert result["key"] == "no_error"

    def test_with_error(self):
        """에러 있으면 score=0"""
        from evaluation.evaluators import no_error

        result = no_error(
            inputs={},
            outputs={"error": "API 호출 실패"},
            reference_outputs={},
        )
        assert result["score"] == 0
        assert "API 호출 실패" in result["comment"]

    def test_missing_error_field(self):
        """error 필드가 없으면 (None과 동일) score=1"""
        from evaluation.evaluators import no_error

        result = no_error(
            inputs={},
            outputs={},
            reference_outputs={},
        )
        assert result["score"] == 1


class TestTokenEfficiency:
    """token_efficiency 평가자 테스트"""

    def test_within_limit(self):
        """토큰이 기준 이하면 score=1.0"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 3000},
            reference_outputs={},
        )
        assert result["score"] == 1.0
        assert result["key"] == "token_efficiency"

    def test_over_limit(self):
        """토큰이 기준 초과면 score < 1.0"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 7500},
            reference_outputs={},
        )
        assert result["score"] < 1.0
        assert result["score"] > 0

    def test_exactly_at_limit(self):
        """토큰이 정확히 기준값이면 score=1.0"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 5000},
            reference_outputs={},
        )
        assert result["score"] == 1.0

    def test_zero_tokens(self):
        """토큰이 0이면 score=1.0"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 0},
            reference_outputs={},
        )
        assert result["score"] == 1.0

    def test_very_high_tokens(self):
        """토큰이 매우 많으면 score=0"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 10000},
            reference_outputs={},
        )
        assert result["score"] == 0

    def test_result_has_comment_with_token_info(self):
        """결과 comment에 토큰 정보 포함"""
        from evaluation.evaluators import token_efficiency

        result = token_efficiency(
            inputs={},
            outputs={"total_tokens": 3000},
            reference_outputs={},
        )
        assert "3000" in result["comment"]
        assert "5000" in result["comment"]
