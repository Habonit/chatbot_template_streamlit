"""Phase 02-7: 추론 모드 감지기 테스트 (TDD)"""
import pytest


class TestReasoningDetectorImport:
    """모듈 임포트 테스트"""

    def test_import_detect_reasoning_need(self):
        """detect_reasoning_need 함수 임포트 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        assert callable(detect_reasoning_need)

    def test_import_patterns(self):
        """패턴 상수 임포트 테스트"""
        from service.reasoning_detector import REASONING_PATTERNS, CASUAL_PATTERNS

        assert isinstance(REASONING_PATTERNS, list)
        assert isinstance(CASUAL_PATTERNS, list)


class TestCasualDetection:
    """일상적 대화 감지 테스트"""

    @pytest.mark.parametrize("user_input", [
        "오호",
        "그렇구나",
        "아하",
        "알겠어",
        "네",
        "응",
        "ㅎㅎ",
        "ㅋㅋ",
        "안녕",
        "반가워",
        "고마워",
        "감사",
        "좋아",
        "괜찮아",
        "됐어",
        "오케이",
        "ok",
    ])
    def test_casual_inputs_detected(self, user_input):
        """일상적 대화가 casual로 감지되는지 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need(user_input)
        assert result == "casual", f"'{user_input}'은 casual로 감지되어야 함"

    def test_short_input_is_casual(self):
        """짧은 입력이 casual로 감지되는지 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need("ㅇㅋ")
        assert result == "casual"


class TestReasoningDetection:
    """추론 모드 감지 테스트"""

    @pytest.mark.parametrize("user_input", [
        "A와 B를 비교해줘",
        "차이점이 뭐야?",
        "장단점 분석해줘",
        "왜 그런거야?",
        "어떻게 작동해?",
        "원인이 뭐야?",
        "단계별로 설명해줘",
        "자세히 알려줘",
        "1+2+3+4를 계산해줘",
        "논리적으로 증명해줘",
    ])
    def test_reasoning_inputs_detected(self, user_input):
        """추론 질문이 reasoning으로 감지되는지 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need(user_input)
        assert result == "reasoning", f"'{user_input}'은 reasoning으로 감지되어야 함"


class TestNormalDetection:
    """일반 모드 감지 테스트"""

    @pytest.mark.parametrize("user_input", [
        "날씨 어때?",
        "파이썬 알려줘",
        "오늘 뉴스 검색해줘",
        "현재 시간 알려줘",
    ])
    def test_normal_inputs_detected(self, user_input):
        """일반 질문이 normal로 감지되는지 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need(user_input)
        assert result == "normal", f"'{user_input}'은 normal로 감지되어야 함"


class TestReturnType:
    """반환 타입 테스트"""

    def test_returns_literal_type(self):
        """반환값이 Literal 타입인지 테스트"""
        from service.reasoning_detector import detect_reasoning_need

        result = detect_reasoning_need("테스트")
        assert result in ("casual", "reasoning", "normal")
