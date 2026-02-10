"""Phase 02-7: 모드 감지기

사용자 입력을 분석하여 적절한 응답 모드를 결정합니다.
- casual: 일상적 대화 (툴 사용 불필요)
- normal: 일반 모드 (도구 사용 포함)
"""
import re
from typing import Literal

# 일상적 대화 패턴
CASUAL_PATTERNS = [
    # 감탄사/인사
    r"^(오호|아하|그렇구나|알겠|네|응|아|오|음|흠|ㅎㅎ|ㅋㅋ|안녕|반가워|고마워|감사|고맙습니다|감사합니다|감사|)",
    # 단순 긍정/부정
    r"^(좋아|싫어|괜찮|그래|됐어|알았어|오케이|ok|ㅇㅋ)",
]

# 질문/요청 패턴 (이런 패턴이 있으면 casual이 아님)
# Phase 03-3-3: 패턴 강화
REQUEST_PATTERNS = [
    r"\?$",  # 물음표로 끝남
    r"해줘|알려줘|찾아줘|검색|설명|뭐야|어때",  # 요청/질문
    r"해봐|사용해|해라|해\s*$",  # 명령형 ("툴을 사용해")
    r"몇\s*(시|월|년|일|분|초|개|번)",  # 시간/날짜 질문
    r"언제|어디|누구|무엇|얼마",  # 의문사
    r"(이|가)\s*뭐",  # "X가 뭐야" 패턴
]


def detect_reasoning_need(user_input: str) -> Literal["casual", "normal"]:
    """질문 유형을 감지하여 적절한 모드 반환

    Args:
        user_input: 사용자 입력

    Returns:
        "casual": 일상적 대화 (툴 사용 불필요)
        "normal": 일반 모드
    """
    if not user_input or not user_input.strip():
        return "casual"

    user_input = user_input.strip()

    # 0. 질문/요청 패턴이 있으면 casual이 아님
    is_request = any(re.search(p, user_input, re.IGNORECASE) for p in REQUEST_PATTERNS)

    # 1. 질문/요청이면 normal
    if is_request:
        return "normal"

    # 2. 일상적 대화 체크
    for pattern in CASUAL_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "casual"

    # 3. 짧은 입력 (5자 이하, 질문/요청이 아닌 경우)
    if len(user_input) <= 5:
        return "casual"

    return "normal"
