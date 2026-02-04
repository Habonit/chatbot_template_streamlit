"""Phase 02-7: 추론 모드 감지기

사용자 입력을 분석하여 적절한 응답 모드를 결정합니다.
- casual: 일상적 대화 (툴 사용 불필요)
- reasoning: 추론 모드 필요 (gemini-2.5-pro 사용)
- normal: 일반 모드
"""
import re
from typing import Literal

# 추론 모드가 필요한 패턴
REASONING_PATTERNS = [
    # 비교/분석 키워드
    r"비교|분석|차이점|장단점|pros.?cons",
    # 설명 요구
    r"왜|어떻게|원인|이유|메커니즘|원리",
    # 단계별 설명
    r"단계별|step.?by.?step|하나씩|순서대로",
    # 수학/논리 (한글 word boundary 제거)
    r"\d+\s*[\+\-\*\/\=]|계산|수식|공식|증명|논리",
    # 전문 분석
    r"심층|심화|깊이|자세히|상세히|구체적으로",
]

# 일상적 대화 패턴
CASUAL_PATTERNS = [
    # 감탄사/인사
    r"^(오호|아하|그렇구나|알겠|네|응|아|오|음|흠|ㅎㅎ|ㅋㅋ|안녕|반가워|고마워|감사)",
    # 단순 긍정/부정
    r"^(좋아|싫어|괜찮|그래|됐어|알았어|오케이|ok|ㅇㅋ)",
]

# 질문/요청 패턴 (이런 패턴이 있으면 casual이 아님)
REQUEST_PATTERNS = [
    r"\?$",  # 물음표로 끝남
    r"해줘|알려줘|찾아줘|검색|설명|뭐야|어때",  # 요청/질문
]


def detect_reasoning_need(user_input: str) -> Literal["reasoning", "casual", "normal"]:
    """질문 유형을 감지하여 적절한 모드 반환

    Args:
        user_input: 사용자 입력

    Returns:
        "reasoning": 추론 모드 필요
        "casual": 일상적 대화 (툴 사용 불필요)
        "normal": 일반 모드
    """
    if not user_input or not user_input.strip():
        return "casual"

    user_input = user_input.strip()

    # 0. 질문/요청 패턴이 있으면 casual이 아님
    is_request = any(re.search(p, user_input, re.IGNORECASE) for p in REQUEST_PATTERNS)

    # 1. 추론 모드 필요 체크 (먼저)
    for pattern in REASONING_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "reasoning"

    # 2. 질문/요청이면 normal
    if is_request:
        return "normal"

    # 3. 일상적 대화 체크
    for pattern in CASUAL_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return "casual"

    # 4. 짧은 입력 (5자 이하, 질문/요청이 아닌 경우)
    if len(user_input) <= 5:
        return "casual"

    return "normal"
