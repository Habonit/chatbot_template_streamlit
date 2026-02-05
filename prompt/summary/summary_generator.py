"""Phase 02-5 & 03-3: 요약 생성 프롬프트

Phase 03-3: target_length 파라미터 추가 (compression_rate 기반)
Phase 03-3-3: 핵심 정보 보존 강화
"""

SUMMARY_GENERATOR_PROMPT = """
다음 대화 내용을 요약하세요.

## 요약 규칙 (중요도 순)

### 1. 반드시 보존할 정보 (생략 금지)
- 고유명사: 사람 이름, 브랜드, 제품명, 서비스명 등 (예: "김상민그는감히전설이라고할수있다", "LangChain")
- 구체적 사실: 날짜, 숫자, 버전, 가격 등
- 사용자의 핵심 질문/요청
- 도구 검색 결과의 핵심 정보

### 2. 길이 가이드
- 목표: 약 {target_length}자
- 최소: 100자 이상 (정보 손실 방지)
- 핵심 정보가 많으면 목표 초과 허용

### 3. 형식
- 객관적 서술체 사용
- "사용자가 X에 대해 질문함. Y라는 답변을 받음." 형식

## 이전 요약
{previous_summary}

## 요약할 대화
{conversation}

## 요약
""".strip()


def get_prompt(
    previous_summary: str,
    conversation: str,
    target_length: int = 200,
) -> str:
    """요약 생성 프롬프트 반환

    Args:
        previous_summary: 이전 요약본 (없으면 빈 문자열)
        conversation: 요약할 대화 내용
        target_length: 목표 요약 길이 (글자 수, 기본값 200)

    Returns:
        str: 포맷된 프롬프트
    """
    return SUMMARY_GENERATOR_PROMPT.format(
        previous_summary=previous_summary if previous_summary else "없음",
        conversation=conversation,
        target_length=target_length,
    )
