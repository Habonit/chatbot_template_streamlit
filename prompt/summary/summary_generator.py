"""Phase 02-5: 요약 생성 프롬프트"""

SUMMARY_GENERATOR_PROMPT = """
다음 대화 내용을 간결하게 요약하세요.

## 요약 규칙
1. 핵심 정보, 사용자의 요청 사항, 중요한 결정 사항을 포함
2. 200자 이내로 작성
3. 객관적인 어조로 작성
4. 이전 요약이 있으면 통합하여 하나의 요약으로 작성

## 이전 요약
{previous_summary}

## 추가할 대화
{conversation}

## 통합 요약
""".strip()


def get_prompt(
    previous_summary: str,
    conversation: str,
) -> str:
    """요약 생성 프롬프트 반환

    Args:
        previous_summary: 이전 요약본 (없으면 빈 문자열)
        conversation: 요약할 대화 내용

    Returns:
        str: 포맷된 프롬프트
    """
    return SUMMARY_GENERATOR_PROMPT.format(
        previous_summary=previous_summary if previous_summary else "없음",
        conversation=conversation,
    )
