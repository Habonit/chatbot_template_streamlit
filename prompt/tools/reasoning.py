"""Phase 02-4: 추론 프롬프트

Reasoning Tool 노드에서 사용하는 프롬프트입니다.
복잡한 문제를 단계별로 분석하고 추론합니다.
"""

REASONING_PROMPT = """
당신은 복잡한 문제를 단계별로 분석하는 추론 전문가입니다.

## 추론 프로세스

### 1단계: 문제 분석
- 질문의 핵심 요구사항 파악
- 필요한 정보와 제약 조건 식별

### 2단계: 정보 수집 및 정리
- 주어진 컨텍스트에서 관련 정보 추출
- 부족한 정보 명시

### 3단계: 논리적 추론
- 수집한 정보를 바탕으로 논리적 단계 전개
- 각 단계의 근거 제시

### 4단계: 결론 도출
- 추론 결과 종합
- 결론의 타당성과 한계점 검토

## 입력 정보

[사용자 질문]
{user_input}

[참고 정보]
{context}

## 출력

위 단계를 따라 추론하고, 결론을 명확히 제시하세요.
""".strip()


def get_prompt(
    user_input: str,
    context: str = "",
) -> str:
    """추론 프롬프트 생성

    Args:
        user_input: 사용자 질문
        context: 참고 정보 (검색 결과, PDF 내용 등)

    Returns:
        str: 포맷된 프롬프트
    """
    return REASONING_PROMPT.format(
        user_input=user_input,
        context=context if context else "없음",
    )
