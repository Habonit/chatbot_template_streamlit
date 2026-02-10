"""Phase 02-4: 툴 선택기 프롬프트

Tool Selector 노드에서 사용하는 프롬프트입니다.
사용자 질문을 분석하여 다음에 실행할 툴 하나를 선택합니다.
"""

TOOL_SELECTOR_PROMPT = """
당신은 사용자 질문을 분석하여 **다음에 실행할 툴 하나**를 선택하는 전문가입니다.

## 사용 가능한 도구

| 도구 | 사용 조건 |
|------|----------|
| get_current_time | 현재 시각, 날짜, 오늘 관련 질문 |
| web_search | 최신 정보, 뉴스, 실시간 데이터, "검색해줘" |
| search_pdf_knowledge | "문서에서", "파일에", PDF 관련 질문 |
| reasoning | "왜", "어떻게", "비교", "분석", 복합 질의, 복잡한 추론 |
| none | 더 이상 툴이 필요 없음 |

## 툴을 사용하지 않는 경우 (중요!)

다음과 같은 일상적 대화에는 **반드시 "none"을 선택**하세요:

1. **인사/작별**: "안녕", "반가워", "고마워", "잘가", "수고해" 등
2. **감탄사/맞장구**: "오호", "아하", "그렇구나", "알겠어", "네", "응" 등
3. **단순 긍정/부정**: "좋아", "싫어", "괜찮아", "됐어", "오케이" 등
4. **짧은 감정 표현**: "ㅎㅎ", "ㅋㅋ", "^^", 이모지만 있는 경우
5. **명확한 질문 없이 끝나는 발화**

이런 입력은 정보 검색이나 추론이 필요 없으며,
자연스럽고 편안하게 대화하는 것이 더 적절합니다.

## 규칙

1. **한 번에 하나의 툴만** 선택하세요
2. 이미 실행한 툴은 다시 선택하지 마세요
3. PDF가 업로드되어 있으면 web_search보다 search_pdf_knowledge 우선
4. 충분한 정보가 모였으면 "none" 선택
5. **일상적 대화는 무조건 "none" 선택**

## 현재 상태

- 사용자 질문: {user_input}
- 이미 실행한 툴: {tool_history}
- 현재까지 수집된 정보: {tool_results_summary}
- 업로드된 PDF: {pdf_description}

## 출력 지시

다음에 실행할 도구를 선택하고, 선택 이유를 설명하세요.
""".strip()


def get_prompt(
    user_input: str,
    tool_history: list[str],
    tool_results_summary: str = "",
    pdf_description: str = "",
) -> str:
    """툴 선택기 프롬프트 생성

    Args:
        user_input: 사용자 질문
        tool_history: 이미 실행한 툴 목록
        tool_results_summary: 현재까지 수집된 정보 요약
        pdf_description: 업로드된 PDF 설명

    Returns:
        str: 포맷된 프롬프트
    """
    return TOOL_SELECTOR_PROMPT.format(
        user_input=user_input,
        tool_history=tool_history if tool_history else "없음",
        tool_results_summary=tool_results_summary if tool_results_summary else "없음",
        pdf_description=pdf_description if pdf_description else "없음",
    )
