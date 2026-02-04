"""Phase 02-4: 결과 처리기 프롬프트

Result Processor 노드에서 사용하는 프롬프트입니다.
툴 실행 결과를 분석하고 추가 툴 필요 여부를 판단합니다.
"""

RESULT_PROCESSOR_PROMPT = """
당신은 툴 실행 결과를 분석하고, 다음 행동을 결정하는 전문가입니다.

## 역할

1. **결과 해석**: 툴 실행 결과를 이해하고 요약
2. **충분성 판단**: 사용자 질문에 답하기에 충분한 정보가 모였는지 판단
3. **다음 행동 결정**: 추가 툴이 필요한지, 응답 생성으로 넘어갈지 결정

## 판단 기준

### 추가 툴이 필요한 경우
- 사용자 질문에 답하기에 정보가 부족함
- "분석해줘", "비교해줘" 요청이 있는데 reasoning을 안 했음
- 검색 결과가 불충분하거나 신뢰도가 낮음

### 충분한 경우
- 사용자 질문에 답할 수 있는 정보가 모두 모임
- 이미 필요한 모든 툴을 실행함
- 최대 반복 횟수에 도달

## 현재 상태

- 사용자 질문: {user_input}
- 실행한 툴: {tool_history}
- 툴 결과:
{tool_results}
- 반복 횟수: {iteration}/{max_iterations}

## 출력 형식

반드시 아래 JSON 형식으로만 출력하세요:
{{"summary": "현재까지 수집된 정보 요약 (2-3문장)", "needs_more_tools": true 또는 false, "reason": "판단 이유"}}
""".strip()


def get_prompt(
    user_input: str,
    tool_history: list[str],
    tool_results: dict[str, str],
    iteration: int,
    max_iterations: int,
) -> str:
    """결과 처리기 프롬프트 생성

    Args:
        user_input: 사용자 질문
        tool_history: 실행한 툴 목록
        tool_results: 툴별 실행 결과
        iteration: 현재 반복 횟수
        max_iterations: 최대 반복 횟수

    Returns:
        str: 포맷된 프롬프트
    """
    # 툴 결과 포맷팅
    formatted_results = []
    for tool_name, result in tool_results.items():
        formatted_results.append(f"[{tool_name}]\n{result}")
    tool_results_str = "\n\n".join(formatted_results) if formatted_results else "없음"

    return RESULT_PROCESSOR_PROMPT.format(
        user_input=user_input,
        tool_history=tool_history if tool_history else "없음",
        tool_results=tool_results_str,
        iteration=iteration,
        max_iterations=max_iterations,
    )
