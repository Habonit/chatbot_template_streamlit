"""평가자 함수 정의

langsmith 0.6.8은 (inputs, outputs, reference_outputs) 시그니처를 자동 인식하여
run.outputs → outputs, example.outputs → reference_outputs로 매핑한다.

각 함수는 {"key": str, "score": float, "comment": str} 형태의 dict를 반환한다.
"""


def tool_usage_correct(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """도구 사용 정확도 평가

    - expected_tool이 실제 사용되었는지 확인
    - unexpected_tools가 사용되지 않았는지 확인
    """
    expected_tool = reference_outputs.get("expected_tool")
    unexpected_tools = reference_outputs.get("unexpected_tools", [])
    actual_tools = outputs.get("tool_history", [])

    if expected_tool is None:
        score = 1 if len(actual_tools) == 0 else 0
        comment = "도구 미사용 OK" if score else f"불필요한 도구 사용: {actual_tools}"
    else:
        if expected_tool not in actual_tools:
            score = 0
            comment = f"기대 도구 미사용. 기대: {expected_tool}, 실제: {actual_tools}"
        elif unexpected_tools and any(t in actual_tools for t in unexpected_tools):
            used_unexpected = [t for t in unexpected_tools if t in actual_tools]
            score = 0.5
            comment = f"기대 도구 사용했으나 불필요 도구도 사용: {used_unexpected}"
        else:
            score = 1
            comment = f"기대: {expected_tool}, 실제: {actual_tools}"

    return {"key": "tool_usage_correct", "score": score, "comment": comment}


def answer_contains_keywords(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """응답에 필수 키워드 포함 여부 평가"""
    expected_keywords = reference_outputs.get("answer_contains", [])
    answer = outputs.get("text", "")

    if not expected_keywords:
        return {
            "key": "answer_contains_keywords",
            "score": 1.0,
            "comment": "키워드 검사 불필요",
        }

    found = [kw for kw in expected_keywords if kw in answer]
    score = len(found) / len(expected_keywords)

    return {
        "key": "answer_contains_keywords",
        "score": score,
        "comment": f"발견: {found}, 누락: {set(expected_keywords) - set(found)}",
    }


def response_not_empty(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """응답이 비어있지 않은지 확인"""
    answer = outputs.get("text", "")
    score = 1 if answer.strip() else 0
    return {
        "key": "response_not_empty",
        "score": score,
        "comment": "응답 있음" if score else "빈 응답",
    }


def no_error(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """에러 없이 완료되었는지 확인"""
    error = outputs.get("error")
    score = 1 if error is None else 0
    return {
        "key": "no_error",
        "score": score,
        "comment": "정상 완료" if score else f"에러: {error}",
    }


def token_efficiency(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """토큰 효율성 평가"""
    from evaluation.config import MAX_TOKENS_PER_RESPONSE

    total_tokens = outputs.get("total_tokens", 0)

    if total_tokens <= MAX_TOKENS_PER_RESPONSE:
        score = 1.0
    else:
        score = max(
            0,
            1 - (total_tokens - MAX_TOKENS_PER_RESPONSE) / MAX_TOKENS_PER_RESPONSE,
        )

    return {
        "key": "token_efficiency",
        "score": round(score, 3),
        "comment": f"사용 토큰: {total_tokens} / 기준: {MAX_TOKENS_PER_RESPONSE}",
    }
