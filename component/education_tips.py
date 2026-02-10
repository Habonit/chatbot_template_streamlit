"""Phase 05: 교육 팁 모듈 (순수 함수)

"개념이 사용될 때 그 개념을 설명한다" 원칙에 따라
각 기능이 실행되는 시점에 관련 개념을 설명하는 교육 텍스트를 반환합니다.

Streamlit 의존 없음 — 순수 함수로 str 또는 dict만 반환합니다.
"""


def get_prompt_education(actual_prompts: dict) -> dict:
    """LLM에 전송된 실제 프롬프트 교육 정보

    Args:
        actual_prompts: {"system_prompt": str, "user_messages_count": int, "context_turns": int, ...}

    Returns:
        {"title": str, "system_prompt_preview": str, "explanation": str}
        빈 입력이면 빈 dict 반환
    """
    if not actual_prompts:
        return {}

    system_prompt = actual_prompts.get("system_prompt", "")
    user_messages_count = actual_prompts.get("user_messages_count", 0)
    context_turns = actual_prompts.get("context_turns", 0)

    explanation = (
        "System Prompt Builder는 매 턴마다 동적으로 프롬프트를 구성합니다. "
        f"이번 턴에는 시스템 프롬프트 + 최근 {context_turns}턴의 대화 컨텍스트 + "
        f"{user_messages_count}개의 사용자 메시지가 LLM에 전달되었습니다. "
        "요약 히스토리가 있으면 시스템 프롬프트에 포함되어 장기 기억 역할을 합니다."
    )

    return {
        "title": "프롬프트 구성",
        "system_prompt_preview": system_prompt,
        "explanation": explanation,
    }


def get_streaming_education() -> dict:
    """스트리밍 개념 교육 정보

    Returns:
        {"title": str, "explanation": str, "terms": list[dict]}
    """
    return {
        "title": "스트리밍 응답",
        "explanation": (
            "스트리밍은 LLM의 응답을 토큰 단위로 실시간 전달하는 방식입니다. "
            "전체 응답이 생성될 때까지 기다리지 않고, 생성되는 즉시 사용자에게 표시합니다."
        ),
        "terms": [
            {
                "term": "TTFT",
                "desc": "Time To First Token — 요청 후 첫 번째 토큰이 도착하는 데 걸리는 시간",
            },
            {
                "term": "Chunk",
                "desc": "스트리밍에서 한 번에 전달되는 데이터 조각 (보통 1~수 개의 토큰)",
            },
            {
                "term": "SSE",
                "desc": "Server-Sent Events — 서버에서 클라이언트로 실시간 데이터를 보내는 HTTP 프로토콜",
            },
        ],
    }


def get_summary_education(summary_triggered: bool, summary_history: list) -> dict:
    """요약/컨텍스트 관리 교육 정보

    Args:
        summary_triggered: 이번 턴에 요약이 트리거되었는지
        summary_history: 요약 히스토리 리스트

    Returns:
        {"title": str, "explanation": str, "memory_diagram": str}
        summary_triggered=False이고 summary_history 비어있으면 빈 dict
    """
    if not summary_triggered and not summary_history:
        return {}

    if summary_triggered:
        explanation = (
            "이번 턴에서 대화 요약이 생성되었습니다. "
            "단기 기억(최근 3턴의 raw 메시지)은 그대로 유지되고, "
            "이전 대화는 요약되어 장기 기억(시스템 프롬프트 내 요약)으로 전환됩니다. "
            "이를 통해 컨텍스트 윈도우를 효율적으로 관리합니다."
        )
    else:
        explanation = (
            "이전 대화의 요약이 시스템 프롬프트에 포함되어 장기 기억 역할을 합니다. "
            "LLM은 최근 3턴의 원본 메시지(단기 기억)와 요약(장기 기억)을 함께 참조합니다."
        )

    memory_diagram = (
        "[장기 기억: 요약 히스토리] → System Prompt\n"
        "[단기 기억: 최근 3턴 raw] → Context Messages\n"
        "[현재 턴: 사용자 입력] → User Message"
    )

    return {
        "title": "컨텍스트 관리",
        "explanation": explanation,
        "memory_diagram": memory_diagram,
    }


def get_thinking_education(thinking_budget: int, thought_process: str) -> dict:
    """추론/Thinking 토큰 교육 정보

    Args:
        thinking_budget: 사고 토큰 예산 (0이면 비활성화)
        thought_process: 실제 사고 과정 텍스트

    Returns:
        {"title": str, "explanation": str}
        thinking_budget <= 0이면 빈 dict
    """
    if thinking_budget <= 0:
        return {}

    explanation = (
        f"Thinking Budget {thinking_budget} 토큰이 reasoning 도구 전용으로 할당되었습니다. "
        "일반 대화나 다른 도구 호출에서는 thinking 토큰을 소비하지 않고, "
        "reasoning 도구가 호출될 때만 Gemini의 내부 사고 과정이 활성화됩니다. "
        "이는 비용 효율적으로 복잡한 추론 품질을 높이는 전략입니다."
    )

    if thought_process:
        explanation += " 이번 턴에서 reasoning 도구의 사고 과정이 캡처되었습니다."

    return {
        "title": "Reasoning Thinking",
        "explanation": explanation,
    }


def get_tool_education(tool_names: list[str]) -> dict:
    """도구 사용 교육 정보

    Args:
        tool_names: 사용된 도구 이름 리스트

    Returns:
        {"title": str, "explanations": list[dict]}
        빈 리스트면 빈 dict
    """
    if not tool_names:
        return {}

    tool_info = {
        "reasoning": {
            "tool": "reasoning",
            "desc": (
                "추론 체인(Chain of Thought) 도구입니다. "
                "복잡한 질문을 단계별로 분해하여 논리적으로 답변합니다."
            ),
        },
        "web_search": {
            "tool": "web_search",
            "desc": (
                "웹 검색 에이전트입니다. Tavily API를 통해 실시간 웹 정보를 검색하고, "
                "검색 결과를 LLM이 종합하여 답변합니다."
            ),
        },
        "search_pdf_knowledge": {
            "tool": "search_pdf_knowledge",
            "desc": (
                "RAG(Retrieval-Augmented Generation) 파이프라인입니다. "
                "업로드된 PDF를 임베딩 벡터로 변환하고, 질문과 유사한 청크를 "
                "코사인 유사도로 검색하여 관련 정보를 제공합니다."
            ),
        },
        "get_current_time": {
            "tool": "get_current_time",
            "desc": (
                "실시간 시간 정보 도구입니다. LLM의 학습 데이터에는 현재 시간이 "
                "포함되지 않으므로, 이 도구를 통해 정확한 시간 정보에 접근합니다."
            ),
        },
    }

    explanations = []
    seen = set()
    for name in tool_names:
        if name in tool_info and name not in seen:
            explanations.append(tool_info[name])
            seen.add(name)
        elif name not in seen:
            explanations.append({
                "tool": name,
                "desc": f"'{name}' 도구가 호출되었습니다.",
            })
            seen.add(name)

    return {
        "title": "도구 사용",
        "explanations": explanations,
    }


def get_parameter_help(param_name: str) -> str:
    """파라미터별 교육적 help 텍스트

    Args:
        param_name: "temperature", "top_p", "max_output_tokens", "seed",
                    "thinking_budget", "compression_rate"

    Returns:
        교육적 help 텍스트 문자열
    """
    helps = {
        "temperature": (
            "확률 분포의 '온도'를 조절합니다. "
            "낮은 값(0.0~0.3)은 가장 확률 높은 토큰을 선택하여 일관된 답변을, "
            "높은 값(0.7~2.0)은 다양한 토큰에 기회를 주어 창의적인 답변을 생성합니다."
        ),
        "top_p": (
            "Nucleus Sampling — 누적 확률이 top_p에 도달할 때까지의 토큰만 후보로 사용합니다. "
            "0.9면 상위 90% 확률 토큰만 고려합니다. "
            "Temperature와 함께 사용되어 출력 다양성을 제어합니다."
        ),
        "max_output_tokens": (
            "LLM이 한 번에 생성할 수 있는 최대 토큰 수입니다. "
            "1토큰 ≈ 한국어 1~2글자, 영어 4글자. "
            "토큰 수가 많을수록 긴 답변이 가능하지만 비용이 증가합니다."
        ),
        "seed": (
            "동일한 seed 값을 사용하면 같은 입력에 대해 재현 가능한 응답을 생성합니다. "
            "-1은 매번 다른 결과를 생성합니다. "
            "디버깅이나 일관된 테스트에 유용합니다."
        ),
        "thinking_budget": (
            "모델의 '사고 과정'에 할당할 토큰 예산입니다. "
            "활성화하면 모델이 답변 전에 내부적으로 추론 과정을 거칩니다. "
            "thinking 토큰도 출력 토큰으로 소비되므로 비용에 영향을 줍니다."
        ),
        "compression_rate": (
            "대화 요약의 압축 비율입니다. "
            "0.1이면 원문의 10%로 압축(짧은 요약), 0.5면 50%로 압축(상세한 요약). "
            "컨텍스트 윈도우 효율과 정보 보존 사이의 트레이드오프를 조절합니다."
        ),
    }

    return helps.get(param_name, "")
