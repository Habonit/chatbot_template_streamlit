"""Phase 03-2: Structured Output 모델 정의

LLM 응답을 구조화하기 위한 Pydantic 모델
- with_structured_output()에서 사용
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ToolSelectorOutput(BaseModel):
    """Tool Selector의 구조화된 출력"""

    selected_tool: Literal[
        "get_current_time",
        "web_search",
        "search_pdf_knowledge",
        "reasoning",
        "none"
    ] = Field(description="선택된 도구 이름")

    reason: str = Field(description="도구 선택 이유")


class ResultProcessorOutput(BaseModel):
    """Result Processor의 구조화된 출력"""

    needs_more_tools: bool = Field(description="추가 도구 호출 필요 여부")

    summary: str = Field(description="현재까지 수집된 정보 요약")

    next_action: Optional[str] = Field(
        default=None,
        description="다음에 수행할 작업 (추가 도구 필요시)"
    )


class ReasoningOutput(BaseModel):
    """Reasoning Tool의 구조화된 출력"""

    thinking_steps: list[str] = Field(
        description="단계별 추론 과정"
    )

    conclusion: str = Field(
        description="최종 결론"
    )

    confidence: Literal["high", "medium", "low"] = Field(
        description="결론에 대한 확신도"
    )
