from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    turn_id: int
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: Optional[str] = None
    function_calls: list = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    # Phase 04: 교육용 메타데이터
    mode: str = "normal"                        # casual / normal / reasoning
    graph_path: list = field(default_factory=list)  # ["summary_node", "llm_node", "tool_node", ...]
    summary_triggered: bool = False
    thought_process: Optional[str] = None
    thinking_budget: int = 0
    is_casual: bool = False
