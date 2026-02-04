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
