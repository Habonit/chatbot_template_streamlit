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

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_used": self.model_used,
            "function_calls": self.function_calls,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            turn_id=data["turn_id"],
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            model_used=data.get("model_used"),
            function_calls=data.get("function_calls", []),
        )
