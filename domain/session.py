from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Session:
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    total_turns: int = 0
    current_summary: str = ""
    pdf_files: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=lambda: {
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "top_p": 0.9,
    })
    token_usage: dict = field(default_factory=lambda: {
        "input": 0,
        "output": 0,
        "total": 0,
    })
    pdf_description: str = ""
    summary_history: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            total_turns=data.get("total_turns", 0),
            current_summary=data.get("current_summary", ""),
            pdf_files=data.get("pdf_files", []),
            settings=data.get("settings", {}),
            token_usage=data.get("token_usage", {"input": 0, "output": 0, "total": 0}),
            pdf_description=data.get("pdf_description", ""),
            summary_history=data.get("summary_history", []),
        )

    @staticmethod
    def generate_id() -> str:
        return datetime.now().strftime("%Y%m%d%H%M")
