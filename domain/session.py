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

    def add_turn(self) -> None:
        self.total_turns += 1
        self.last_updated = datetime.now().isoformat()

    def update_summary(self, summary: str) -> None:
        self.current_summary = summary
        self.last_updated = datetime.now().isoformat()

    def add_pdf(self, pdf_name: str) -> None:
        if pdf_name not in self.pdf_files:
            self.pdf_files.append(pdf_name)
            self.last_updated = datetime.now().isoformat()

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.token_usage["input"] += input_tokens
        self.token_usage["output"] += output_tokens
        self.token_usage["total"] = self.token_usage["input"] + self.token_usage["output"]
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_turns": self.total_turns,
            "current_summary": self.current_summary,
            "pdf_files": self.pdf_files,
            "settings": self.settings,
            "token_usage": self.token_usage,
            "pdf_description": self.pdf_description,
        }

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
        )

    @staticmethod
    def generate_id() -> str:
        return datetime.now().strftime("%Y%m%d%H%M")
