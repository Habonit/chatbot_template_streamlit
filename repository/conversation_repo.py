import csv
from pathlib import Path
from domain.message import Message


class ConversationRepository:
    def __init__(self, base_path: Path | str = "data/sessions"):
        self.base_path = Path(base_path)

    def _get_csv_path(self, session_id: str) -> Path:
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / "conversation.csv"

    def save_messages(self, session_id: str, messages: list[Message]) -> None:
        csv_path = self._get_csv_path(session_id)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["turn_id", "role", "content", "timestamp", "input_tokens", "output_tokens", "model_used"],
            )
            writer.writeheader()
            for msg in messages:
                writer.writerow(msg.to_dict())

    def load_messages(self, session_id: str) -> list[Message]:
        csv_path = self._get_csv_path(session_id)
        if not csv_path.exists():
            return []

        messages = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["turn_id"] = int(row["turn_id"])
                row["input_tokens"] = int(row["input_tokens"])
                row["output_tokens"] = int(row["output_tokens"])
                row["model_used"] = row["model_used"] if row["model_used"] else None
                messages.append(Message.from_dict(row))
        return messages

    def append_message(self, session_id: str, message: Message) -> None:
        csv_path = self._get_csv_path(session_id)
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["turn_id", "role", "content", "timestamp", "input_tokens", "output_tokens", "model_used"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(message.to_dict())

    def clear_messages(self, session_id: str) -> None:
        csv_path = self._get_csv_path(session_id)
        if csv_path.exists():
            csv_path.unlink()
