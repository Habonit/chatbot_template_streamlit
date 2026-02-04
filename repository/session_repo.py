import json
from pathlib import Path
from domain.session import Session


class SessionRepository:
    """세션 메타데이터 저장/로드 담당 Repository"""

    def __init__(self, base_path: Path | str = "data/sessions"):
        self.base_path = Path(base_path)

    def _get_metadata_path(self, session_id: str) -> Path:
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / "metadata.json"

    def save_session(self, session: Session) -> None:
        """세션 메타데이터를 JSON 파일로 저장"""
        metadata_path = self._get_metadata_path(session.session_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Session | None:
        """세션 메타데이터를 JSON 파일에서 로드"""
        metadata_path = self._get_metadata_path(session_id)
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Session.from_dict(data)

    def list_sessions(self) -> list[str]:
        """저장된 모든 세션 ID 목록 반환"""
        if not self.base_path.exists():
            return []

        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    sessions.append(session_dir.name)
        return sorted(sessions)

    def delete_session(self, session_id: str) -> None:
        """세션 메타데이터 삭제"""
        metadata_path = self._get_metadata_path(session_id)
        if metadata_path.exists():
            metadata_path.unlink()

    def exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        metadata_path = self.base_path / session_id / "metadata.json"
        return metadata_path.exists()
