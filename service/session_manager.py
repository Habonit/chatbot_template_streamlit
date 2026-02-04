"""Phase 02-6: SQLite 기반 세션 관리 서비스

SqliteSaver가 저장한 체크포인트에서 세션 정보를 조회/관리합니다.
ConversationRepository, SessionRepository를 대체합니다.
"""
import csv
import json
import sqlite3
from pathlib import Path


class SessionManager:
    """SQLite 기반 세션 관리 서비스"""

    def __init__(self, db_path: str = "data/langgraph.db"):
        self.db_path = Path(db_path)
        self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결 반환 (lazy initialization)"""
        if self._conn is None:
            # 부모 디렉토리가 없으면 생성
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def _ensure_tables_exist(self) -> bool:
        """체크포인트 테이블이 존재하는지 확인"""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        )
        return cursor.fetchone() is not None

    def list_sessions(self) -> list[str]:
        """저장된 모든 세션 ID 목록 반환

        Returns:
            세션 ID 리스트 (thread_id 기준)
        """
        if not self._ensure_tables_exist():
            return []

        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_session_history(self, session_id: str) -> list[dict]:
        """세션의 대화 히스토리 추출

        Args:
            session_id: 세션 ID

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        if not self._ensure_tables_exist():
            return []

        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT checkpoint
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return []

        try:
            checkpoint_data = json.loads(row[0])
            channel_values = checkpoint_data.get("channel_values", {})
            messages = channel_values.get("messages", [])

            history = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Serialized message format
                    msg_type = msg.get("type", "")
                    if msg_type == "human":
                        role = "user"
                    elif msg_type == "ai":
                        role = "assistant"
                    elif msg_type == "system":
                        role = "system"
                    else:
                        role = "unknown"
                    content = msg.get("content", "")
                else:
                    # LangChain BaseMessage object (shouldn't happen in JSON)
                    role = "unknown"
                    content = str(msg)

                history.append({"role": role, "content": content})

            return history

        except (json.JSONDecodeError, KeyError, TypeError):
            return []


    def get_session_metadata(self, session_id: str) -> dict:
        """세션의 메타데이터 반환 (토큰 사용량, 턴 수 등)

        Args:
            session_id: 세션 ID

        Returns:
            메타데이터 딕셔너리
        """
        if not self._ensure_tables_exist():
            return {}

        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT checkpoint
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}

        try:
            checkpoint_data = json.loads(row[0])
            channel_values = checkpoint_data.get("channel_values", {})

            return {
                "turn_count": channel_values.get("turn_count", 0),
                "summary": channel_values.get("summary", ""),
                "summary_history": channel_values.get("summary_history", []),
                "pdf_description": channel_values.get("pdf_description", ""),
                "model_used": channel_values.get("model_used", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    def close(self) -> None:
        """DB 연결 닫기"""
        if self._conn:
            self._conn.close()
            self._conn = None
