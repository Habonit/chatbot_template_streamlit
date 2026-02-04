"""Phase 02-6: SessionManager 테스트 (TDD)"""
import pytest
import tempfile
import os
from pathlib import Path


class TestSessionManagerImport:
    """SessionManager 임포트 테스트"""

    def test_import_session_manager(self):
        """SessionManager 클래스 임포트 테스트"""
        from service.session_manager import SessionManager

        assert SessionManager is not None

    def test_create_session_manager(self):
        """SessionManager 인스턴스 생성 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            assert manager is not None
            assert manager.db_path == Path(db_path)


class TestSessionManagerListSessions:
    """list_sessions() 메서드 테스트"""

    def test_list_sessions_empty(self):
        """빈 DB에서 세션 목록 조회"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            sessions = manager.list_sessions()

            assert sessions == []

    def test_list_sessions_returns_list(self):
        """list_sessions()가 리스트를 반환하는지 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            sessions = manager.list_sessions()

            assert isinstance(sessions, list)

    def test_list_sessions_after_graph_build(self):
        """그래프 빌드 후 테이블이 생성되는지 테스트"""
        from service.session_manager import SessionManager
        from service.react_graph import ReactGraphBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # ReactGraphBuilder로 그래프 빌드 (invoke 없이)
            builder = ReactGraphBuilder(
                api_key="test-api-key",
                db_path=db_path,
            )
            builder.build()

            # SessionManager로 세션 목록 조회
            manager = SessionManager(db_path=db_path)
            sessions = manager.list_sessions()

            # 테이블이 생성되어 빈 리스트 반환 (invoke 전이므로 세션 없음)
            assert isinstance(sessions, list)


class TestSessionManagerGetSessionHistory:
    """get_session_history() 메서드 테스트"""

    def test_get_session_history_returns_list(self):
        """get_session_history()가 리스트를 반환하는지 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            history = manager.get_session_history("nonexistent-session")

            assert isinstance(history, list)

    def test_get_session_history_empty_for_nonexistent(self):
        """존재하지 않는 세션에 대해 빈 리스트 반환"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            history = manager.get_session_history("nonexistent-session")

            assert history == []

    def test_get_session_history_method_exists(self):
        """get_session_history 메서드가 있는지 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            assert hasattr(manager, "get_session_history")
            assert callable(manager.get_session_history)


class TestSessionManagerConnection:
    """DB 연결 관리 테스트"""

    def test_manager_creates_db_file(self):
        """SessionManager가 DB 파일을 생성하는지 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SessionManager(db_path=db_path)

            # list_sessions() 호출하면 연결이 생성됨
            manager.list_sessions()

            assert os.path.exists(db_path)

    def test_manager_handles_nonexistent_db(self):
        """존재하지 않는 DB 경로를 처리하는지 테스트"""
        from service.session_manager import SessionManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "subdir", "test.db")
            manager = SessionManager(db_path=db_path)

            # 부모 디렉토리가 없어도 처리해야 함
            sessions = manager.list_sessions()

            assert isinstance(sessions, list)
