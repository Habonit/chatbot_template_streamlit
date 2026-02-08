"""Step 2-2: auto_reasoning 설정 전달 및 동작 테스트"""
import inspect
from unittest.mock import patch, MagicMock


class TestAutoReasoningFlag:
    """auto_reasoning 플래그가 모드 감지에 영향을 미치는지 테스트"""

    def test_auto_reasoning_true_returns_reasoning_mode(self):
        """auto_reasoning=True일 때 reasoning 입력 → mode=reasoning"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            auto_reasoning=True,
            db_path=":memory:",
        )

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="reasoning"):
            result = builder._prepare_invocation(
                user_input="A와 B를 비교 분석해줘",
                session_id="test",
                turn_count=1,
            )

        assert result is not None
        mode, _, _, _ = result
        assert mode == "reasoning"

    def test_auto_reasoning_false_demotes_reasoning_to_normal(self):
        """auto_reasoning=False일 때 reasoning 입력 → mode=normal"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            auto_reasoning=False,
            db_path=":memory:",
        )

        with patch("service.reasoning_detector.detect_reasoning_need", return_value="reasoning"):
            result = builder._prepare_invocation(
                user_input="A와 B를 비교 분석해줘",
                session_id="test",
                turn_count=1,
            )

        assert result is not None
        mode, _, _, _ = result
        assert mode == "normal"

    def test_casual_unaffected_by_auto_reasoning(self):
        """casual 입력은 auto_reasoning 설정과 무관하게 casual (None 반환)"""
        from service.react_graph import ReactGraphBuilder

        for auto_val in [True, False]:
            builder = ReactGraphBuilder(
                api_key="test-key",
                auto_reasoning=auto_val,
                db_path=":memory:",
            )

            with patch("service.reasoning_detector.detect_reasoning_need", return_value="casual"):
                result = builder._prepare_invocation(
                    user_input="안녕",
                    session_id="test",
                    turn_count=1,
                )

            assert result is None, f"auto_reasoning={auto_val}에서 casual이 None이어야 함"

    def test_normal_unaffected_by_auto_reasoning(self):
        """normal 입력은 auto_reasoning 설정과 무관하게 normal"""
        from service.react_graph import ReactGraphBuilder

        for auto_val in [True, False]:
            builder = ReactGraphBuilder(
                api_key="test-key",
                auto_reasoning=auto_val,
                db_path=":memory:",
            )

            with patch("service.reasoning_detector.detect_reasoning_need", return_value="normal"):
                result = builder._prepare_invocation(
                    user_input="오늘 날씨 알려줘",
                    session_id="test",
                    turn_count=1,
                )

            assert result is not None
            mode, _, _, _ = result
            assert mode == "normal"


class TestAutoReasoningBuilderConfig:
    """ReactGraphBuilder의 auto_reasoning 설정 저장 테스트"""

    def test_builder_stores_auto_reasoning(self):
        """auto_reasoning 값이 저장되는지"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            auto_reasoning=False,
            db_path=":memory:",
        )
        assert builder.auto_reasoning is False

    def test_builder_default_auto_reasoning_true(self):
        """기본값은 True"""
        from service.react_graph import ReactGraphBuilder

        builder = ReactGraphBuilder(
            api_key="test-key",
            db_path=":memory:",
        )
        assert builder.auto_reasoning is True


class TestAppAutoReasoningPassing:
    """app.py가 auto_reasoning 설정을 전달하는지 테스트"""

    def test_app_passes_auto_reasoning(self):
        """app.py 소스코드에 auto_reasoning 전달이 있어야 함"""
        import app

        source = inspect.getsource(app._create_graph_builder)
        assert "auto_reasoning=" in source
