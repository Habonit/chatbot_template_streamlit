"""Phase 04: Message 교육용 메타데이터 필드 테스트

Message dataclass에 추가된 필드:
- mode, graph_path, summary_triggered, thought_process, thinking_budget, is_casual
"""
import pytest
from domain.message import Message


class TestMessageDefaultValues:
    """Message 새 필드 기본값 테스트"""

    def test_message_creation_with_defaults(self):
        """기본값으로 Message 생성"""
        msg = Message(turn_id=1, role="assistant", content="Hello")
        assert msg.turn_id == 1
        assert msg.role == "assistant"
        assert msg.content == "Hello"

    def test_mode_defaults_to_normal(self):
        """mode 기본값은 'normal'"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.mode == "normal"

    def test_graph_path_defaults_to_empty_list(self):
        """graph_path 기본값은 빈 리스트"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.graph_path == []
        assert isinstance(msg.graph_path, list)

    def test_summary_triggered_defaults_to_false(self):
        """summary_triggered 기본값은 False"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.summary_triggered is False

    def test_thought_process_defaults_to_none(self):
        """thought_process 기본값은 None"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.thought_process is None

    def test_thinking_budget_defaults_to_zero(self):
        """thinking_budget 기본값은 0"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.thinking_budget == 0

    def test_is_casual_defaults_to_false(self):
        """is_casual 기본값은 False"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.is_casual is False


class TestMessageExplicitValues:
    """Message 새 필드 명시적 값 테스트"""

    def test_message_with_explicit_mode(self):
        """mode 명시적 설정"""
        msg = Message(turn_id=1, role="assistant", content="test", mode="casual")
        assert msg.mode == "casual"

    def test_message_with_reasoning_mode(self):
        """reasoning mode 설정"""
        msg = Message(turn_id=1, role="assistant", content="test", mode="reasoning")
        assert msg.mode == "reasoning"

    def test_message_with_graph_path(self):
        """graph_path 명시적 설정"""
        path = ["summary_node", "llm_node", "tool_node", "llm_node"]
        msg = Message(turn_id=1, role="assistant", content="test", graph_path=path)
        assert msg.graph_path == ["summary_node", "llm_node", "tool_node", "llm_node"]

    def test_message_with_summary_triggered(self):
        """summary_triggered True 설정"""
        msg = Message(turn_id=1, role="assistant", content="test", summary_triggered=True)
        assert msg.summary_triggered is True

    def test_message_with_thought_process(self):
        """thought_process 설정"""
        msg = Message(
            turn_id=1, role="assistant", content="test",
            thought_process="사용자가 복잡한 질문을 했으므로..."
        )
        assert msg.thought_process == "사용자가 복잡한 질문을 했으므로..."

    def test_message_with_thinking_budget(self):
        """thinking_budget 설정"""
        msg = Message(turn_id=1, role="assistant", content="test", thinking_budget=4096)
        assert msg.thinking_budget == 4096

    def test_message_with_is_casual(self):
        """is_casual True 설정"""
        msg = Message(turn_id=1, role="assistant", content="test", is_casual=True)
        assert msg.is_casual is True


class TestMessageAllFieldsPopulated:
    """모든 필드가 채워진 Message 테스트"""

    def test_all_fields_populated(self):
        """모든 Phase 04 필드를 포함한 Message 생성"""
        msg = Message(
            turn_id=3,
            role="assistant",
            content="검색 결과입니다",
            input_tokens=100,
            output_tokens=50,
            model_used="gemini-2.0-flash",
            function_calls=[{"name": "web_search", "args": {}}],
            tool_results={"web_search": "검색 결과"},
            mode="normal",
            graph_path=["summary_node", "llm_node", "tool_node", "llm_node"],
            summary_triggered=True,
            thought_process="분석 과정...",
            thinking_budget=2048,
            is_casual=False,
        )
        assert msg.turn_id == 3
        assert msg.role == "assistant"
        assert msg.content == "검색 결과입니다"
        assert msg.input_tokens == 100
        assert msg.output_tokens == 50
        assert msg.model_used == "gemini-2.0-flash"
        assert msg.function_calls == [{"name": "web_search", "args": {}}]
        assert msg.tool_results == {"web_search": "검색 결과"}
        assert msg.mode == "normal"
        assert msg.graph_path == ["summary_node", "llm_node", "tool_node", "llm_node"]
        assert msg.summary_triggered is True
        assert msg.thought_process == "분석 과정..."
        assert msg.thinking_budget == 2048
        assert msg.is_casual is False

    def test_graph_path_not_shared_between_instances(self):
        """graph_path가 인스턴스 간 공유되지 않음"""
        msg1 = Message(turn_id=1, role="assistant", content="a")
        msg2 = Message(turn_id=2, role="assistant", content="b")
        msg1.graph_path.append("summary_node")
        assert msg2.graph_path == []

    def test_casual_message(self):
        """casual 모드 메시지 구성"""
        msg = Message(
            turn_id=2,
            role="assistant",
            content="안녕하세요!",
            mode="casual",
            graph_path=["casual_bypass"],
            summary_triggered=False,
            is_casual=True,
        )
        assert msg.mode == "casual"
        assert msg.graph_path == ["casual_bypass"]
        assert msg.summary_triggered is False
        assert msg.is_casual is True
