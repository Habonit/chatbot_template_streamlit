"""Phase 05 Step 1: Message actual_prompts 필드 테스트

Message dataclass에 추가된 actual_prompts 필드:
- 기본값: 빈 dict
- dict에 값 설정 가능
"""
import pytest
from domain.message import Message


class TestActualPromptsDefaults:
    """actual_prompts 필드 기본값 테스트"""

    def test_actual_prompts_defaults_to_empty_dict(self):
        """actual_prompts 기본값은 빈 dict"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert msg.actual_prompts == {}

    def test_actual_prompts_is_dict_type(self):
        """actual_prompts는 dict 타입"""
        msg = Message(turn_id=1, role="assistant", content="test")
        assert isinstance(msg.actual_prompts, dict)

    def test_actual_prompts_independent_per_instance(self):
        """각 Message 인스턴스의 actual_prompts는 독립적"""
        msg1 = Message(turn_id=1, role="assistant", content="a")
        msg2 = Message(turn_id=2, role="assistant", content="b")
        msg1.actual_prompts["key"] = "val"
        assert msg2.actual_prompts == {}


class TestActualPromptsValues:
    """actual_prompts 필드 값 설정 테스트"""

    def test_actual_prompts_set_on_creation(self):
        """생성 시 actual_prompts 설정"""
        prompts = {
            "system_prompt": "You are helpful.",
            "user_messages_count": 2,
            "context_turns": 3,
        }
        msg = Message(
            turn_id=1, role="assistant", content="test",
            actual_prompts=prompts,
        )
        assert msg.actual_prompts["system_prompt"] == "You are helpful."
        assert msg.actual_prompts["user_messages_count"] == 2
        assert msg.actual_prompts["context_turns"] == 3

    def test_actual_prompts_with_casual_prompt(self):
        """casual 모드의 casual_prompt 포함"""
        prompts = {
            "system_prompt": "",
            "user_messages_count": 1,
            "context_turns": 0,
            "casual_prompt": '사용자가 "안녕"라고 말했습니다.',
        }
        msg = Message(
            turn_id=1, role="assistant", content="test",
            actual_prompts=prompts,
        )
        assert "casual_prompt" in msg.actual_prompts
        assert "안녕" in msg.actual_prompts["casual_prompt"]
