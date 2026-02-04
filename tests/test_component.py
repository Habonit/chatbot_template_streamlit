import pytest
from domain.message import Message


class TestSidebarHelpers:
    def test_generate_csv_data(self):
        """CSV 생성 헬퍼 함수 테스트"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content="Hello"),
            Message(turn_id=1, role="assistant", content="Hi there!", model_used="gemini-2.5-flash", input_tokens=10, output_tokens=5),
            Message(turn_id=2, role="user", content="How are you?"),
            Message(turn_id=2, role="assistant", content="I'm doing well!", model_used="gemini-2.5-flash", input_tokens=15, output_tokens=8),
        ]

        csv_data = _generate_csv_data(messages)

        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_data
        assert "1,user,Hello" in csv_data
        assert "1,assistant,Hi there!,gemini-2.5-flash,10,5" in csv_data
        assert "2,user,How are you?" in csv_data

    def test_generate_csv_data_empty(self):
        """빈 메시지 리스트 CSV 생성 테스트"""
        from component.sidebar import _generate_csv_data

        messages = []
        csv_data = _generate_csv_data(messages)

        # 헤더만 있어야 함
        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_data
        lines = csv_data.strip().split("\n")
        assert len(lines) == 1  # 헤더만

    def test_generate_csv_data_special_characters(self):
        """특수 문자 포함 메시지 CSV 생성 테스트"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content='Hello, "world"'),
            Message(turn_id=1, role="assistant", content="Line1\nLine2"),
        ]

        csv_data = _generate_csv_data(messages)

        # CSV가 생성되어야 함 (csv 모듈이 특수 문자 처리)
        assert len(csv_data) > 0
