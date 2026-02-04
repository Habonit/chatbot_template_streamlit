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

        # bytes 타입으로 반환되어야 함 (UTF-8 BOM 포함)
        assert isinstance(csv_data, bytes)

        # UTF-8 BOM 확인
        assert csv_data.startswith(b'\xef\xbb\xbf')

        # 내용 확인 (decode 후)
        csv_str = csv_data.decode("utf-8-sig")
        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_str
        assert "1,user,Hello" in csv_str
        assert "1,assistant,Hi there!,gemini-2.5-flash,10,5" in csv_str
        assert "2,user,How are you?" in csv_str

    def test_generate_csv_data_empty(self):
        """빈 메시지 리스트 CSV 생성 테스트"""
        from component.sidebar import _generate_csv_data

        messages = []
        csv_data = _generate_csv_data(messages)

        # bytes 타입 확인
        assert isinstance(csv_data, bytes)

        # 헤더만 있어야 함
        csv_str = csv_data.decode("utf-8-sig")
        assert "turn_id,role,content,model_used,input_tokens,output_tokens" in csv_str
        lines = csv_str.strip().split("\n")
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
        assert isinstance(csv_data, bytes)
        assert len(csv_data) > 0

    def test_generate_csv_data_korean(self):
        """한글 메시지 CSV 생성 테스트 (Excel 호환)"""
        from component.sidebar import _generate_csv_data

        messages = [
            Message(turn_id=1, role="user", content="안녕하세요"),
            Message(turn_id=1, role="assistant", content="반갑습니다!"),
        ]

        csv_data = _generate_csv_data(messages)

        # UTF-8 BOM 포함 확인 (Excel 한글 호환)
        assert csv_data.startswith(b'\xef\xbb\xbf')

        # 한글 정상 인코딩 확인
        csv_str = csv_data.decode("utf-8-sig")
        assert "안녕하세요" in csv_str
        assert "반갑습니다!" in csv_str


class TestPdfTabHelpers:
    """PDF 탭 헬퍼 함수 테스트 (항목 7)"""

    def test_format_time_seconds(self):
        """초 단위 시간 포맷 테스트"""
        from component.pdf_tab import _format_time

        assert _format_time(30) == "약 30초"
        assert _format_time(45.7) == "약 45초"
        assert _format_time(0) == "약 0초"

    def test_format_time_minutes(self):
        """분 단위 시간 포맷 테스트"""
        from component.pdf_tab import _format_time

        assert _format_time(60) == "약 1분 0초"
        assert _format_time(90) == "약 1분 30초"
        assert _format_time(150) == "약 2분 30초"
        assert _format_time(125.5) == "약 2분 5초"


class TestPromptsTab:
    """프롬프트 탭 테스트 (항목 8)"""

    def test_get_prompt_info_returns_all_prompts(self):
        """get_prompt_info()가 모든 프롬프트를 반환하는지 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        # 필수 프롬프트 키 목록
        required_keys = [
            "system_prompt",
            "pdf_extension",
            "chain_of_thought",  # 항목 2에서 추가
            "tavily_instruction",  # 항목 6에서 추가
            "summary_prompt",
            "normalization_prompt",
            "description_prompt",
        ]

        for key in required_keys:
            assert key in prompt_info, f"Missing prompt: {key}"

    def test_get_prompt_info_has_required_fields(self):
        """각 프롬프트 정보에 필수 필드가 있는지 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        required_fields = ["title", "description", "usage", "content"]

        for key, info in prompt_info.items():
            for field in required_fields:
                assert field in info, f"Missing field '{field}' in prompt '{key}'"

    def test_chain_of_thought_prompt_content(self):
        """chain_of_thought 프롬프트 내용 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert "chain_of_thought" in prompt_info
        cot = prompt_info["chain_of_thought"]
        assert "추론" in cot["content"] or "reasoning" in cot["content"].lower()
        assert "단계" in cot["content"] or "step" in cot["content"].lower()

    def test_tavily_instruction_prompt_content(self):
        """tavily_instruction 프롬프트 내용 테스트"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()

        assert "tavily_instruction" in prompt_info
        tavily = prompt_info["tavily_instruction"]
        assert "검색" in tavily["content"] or "search" in tavily["content"].lower()

    def test_prompt_count(self):
        """프롬프트 개수 테스트 (7개)"""
        from component.prompts_tab import get_prompt_info

        prompt_info = get_prompt_info()
        assert len(prompt_info) == 7
