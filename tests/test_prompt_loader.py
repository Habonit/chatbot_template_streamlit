import pytest
from pathlib import Path
import tempfile
import os


class TestPromptLoader:
    def test_load_prompt_file(self, tmp_path):
        """프롬프트 파일 로드 테스트"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "system"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "base.txt"
        prompt_file.write_text("This is a test prompt.", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        content = loader.load("system", "base.txt")

        assert content == "This is a test prompt."

    def test_load_prompt_file_not_found(self, tmp_path):
        """존재하지 않는 프롬프트 파일 로드 시 에러"""
        from service.prompt_loader import PromptLoader

        loader = PromptLoader(base_path=tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load("system", "nonexistent.txt")

    def test_format_template(self):
        """템플릿 변수 치환 테스트"""
        from service.prompt_loader import PromptLoader

        loader = PromptLoader()
        template = "Hello, {name}! Your score is {score}."
        result = loader.format(template, name="Alice", score=100)

        assert result == "Hello, Alice! Your score is 100."

    def test_cache_prompt(self, tmp_path):
        """프롬프트 캐싱 테스트"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "system"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "base.txt"
        prompt_file.write_text("Cached content", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)

        # 첫 번째 로드
        content1 = loader.load("system", "base.txt")

        # 파일 내용 변경
        prompt_file.write_text("Updated content", encoding="utf-8")

        # 두 번째 로드 (캐시에서 반환되어야 함)
        content2 = loader.load("system", "base.txt")

        assert content1 == content2 == "Cached content"

    def test_get_system_prompt_without_pdf(self, tmp_path):
        """PDF 없이 시스템 프롬프트 가져오기"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "system"
        prompt_dir.mkdir()
        base_file = prompt_dir / "base.txt"
        base_file.write_text("Base system prompt.", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        prompt = loader.get_system_prompt()

        assert prompt == "Base system prompt."

    def test_get_system_prompt_with_pdf(self, tmp_path):
        """PDF와 함께 시스템 프롬프트 가져오기"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "system"
        prompt_dir.mkdir()
        base_file = prompt_dir / "base.txt"
        base_file.write_text("Base system prompt.", encoding="utf-8")

        ext_file = prompt_dir / "pdf_extension.txt"
        ext_file.write_text("[PDF Info]\n{pdf_description}", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        prompt = loader.get_system_prompt(pdf_description="Test PDF document")

        assert "Base system prompt." in prompt
        assert "[PDF Info]" in prompt
        assert "Test PDF document" in prompt

    def test_get_summary_prompt(self, tmp_path):
        """요약 프롬프트 가져오기"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "summary"
        prompt_dir.mkdir()
        summary_file = prompt_dir / "summary.txt"
        summary_file.write_text("Previous: {previous_summary}\nConvo: {conversation}", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        prompt = loader.get_summary_prompt(previous_summary="Old summary", conversation="New chat")

        assert "Old summary" in prompt
        assert "New chat" in prompt

    def test_get_normalization_prompt(self, tmp_path):
        """정규화 프롬프트 가져오기"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "pdf"
        prompt_dir.mkdir()
        norm_file = prompt_dir / "normalization.txt"
        norm_file.write_text("Normalize: {chunk_text}", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        prompt = loader.get_normalization_prompt(chunk_text="Sample text")

        assert "Normalize: Sample text" in prompt

    def test_get_description_prompt(self, tmp_path):
        """설명 생성 프롬프트 가져오기"""
        from service.prompt_loader import PromptLoader

        # 테스트 프롬프트 파일 생성
        prompt_dir = tmp_path / "pdf"
        prompt_dir.mkdir()
        desc_file = prompt_dir / "description.txt"
        desc_file.write_text("Describe: {sample_text}", encoding="utf-8")

        loader = PromptLoader(base_path=tmp_path)
        prompt = loader.get_description_prompt(sample_text="Document content")

        assert "Describe: Document content" in prompt
