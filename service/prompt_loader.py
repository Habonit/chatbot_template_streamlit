from pathlib import Path
from typing import Optional


class PromptLoader:
    """프롬프트 파일을 로드하고 관리하는 클래스"""

    def __init__(self, base_path: Path | str = Path("prompt")):
        self.base_path = Path(base_path)
        self._cache: dict[str, str] = {}

    def load(self, category: str, filename: str) -> str:
        """프롬프트 파일 로드"""
        cache_key = f"{category}/{filename}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self.base_path / category / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        self._cache[cache_key] = content
        return content

    def format(self, template: str, **kwargs) -> str:
        """템플릿 변수 치환"""
        return template.format(**kwargs)

    def get_normalization_prompt(self, chunk_text: str) -> str:
        """PDF 정규화 프롬프트 가져오기"""
        template = self.load("pdf", "normalization.txt")
        return self.format(template, chunk_text=chunk_text)

    def get_description_prompt(self, sample_text: str) -> str:
        """PDF 설명 생성 프롬프트 가져오기"""
        template = self.load("pdf", "description.txt")
        return self.format(template, sample_text=sample_text)