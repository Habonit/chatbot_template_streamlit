from pathlib import Path
import pdfplumber


class PDFExtractor:
    def __init__(self, max_size_mb: int = 20):
        self.max_size_mb = max_size_mb

    def validate_size(self, file_size_bytes: int) -> bool:
        max_bytes = self.max_size_mb * 1024 * 1024
        return file_size_bytes <= max_bytes

    def validate_extension(self, filename: str) -> bool:
        return filename.lower().endswith(".pdf")

    def extract_text(self, pdf_path: Path | str) -> tuple[str, int]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not self.validate_extension(pdf_path.name):
            raise ValueError("Invalid file extension. Only .pdf files are supported.")

        if not self.validate_size(pdf_path.stat().st_size):
            raise ValueError(f"File too large. Maximum size is {self.max_size_mb}MB.")

        text_parts = []
        page_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF (may be image-based).")

        return full_text, page_count
