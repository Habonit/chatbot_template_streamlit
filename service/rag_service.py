from domain.chunk import Chunk


class RAGService:
    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1024,
        overlap: int = 256,
        source_file: str = "",
        source_page: int = 1,
    ) -> list[Chunk]:
        chunks = []
        text_chunks = self._recursive_split(text, chunk_size, self.SEPARATORS)

        start_char = 0
        for i, chunk_text in enumerate(text_chunks):
            end_char = start_char + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_index=i,
                    original_text=chunk_text,
                    normalized_text="",
                    source_file=source_file,
                    source_page=source_page,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            start_char = end_char - overlap if overlap < len(chunk_text) else end_char

        return chunks

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        separators: list[str],
    ) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._split_by_size(text, chunk_size)

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(part) > chunk_size:
                    sub_chunks = self._recursive_split(part, chunk_size, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_size(self, text: str, chunk_size: int) -> list[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
