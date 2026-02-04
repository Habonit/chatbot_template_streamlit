from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    chunk_index: int
    original_text: str
    normalized_text: str
    source_file: str
    source_page: int
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": {
                "source_file": self.source_file,
                "source_page": self.source_page,
                "start_char": self.start_char,
                "end_char": self.end_char,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        metadata = data.get("metadata", {})
        embedding = data.get("embedding")
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float32)

        return cls(
            chunk_index=data["chunk_index"],
            original_text=data["original_text"],
            normalized_text=data["normalized_text"],
            source_file=metadata.get("source_file", ""),
            source_page=metadata.get("source_page", 0),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", 0),
            embedding=embedding,
        )
