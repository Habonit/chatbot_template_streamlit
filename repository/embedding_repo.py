import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import faiss

from domain.chunk import Chunk


class EmbeddingRepository:
    def __init__(self, base_path: Path | str = "data/sessions"):
        self.base_path = Path(base_path)
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._chunks: dict[str, list[Chunk]] = {}

    def _get_pkl_path(self, session_id: str) -> Path:
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / "embeddings.pkl"

    def save_chunks(
        self,
        session_id: str,
        chunks: list[Chunk],
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        pkl_path = self._get_pkl_path(session_id)

        embeddings = []
        for chunk in chunks:
            if chunk.embedding is not None:
                normalized = chunk.embedding / np.linalg.norm(chunk.embedding)
                embeddings.append(normalized)

        index = faiss.IndexFlatIP(embedding_dim)
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            index.add(embeddings_array)

        data = {
            "faiss_index": faiss.serialize_index(index),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "config": {
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "created_at": datetime.now().isoformat(),
            },
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        self._indices[session_id] = index
        self._chunks[session_id] = chunks

    def load_chunks(self, session_id: str) -> tuple[list[Chunk], dict]:
        pkl_path = self._get_pkl_path(session_id)
        if not pkl_path.exists():
            return [], {}

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        index = faiss.deserialize_index(data["faiss_index"])
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        config = data.get("config", {})

        self._indices[session_id] = index
        self._chunks[session_id] = chunks

        return chunks, config

    def search_similar(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        if session_id not in self._indices:
            self.load_chunks(session_id)

        if session_id not in self._indices:
            return []

        index = self._indices[session_id]
        chunks = self._chunks[session_id]

        if index.ntotal == 0:
            return []

        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)

        scores, indices = index.search(query_normalized, min(top_k, index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(chunks):
                results.append({
                    "chunk": chunks[idx],
                    "score": float(score),
                })

        return results

    def delete_chunks(self, session_id: str) -> None:
        pkl_path = self._get_pkl_path(session_id)
        if pkl_path.exists():
            pkl_path.unlink()

        if session_id in self._indices:
            del self._indices[session_id]
        if session_id in self._chunks:
            del self._chunks[session_id]
