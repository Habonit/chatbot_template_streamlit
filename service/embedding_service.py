from google import genai
from google.genai import types
import numpy as np


class EmbeddingService:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dim: int = 768,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dim = output_dim

    def create_embedding(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=self.output_dim),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def create_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self.output_dim),
        )
        return [np.array(e.values, dtype=np.float32) for e in response.embeddings]
