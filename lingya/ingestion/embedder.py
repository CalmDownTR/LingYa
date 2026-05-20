from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._load_error: str | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None and self._load_error is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                self._load_error = (
                    f"Failed to download embedding model '{self.model_name}' from HuggingFace.\n"
                    f"Error: {e}\n\n"
                    "If you are in a region where HuggingFace is blocked, try setting a mirror:\n"
                    "  export HF_ENDPOINT=https://hf-mirror.com\n"
                    "Or download the model manually to ~/.cache/sentence-transformers/"
                )
                raise RuntimeError(self._load_error) from e
        if self._model is None:
            raise RuntimeError(self._load_error)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.encode_sync, texts)

    def encode_sync(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
