from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingModel(Protocol):
    """
    Interface for a text embedding model.

    Implementations may use BGE-M3 or another embedding model.
    The matching layer depends only on this interface.
    """

    def encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingModel:
    """
    Sentence-Transformers adapter for a text embedding model.

    This adapter is responsible only for converting text into vectors.
    It does not perform:
    - candidate retrieval
    - technical attribute comparison
    - engineering equivalence decisions
    - MatchTier decisions
    - canonical_id lookup
    - extracted_* lookup
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
        )

        return embeddings.tolist()