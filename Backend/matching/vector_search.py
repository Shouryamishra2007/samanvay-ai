from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorCandidate:
    """
    A candidate returned by semantic vector retrieval.

    This object represents retrieval evidence only.
    It does not determine technical equivalence or MatchTier.
    """

    material_id: str
    score: float


def rank_candidates(
    candidates: list[VectorCandidate],
) -> list[VectorCandidate]:
    """
    Return candidates ordered by descending semantic similarity score.

    This function performs ranking only.
    It does not:
    - compare technical attributes
    - infer engineering equivalence
    - assign MatchTier
    - use canonical_id as production inference input
    - use extracted_* fields
    """

    return sorted(
        candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )


def retrieve_candidates(
    query_text: str,
    candidate_texts: dict[str, str],
    embedding_model,
) -> list[VectorCandidate]:
    """
    Retrieve candidate materials ranked by semantic similarity.

    Parameters
    ----------
    query_text:
        Normalized material description used as the query.

    candidate_texts:
        Mapping from production material ID to material description.

    embedding_model:
        EmbeddingModel-compatible object.

    Returns
    -------
    list[VectorCandidate]
        Candidates ordered by descending cosine similarity.

    This function performs semantic retrieval only.
    It does not:
    - compare technical attributes
    - infer engineering equivalence
    - assign MatchTier
    - use canonical_id
    - use extracted_* fields
    """

    if not query_text:
        return []

    if not candidate_texts:
        return []

    material_ids = list(candidate_texts.keys())
    texts = [query_text, *[candidate_texts[mid] for mid in material_ids]]

    embeddings = embedding_model.encode(texts)

    query_embedding = embeddings[0]

    candidates = []

    for material_id, candidate_embedding in zip(
        material_ids,
        embeddings[1:],
    ):
        score = sum(
            query_value * candidate_value
            for query_value, candidate_value in zip(
                query_embedding,
                candidate_embedding,
            )
        )

        candidates.append(
            VectorCandidate(
                material_id=material_id,
                score=score,
            )
        )

    return rank_candidates(candidates)