from matching.vector_search import (
    VectorCandidate,
    rank_candidates,
)


def test_rank_candidates_orders_by_descending_score():
    candidates = [
        VectorCandidate(material_id="MAT-003", score=0.72),
        VectorCandidate(material_id="MAT-001", score=0.94),
        VectorCandidate(material_id="MAT-002", score=0.81),
    ]

    result = rank_candidates(candidates)

    assert result == [
        VectorCandidate(material_id="MAT-001", score=0.94),
        VectorCandidate(material_id="MAT-002", score=0.81),
        VectorCandidate(material_id="MAT-003", score=0.72),
    ]


def test_rank_candidates_does_not_modify_input():
    candidates = [
        VectorCandidate(material_id="MAT-002", score=0.81),
        VectorCandidate(material_id="MAT-001", score=0.94),
    ]

    original = candidates.copy()

    rank_candidates(candidates)

    assert candidates == original


def test_rank_candidates_handles_empty_list():
    assert rank_candidates([]) == []


def test_vector_candidate_stores_retrieval_evidence():
    candidate = VectorCandidate(
        material_id="MAT-001",
        score=0.94,
    )

    assert candidate.material_id == "MAT-001"
    assert candidate.score == 0.94

def test_retrieve_candidates_ranks_by_semantic_similarity():
    from matching.vector_search import retrieve_candidates

    class FakeEmbeddingModel:
        def encode(self, texts):
            vectors = {
                "QUERY": [1.0, 0.0],
                "A": [0.8, 0.6],
                "B": [0.6, 0.8],
                "C": [1.0, 0.0],
            }
            return [vectors[text] for text in texts]

    candidates = retrieve_candidates(
        "QUERY",
        {
            "MAT-A": "A",
            "MAT-B": "B",
            "MAT-C": "C",
        },
        FakeEmbeddingModel(),
    )

    assert [candidate.material_id for candidate in candidates] == [
        "MAT-C",
        "MAT-A",
        "MAT-B",
    ]

    assert candidates[0].score == 1.0

def test_retrieve_candidates_empty_query():
    from matching.vector_search import retrieve_candidates

    class FakeEmbeddingModel:
        def encode(self, texts):
            raise AssertionError("Embedding model should not be called")

    assert retrieve_candidates(
        "",
        {"MAT-001": "GATE VALVE"},
        FakeEmbeddingModel(),
    ) == []


def test_retrieve_candidates_empty_candidates():
    from matching.vector_search import retrieve_candidates

    class FakeEmbeddingModel:
        def encode(self, texts):
            raise AssertionError("Embedding model should not be called")

    assert retrieve_candidates(
        "GATE VALVE",
        {},
        FakeEmbeddingModel(),
    ) == []