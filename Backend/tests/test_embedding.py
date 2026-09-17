from matching.embedding import EmbeddingModel


class FakeEmbeddingModel:
    def encode(self, texts):
        return [
            [float(len(text)), 1.0]
            for text in texts
        ]


def test_embedding_model_protocol_is_available():
    assert EmbeddingModel is not None


def test_fake_embedding_model_returns_one_vector_per_text():
    model = FakeEmbeddingModel()

    result = model.encode([
        "GATE VALVE",
        "BALL VALVE",
    ])

    assert len(result) == 2
    assert result[0] == [10.0, 1.0]
    assert result[1] == [10.0, 1.0]


def test_embedding_model_accepts_empty_sequence():
    model = FakeEmbeddingModel()

    assert model.encode([]) == []