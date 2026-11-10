from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import SentenceTransformer

# BGE models expect this instruction in front of questions (not documents) at search time.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(
    model_name: str, texts: Sequence[str], prefix: str = ""
) -> list[list[float]]:
    """Embed locally. Normalized vectors make cosine distance a plain dot-product distance."""
    if not texts:
        return []
    model = _load_model(model_name)
    vectors = model.encode([prefix + t for t in texts], normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
