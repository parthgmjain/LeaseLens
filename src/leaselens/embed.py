from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

# BGE models expect this instruction in front of questions (not documents) at search time.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _load_model(model_name: str) -> TextEmbedding:
    # ONNX runtime instead of PyTorch: about 150 MB of memory instead of 500 MB, so the
    # API fits on small free hosts.
    return TextEmbedding(model_name=model_name)


def embed_texts(
    model_name: str, texts: Sequence[str], prefix: str = ""
) -> list[list[float]]:
    """Embed locally. Normalized vectors make cosine distance a plain dot-product distance."""
    if not texts:
        return []
    model = _load_model(model_name)
    vectors = np.asarray(list(model.embed([prefix + t for t in texts])), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)
    return [vector.tolist() for vector in vectors]
