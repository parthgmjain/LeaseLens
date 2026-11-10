from collections.abc import Sequence

from openai import OpenAI

# The embeddings endpoint accepts up to 2048 inputs per request. Stay well under it.
BATCH_SIZE = 100


def embed_texts(client: OpenAI, model: str, texts: Sequence[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[start : start + BATCH_SIZE])
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(
            item.embedding for item in sorted(response.data, key=lambda d: d.index)
        )
    return vectors
