"""Embedding generation and management"""
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    model = None


def get_embedding(text: str):
    """
    Returns a dense vector embedding.
    Falls back to a tiny hashing trick if transformers aren't installed.
    """
    if model:
        return model.encode(text).tolist()

    # lightweight fallback embedding (hash-based)
    return [hash(text) % 997 / 997.0]