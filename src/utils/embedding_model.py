"""Single shared sentence-embedding model for the whole engine.

Previously each module (and each character's PerceptualState) loaded its own
copy of all-MiniLM-L6-v2 at import time: ~7 loads at startup plus one per
character. All callers now share this one instance.
"""
from functools import lru_cache

MODEL_NAME = 'all-MiniLM-L6-v2'


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as e:
        print(f"Warning: could not load embedding model from local cache ({e}); downloading")
        return SentenceTransformer(MODEL_NAME)
