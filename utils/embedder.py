"""
Embedder factory.

Returns a sentence-transformers model that can be used by both
the VectorIndex and GraphBuilder.  Caches the model so it's only
loaded once per process.
"""

from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)

_CACHE: dict[str, object] = {}


def get_embedder(model_name: str = config.EMBEDDING_MODEL):
    """
    Return a cached SentenceTransformer instance.

    Parameters
    ----------
    model_name: HuggingFace model ID or local path.

    Returns
    -------
    SentenceTransformer with .encode(texts, ...) -> np.ndarray
    """
    if model_name in _CACHE:
        return _CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    _CACHE[model_name] = model
    logger.info("Embedding model loaded.")
    return model
