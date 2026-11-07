"""Two job<->profile matchers scored by the eval harness.

1. KeywordMatcher  - TF-IDF word n-gram cosine similarity (baseline).
2. SemanticMatcher - dense embeddings via fastembed (BAAI/bge-small-en-v1.5,
   ONNX, runs locally). If fastembed or its model is unavailable it degrades
   to SemanticFallbackMatcher (TF-IDF char n-grams + word overlap) and says
   so in its `name`, so reported results never claim embeddings that did
   not run.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class KeywordMatcher:
    name = "keyword-tfidf"

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)

    def fit(self, corpus: list[str]) -> None:
        self._matrix = self._vec.fit_transform(corpus)

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = self._vec.transform([query])
        d = self._vec.transform(docs)
        return cosine_similarity(q, d)[0].tolist()


class SemanticFallbackMatcher:
    """Char n-gram TF-IDF + Jaccard word overlap. NOT embeddings."""

    name = "semantic-fallback (char-ngram tf-idf + word overlap, no embeddings)"

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    def fit(self, corpus: list[str]) -> None:
        self._vec.fit(corpus)

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = self._vec.transform([query])
        d = self._vec.transform(docs)
        char_cos = cosine_similarity(q, d)[0]
        q_tokens = _tokens(query)
        scores = []
        for i, doc in enumerate(docs):
            d_tokens = _tokens(doc)
            union = q_tokens | d_tokens
            jaccard = len(q_tokens & d_tokens) / len(union) if union else 0.0
            scores.append(0.7 * float(char_cos[i]) + 0.3 * jaccard)
        return scores


class SemanticMatcher:
    """Dense-embedding matcher; falls back honestly if unavailable."""

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.MODEL)
            self._fallback: SemanticFallbackMatcher | None = None
            self.name = f"semantic-embeddings ({self.MODEL} via fastembed)"
        except Exception:
            self._model = None
            self._fallback = SemanticFallbackMatcher()
            self.name = self._fallback.name

    def fit(self, corpus: list[str]) -> None:
        if self._fallback is not None:
            self._fallback.fit(corpus)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self._fallback is not None:
            return self._fallback.score(query, docs)
        texts = [query, *docs]
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return (vecs[1:] @ vecs[0]).tolist()
