"""BM25 keyword index, implemented in pure numpy.

Chinese text is tokenised into character unigrams + bigrams, so no external
tokeniser (jieba, etc.) is needed. Good enough for the document sets this
project targets, and it keeps keyword retrieval fully offline.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .text import tokenize


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.size = len(documents)
        self.doc_len = np.zeros(self.size, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for index, text in enumerate(documents):
            tokens = tokenize(text)
            self.doc_len[index] = len(tokens)
            for token, count in Counter(tokens).items():
                postings[token].append((index, count))
                doc_freq[token] += 1

        self.avg_len = float(self.doc_len.mean()) if self.size else 0.0
        self.idf = {
            token: math.log(1.0 + (self.size - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        self.postings = {
            token: (
                np.fromiter((row[0] for row in rows), dtype=np.int32, count=len(rows)),
                np.fromiter((row[1] for row in rows), dtype=np.float32, count=len(rows)),
            )
            for token, rows in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(self.size, dtype=np.float32)
        if not self.size:
            return result

        for token in set(tokenize(query)):
            posting = self.postings.get(token)
            if posting is None:
                continue
            rows, frequencies = posting
            denominator = frequencies + self.k1 * (
                1.0 - self.b + self.b * self.doc_len[rows] / (self.avg_len or 1.0)
            )
            result[rows] += self.idf[token] * frequencies * (self.k1 + 1.0) / denominator
        return result

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        if not scores.size:
            return []
        top_k = min(top_k, scores.size)
        candidates = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(
            ((int(i), float(scores[i])) for i in candidates if scores[i] > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked
