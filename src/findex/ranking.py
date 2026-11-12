from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from findex.index import Index


class Scorer(Protocol):
    """
    Протокол (інтерфейс), що описує алгоритм ранжування:
    підтримує як скалярний розрахунок (Лаба 3), так і векторизований NumPy (Лаба 8).
    """

    def score(self, tf: int, df: int, doc_length: int) -> float: ...

    def score_array(
        self,
        tf: NDArray[np.int32],
        df: int,
        doc_lengths: NDArray[np.int32],
    ) -> NDArray[np.float64]: ...


class TfIdfScorer:
    """Класичний алгоритм Term Frequency - Inverse Document Frequency."""

    def __init__(self, index: Index) -> None:
        self.num_docs = index.num_docs

    def score(self, tf: int, df: int, doc_length: int) -> float:
        del doc_length
        tf_weight = 1.0 + math.log10(tf) if tf > 0 else 0.0
        idf_weight = math.log10(self.num_docs / (df + 1)) if self.num_docs > 0 else 0.0
        return tf_weight * idf_weight

    def score_array(
        self,
        tf: NDArray[np.int32],
        df: int,
        doc_lengths: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        del doc_lengths
        if tf.size == 0 or self.num_docs == 0:
            return np.zeros(tf.shape, dtype=np.float64)
        tf_f = tf.astype(np.float64)
        tf_weight = np.where(tf > 0, 1.0 + np.log10(np.maximum(tf_f, 1.0)), 0.0)
        idf_weight = math.log10(self.num_docs / (df + 1))
        return tf_weight * idf_weight


class BM25Scorer:
    """
    Best Matching 25 — стандарт у пошукових системах (Elasticsearch/Lucene).
    Враховує довжину документа (doc_length) та має коефіцієнти насичення.
    """

    def __init__(self, index: Index, k1: float = 1.5, b: float = 0.75) -> None:
        self.num_docs = index.num_docs
        self.avg_dl = index.avg_doc_length
        self.k1 = k1
        self.b = b

    def score(self, tf: int, df: int, doc_length: int) -> float:
        if self.num_docs == 0 or self.avg_dl == 0.0:
            return 0.0
        idf = math.log10((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        length_norm = 1.0 - self.b + self.b * (doc_length / self.avg_dl)
        tf_weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        return idf * tf_weight

    def score_array(
        self,
        tf: NDArray[np.int32],
        df: int,
        doc_lengths: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        if tf.size == 0 or self.num_docs == 0 or self.avg_dl == 0.0:
            return np.zeros(tf.shape, dtype=np.float64)
        idf = math.log10((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        dl = doc_lengths.astype(np.float64)
        tf_f = tf.astype(np.float64)
        length_norm = 1.0 - self.b + self.b * (dl / self.avg_dl)
        tf_weight = (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * length_norm)
        return idf * tf_weight
