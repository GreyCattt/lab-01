import math
from typing import Protocol
from findex.index import Index

class Scorer(Protocol):
    """
    Протокол (інтерфейс), що описує, як має виглядати будь-який алгоритм ранжування.
    Duck Typing у дії: якщо клас має метод score(tf, df, dl) - він є Scorer-ом.
    """
    def score(self, tf: int, df: int, doc_length: int) -> float:
        ...

class TfIdfScorer:
    """Класичний алгоритм Term Frequency - Inverse Document Frequency."""
    def __init__(self, index: Index):
        self.num_docs = index.num_docs

    def score(self, tf: int, df: int, doc_length: int) -> float:
        tf_weight = 1 + math.log10(tf) if tf > 0 else 0.0
        
        idf_weight = math.log10(self.num_docs / (df + 1))
        
        return tf_weight * idf_weight

class BM25Scorer:
    """
    Best Matching 25 - сучасний стандарт у пошукових системах (Elasticsearch/Lucene).
    Враховує довжину документа (doc_length) та має коефіцієнти насичення.
    """
    def __init__(self, index: Index, k1: float = 1.5, b: float = 0.75):
        self.num_docs = index.num_docs
        self.avg_dl = index.avg_doc_length
        self.k1 = k1
        self.b = b

    def score(self, tf: int, df: int, doc_length: int) -> float:
        idf = math.log10((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        length_norm = 1.0 - self.b + self.b * (doc_length / self.avg_dl)
        
        tf_weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
        
        return idf * tf_weight