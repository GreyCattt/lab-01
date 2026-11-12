from __future__ import annotations

import json
import pickle
from collections import defaultdict
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from findex.models import DocMeta, Posting


def iter_documents(jsonl_path: str) -> Iterator[tuple[str, str]]:
    """Генератор, який читає JSONL файл і повертає документи для індексації."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            yield doc["url"], doc["text"]


def build_partial(paths: list[Path], start_doc_id: int) -> Index:
    """
    Функція рівня модуля для паралельної обробки.
    Кожен воркер отримує свій діапазон унікальних doc_id (start_doc_id).
    Оптимізовано у Лабі 8: однопрохідний збір позицій O(N) замість O(U * N).
    """
    from findex.tokenize import tokenize  # Локальний імпорт для уникнення циклів

    idx = Index()
    for i, path in enumerate(paths):
        doc_id = start_doc_id + i
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens = list(tokenize(text))

        idx._metadata[doc_id] = DocMeta(  # type: ignore[reportPrivateUsage]
            doc_id=doc_id, path=str(path), length=len(tokens)
        )
        pos_by_term: dict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            pos_by_term[term].append(pos)

        for term, positions in pos_by_term.items():
            idx._postings[term].append(  # type: ignore[reportPrivateUsage]
                Posting(doc_id=doc_id, frequency=len(positions), positions=tuple(positions))
            )

    return idx


class Index(Mapping[str, list[Posting]]):
    def __init__(self) -> None:
        self._postings: dict[str, list[Posting]] = defaultdict(list)
        self._metadata: dict[int, DocMeta] = {}
        self._np_postings: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}

    def _invalidate_caches(self) -> None:
        self._np_postings.clear()
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("doc_lengths", None)

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings.get(term, [])

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return term in self._postings

    def __repr__(self) -> str:
        """Зручне відображення об'єкта в консолі/дебагері."""
        return f"<Index: {len(self)} terms, {self.num_docs} docs, avg_len: {self.avg_doc_length:.1f}>"

    def merge(self, other: Index) -> None:
        """Зливає інший індекс у поточний."""
        self._metadata.update(other._metadata)  # type: ignore[reportPrivateUsage]

        for term, postings in other._postings.items():  # type: ignore[reportPrivateUsage]
            self._postings[term].extend(postings)
            self._postings[term].sort(key=lambda p: p.doc_id)

        self._invalidate_caches()

    def build(self, corpus_dir: Path) -> None:
        """Оновлений метод для зворотної сумісності (Лаба 4)."""
        paths = sorted(corpus_dir.glob("**/*.txt"))
        if not paths:
            return

        partial = build_partial(paths, 0)
        self.merge(partial)

    def save(self, path: Path) -> None:
        """Зберігає індекс на диск за допомогою pickle."""
        with path.open("wb") as f:
            pickle.dump((self._postings, self._metadata), f)

    @property
    def num_docs(self) -> int:
        """Кількість документів у корпусі."""
        return len(self._metadata)

    @cached_property
    def doc_lengths(self) -> NDArray[np.int32]:
        """Масив довжин усіх документів у форматі NumPy int32."""
        if not self._metadata:
            return np.zeros(0, dtype=np.int32)
        max_id = max(self._metadata.keys())
        lengths = np.zeros(max_id + 1, dtype=np.int32)
        for doc_id, meta in self._metadata.items():
            lengths[doc_id] = meta.length
        return lengths

    @cached_property
    def avg_doc_length(self) -> float:
        """Середня довжина документа (кешується після першого обчислення)."""
        if not self._metadata:
            return 0.0
        return float(np.sum(self.doc_lengths, dtype=np.int64)) / self.num_docs

    def get_postings_arrays(self, term: str) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """
        Повертає постінги терміна як два масиви NumPy int32: (doc_ids, frequencies).
        Масиви створюються ліниво і кешуються в пам'яті.
        """
        cached = self._np_postings.get(term)
        if cached is not None:
            return cached

        postings = self._postings.get(term)
        if not postings:
            empty = np.zeros(0, dtype=np.int32)
            res = (empty, empty)
            self._np_postings[term] = res
            return res

        n = len(postings)
        doc_ids = np.fromiter((p.doc_id for p in postings), dtype=np.int32, count=n)
        freqs = np.fromiter((p.frequency for p in postings), dtype=np.int32, count=n)
        res = (doc_ids, freqs)
        self._np_postings[term] = res
        return res

    def df(self, term: str) -> int:
        """Document Frequency — у скількох документах зустрічається термін."""
        return len(self[term])

    def get_meta(self, doc_id: int) -> DocMeta:
        return self._metadata[doc_id]


@contextmanager
def open_index(path: Path) -> Generator[Index, None, None]:
    f = path.open("rb")
    try:
        postings, metadata = pickle.load(f)
        idx = Index()
        idx._postings = postings  # type: ignore[reportPrivateUsage]
        idx._metadata = metadata  # type: ignore[reportPrivateUsage]
        yield idx
    finally:
        f.close()
