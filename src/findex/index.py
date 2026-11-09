import pickle
from collections import Counter, defaultdict
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from findex.models import DocMeta, Posting


def build_partial(paths: list[Path], start_doc_id: int) -> 'Index':
    """
    Функція рівня модуля для паралельної обробки.
    Кожен воркер отримує свій діапазон унікальних doc_id (start_doc_id).
    """
    from findex.tokenize import tokenize  # Локальний імпорт для уникнення циклів
    
    idx = Index()
    for i, path in enumerate(paths):
        doc_id = start_doc_id + i
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens = list(tokenize(text))
        
        # type: ignore - прямий доступ до внутрішнього стану для максимальної швидкості воркера
        idx._metadata[doc_id] = DocMeta(doc_id=doc_id, path=str(path), length=len(tokens)) # type: ignore
        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            positions = tuple(pos for pos, t in enumerate(tokens) if t == term)
            # type: ignore - прямий доступ до внутрішнього стану для максимальної швидкості воркера
            idx._postings[term].append( # type: ignore
                Posting(doc_id=doc_id, frequency=freq, positions=positions)
            )
            
    return idx

class Index(Mapping[str, list[Posting]]):
    def __init__(self):
        self._postings: dict[str, list[Posting]] = defaultdict(list)
        self._metadata: dict[int, DocMeta] = {}

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings.get(term, [])

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    # 2. Pyright каже, що __contains__ у Mapping має приймати object, а не str
    def __contains__(self, term: object) -> bool:
        if not isinstance(term, str):
            return False
        return term in self._postings

    def __repr__(self) -> str:
        """Зручне відображення об'єкта в консолі/дебагері."""
        return f"<Index: {len(self)} terms, {self.num_docs} docs, avg_len: {self.avg_doc_length:.1f}>"

    def merge(self, other: "Index") -> None: 
        """Зливає інший індекс у поточний."""
        # type: ignore - доступ до protected полів іншого екземпляра Index для злиття
        self._metadata.update(other._metadata) # type: ignore

        # type: ignore - доступ до protected полів іншого екземпляра Index для злиття
        for term, postings in other._postings.items():# type: ignore
            self._postings[term].extend(postings)
            # Оскільки воркери можуть завершуватися не по порядку, 
            # гарантуємо сортування постінгів для бінарного пошуку/перетину
            self._postings[term].sort(key=lambda p: p.doc_id)

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
    def avg_doc_length(self) -> float:
        """
        Середня довжина документа. Обчислюється лише один раз (ледаче обчислення),
        а потім значення зберігається в кеші (__dict__), щоб не ганяти цикл щоразу.
        """
        if not self._metadata:
            return 0.0
        total_len = sum(meta.length for meta in self._metadata.values())
        return total_len / self.num_docs

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
        idx._postings = postings # type: ignore
        idx._metadata = metadata # type: ignore
        yield idx
    finally:
        f.close()


