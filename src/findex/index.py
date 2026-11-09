import argparse
import pickle
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
from contextlib import contextmanager

from findex.tokenize import tokenize
from findex.models import Posting, DocMeta
from findex.utils import timed

class Index(Mapping):
    def __init__(self):
        self._postings: dict[str, list[Posting]] = defaultdict(list)
        self._metadata: dict[int, DocMeta] = {}

    def __getitem__(self, term: str) -> list[Posting]:
        """Дозволяє отримувати постінги як index[term]. Якщо слова немає — пустий список."""
        return self._postings.get(term, [])

    def __iter__(self):
        """Дозволяє ітеруватися по термінах: for term in index."""
        return iter(self._postings)

    def __len__(self) -> int:
        """Повертає розмір словника (кількість унікальних термінів)."""
        return len(self._postings)

    def __contains__(self, term: str) -> bool:
        """Дозволяє використовувати оператор in: if 'cat' in index."""
        return term in self._postings

    def __repr__(self) -> str:
        """Зручне відображення об'єкта в консолі/дебагері."""
        return f"<Index: {len(self)} terms, {self.num_docs} docs, avg_len: {self.avg_doc_length:.1f}>"

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

    @timed
    def build(self, data_dir: Path):
        """Будує індекс, збираючи частоти та позиції токенів."""
        doc_id = 0
        for file_path in data_dir.rglob("*.txt"):
            if not file_path.is_file():
                continue
            
            text = file_path.read_text(encoding="utf-8")
            tokens = list(tokenize(text))
            self._metadata[doc_id] = DocMeta(doc_id, len(tokens), str(file_path))
            
            # Збираємо позиції кожного токена
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
                
            # Створюємо постінги з кортежами позицій
            for term, positions in term_positions.items():
                self._postings[term].append(
                    Posting(doc_id, len(positions), tuple(positions))
                )
                
            doc_id += 1

    @timed
    def save(self, out_path: Path):
        """Зберігає індекс на диск."""
        with out_path.open("wb") as f:
            pickle.dump((dict(self._postings), self._metadata), f)


@contextmanager
def open_index(path: Path):
    """
    Контекстний менеджер для безпечного завантаження індексу.
    Використання: with open_index("index.pkl") as index: ...
    Гарантує, що файл буде закрито навіть якщо під час завантаження виникне помилка.
    """
    f = path.open("rb")
    try:
        postings, metadata = pickle.load(f)
        idx = Index()
        idx._postings = postings
        idx._metadata = metadata
        yield idx
    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(description="Build and Save Index")
    parser.add_argument("data_dir", type=Path, help="Corpus directory")
    parser.add_argument("--out", type=Path, required=True, help="Output index file")
    args = parser.parse_args()

    idx = Index()
    idx.build(args.data_dir)
    idx.save(args.out)

if __name__ == "__main__":
    main()