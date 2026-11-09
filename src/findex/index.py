import time
import pickle
import json
import tracemalloc
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from findex.tokenize import tokenize
from findex.models import Posting, DocMeta

class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, list[Posting]] = defaultdict(list)
        self.metadata: dict[int, DocMeta] = {}

    def build(self, data_dir: Path):
        """Будує індекс за один прохід по корпусу."""
        doc_id = 0
        for file_path in data_dir.rglob("*.txt"):
            if not file_path.is_file():
                continue
            
            text = file_path.read_text(encoding="utf-8")
            tokens = list(tokenize(text))
            
            # Зберігаємо метадані документа
            self.metadata[doc_id] = DocMeta(doc_id, len(tokens), str(file_path))
            
            # Збираємо частоти слів у поточному документі
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.postings[term].append(Posting(doc_id, count))
                
            doc_id += 1

    def save_pickle(self, out_path: Path):
        """Зберігає індекс у бінарний формат pickle."""
        with out_path.open("wb") as f:
            pickle.dump((dict(self.postings), self.metadata), f)
            
    def save_json(self, out_path: Path):
        """Зберігає індекс у JSON (другий формат для порівняння)."""
        # JSON не вміє зберігати кастомні класи, тому перетворюємо їх на словники
        json_postings = {
            term: [{"doc_id": p.doc_id, "freq": p.frequency} for p in plist]
            for term, plist in self.postings.items()
        }
        json_meta = {
            doc_id: {"length": m.length, "path": m.path}
            for doc_id, m in self.metadata.items()
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"postings": json_postings, "metadata": json_meta}, f)

def main():
    parser = argparse.ArgumentParser(description="Build Inverted Index")
    parser.add_argument("data_dir", type=Path, help="Directory with corpus")
    parser.add_argument("--out", type=Path, required=True, help="Output index file base name")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = InvertedIndex()
    index.build(args.data_dir)

    build_time = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Index built in {build_time:.2f} s")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Total unique terms: {len(index.postings)}")

    # Збереження у двох форматах
    pickle_path = args.out.with_suffix('.pkl')
    t0 = time.perf_counter()
    index.save_pickle(pickle_path)
    print(f"Saved Pickle in {time.perf_counter() - t0:.2f} s. Size: {pickle_path.stat().st_size / 1024 / 1024:.2f} MB")

    json_path = args.out.with_suffix('.json')
    t0 = time.perf_counter()
    index.save_json(json_path)
    print(f"Saved JSON in {time.perf_counter() - t0:.2f} s. Size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()