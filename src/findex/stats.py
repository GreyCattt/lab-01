import tracemalloc
import time
import sys
from pathlib import Path
from collections import Counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize

def run_pipeline(data_dir: Path, eager: bool = False):
    tracemalloc.start()
    start_time = time.perf_counter()

    if eager:
        # Жадібний підхід: матеріалізуємо все у списки
        docs = list(iter_documents(data_dir))
        tokens = [token for doc in docs for token in tokenize(doc)]
        vocab = Counter(tokens)
        doc_count = len(docs)
        token_count = len(tokens)
    else:
        # Лінивий конвеєр: у пам'яті лише поточний документ та Counter
        vocab = Counter()# type: ignore - старий скрипт статистики, тип лічильника ігноруємо
        doc_count = 0
        token_count = 0
        for doc in iter_documents(data_dir):
            doc_count += 1
            for token in tokenize(doc):
                vocab[token] += 1
                token_count += 1

    duration = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mode_name = "EAGER (Lists)" if eager else "LAZY (Generators)"
    print(f"\n--- {mode_name} ---")
    print(f"Documents: {doc_count}")
    print(f"Total tokens: {token_count}")
    print(f"Vocabulary size: {len(vocab)}")# type: ignore - словник зібрано динамічно
    print(f"Time: {duration:.4f} s")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")
    
    if not eager:
        print("\nTop 5 terms:", vocab.most_common(5)) # type: ignore - тип елементів лічильника невідомий

if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not data_path.exists():
        print(f"Directory {data_path} not found. Create it and add some .txt files.")
        sys.exit(1)
        
    run_pipeline(data_path, eager=False)
    run_pipeline(data_path, eager=True)