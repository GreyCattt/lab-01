from findex.index import Index
import tempfile

def test_index_num_docs(sample_index: Index):
    """Перевіряє, чи правильно пораховано кількість документів."""
    assert sample_index.num_docs == 2

def test_index_contains(sample_index: Index):
    """Перевіряє роботу dunder-методу __contains__ (оператор in)."""
    assert "hello" in sample_index
    assert "findex" in sample_index
    assert "missing" not in sample_index

def test_index_len(sample_index: Index):
    """Перевіряє кількість унікальних слів (словник)."""
    # hello, world, findex, the, quick, brown, fox, jumps
    assert len(sample_index) == 8

def test_index_df(sample_index: Index):
    """Перевіряє Document Frequency (в скількох документах є слово)."""
    # 'hello' є тільки в doc1, хоча зустрічається там 2 рази. df має бути 1.
    assert sample_index.df("hello") == 1
    assert sample_index.df("world") == 1

def test_index_avg_doc_length(sample_index: Index):
    """Перевіряє властивість cached_property."""
    # doc1: 4 слова, doc2: 5 слів. Середнє: 4.5
    assert sample_index.avg_doc_length == 4.5

from pathlib import Path
from hypothesis import given, strategies as st

# --- Hypothesis Property-Based Тести ---

@given(random_text=st.text(min_size=1, max_size=100))
def test_postings_are_always_sorted(random_text: str):
    """
    Властивість 1: Постінги завжди відсортовані за doc_id.
    Hypothesis генерує випадковий текст 100 разів.
    Для кожної ітерації ми створюємо абсолютно нову, чисту тимчасову папку.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_dir = Path(temp_dir)
        
        # Створюємо кілька тимчасових документів
        (doc_dir / "1.txt").write_text(random_text, encoding="utf-8")
        (doc_dir / "2.txt").write_text(random_text, encoding="utf-8")
        (doc_dir / "3.txt").write_text(random_text, encoding="utf-8")
        
        idx = Index()
        idx.build(doc_dir)
        
        for _, postings in idx.items():
            doc_ids = [p.doc_id for p in postings]
            assert doc_ids == sorted(doc_ids)

