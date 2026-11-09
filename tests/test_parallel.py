from pathlib import Path
from findex.index import Index, build_partial

def test_build_partial_and_merge_equivalence(tmp_path: Path):
    """Доводить, що нова архітектура генерує ідентичний індекс."""
    doc1 = tmp_path / "1.txt"
    doc2 = tmp_path / "2.txt"
    doc1.write_text("the quick brown fox", encoding="utf-8")
    doc2.write_text("fox jumps over", encoding="utf-8")

    # 1. Збірка монолітом (як у Лабі 4)
    idx_mono = Index()
    idx_mono.build(tmp_path)

    # 2. Збірка частинами з подальшим merge
    idx_chunk1 = build_partial([doc1], start_doc_id=0)
    idx_chunk2 = build_partial([doc2], start_doc_id=1)
    
    idx_merged = Index()
    idx_merged.merge(idx_chunk1)
    idx_merged.merge(idx_chunk2)

    # Перевіряємо еквівалентність
    assert idx_merged.num_docs == idx_mono.num_docs
    assert len(idx_merged) == len(idx_mono)
    for term, postings in idx_mono.items():
        assert idx_merged[term] == postings