from pathlib import Path

import pytest

from findex.index import Index


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """
    Створює тимчасову директорію з двома маленькими документами 
    для тестування побудови індексу.
    tmp_path - вбудована фікстура pytest, яка автоматично видаляється після тестів.
    """
    # Створюємо файли у тимчасовій папці
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Hello world! Hello Findex.", encoding="utf-8")
    
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("The quick brown fox jumps.", encoding="utf-8")
    
    return tmp_path

@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    """Повертає вже зібраний індекс на основі тимчасового корпусу."""
    idx = Index()
    idx.build(sample_corpus)
    return idx