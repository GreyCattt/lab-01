from collections.abc import Iterator
from pathlib import Path


def iter_documents(directory: Path) -> Iterator[str]:
    """Ліниво читає всі .txt файли у директорії, не завантажуючи їх одночасно в пам'ять."""
    for file_path in directory.rglob("*.txt"):
        if file_path.is_file():
            yield file_path.read_text(encoding="utf-8")