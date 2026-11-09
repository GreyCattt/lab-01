import re
from collections.abc import Iterator


def tokenize(text: str) -> Iterator[str]:
    """Розбиває текст на токени (слова) у нижньому регістрі."""
    for match in re.finditer(r"[\w'-]+", text.lower()):
        yield match.group(0)