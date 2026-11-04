import re
import unicodedata
from typing import Iterator

# Літери та цифри, можливі внутрішні дефіси та апострофи. 
# Наприклад: "м'ясо", "віце-президент", "42"
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")

def tokenize(text: str) -> Iterator[str]:
    """Нормалізує текст та ліниво видає токени за один прохід."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group(0)