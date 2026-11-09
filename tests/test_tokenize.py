import pytest
from findex.tokenize import tokenize

@pytest.mark.parametrize(
    "text, expected",
    [
        # Базові тести
        ("Hello World", ["hello", "world"]),
        ("  Spaces  here ", ["spaces", "here"]),
        ("", []),
        
        # Знаки пунктуації та спеціальні символи
        ("Hello, world!", ["hello", "world"]),
        ("Email: greycat@example.com", ["email", "greycat", "example", "com"]),
        ("hyphen-word and don't", ["hyphen-word", "and", "don't"]),
        
        # Unicode (Вимога Лаби 4)
        ("Привіт світ", ["привіт", "світ"]),
        ("Café & résumé", ["café", "résumé"]),
        ("こんにちは世界", ["こんにちは世界"]), # Японська (якщо токенізатор це підтримує)
        
        # Числа та змішані рядки
        ("Year 2026", ["year", "2026"]),
        ("C# and .NET", ["c", "and", "net"]),
    ]
)
def test_tokenize_variations(text: str, expected: list[str]):
    """Перевіряє правильність розбиття тексту на токени."""
    assert list(tokenize(text)) == expected