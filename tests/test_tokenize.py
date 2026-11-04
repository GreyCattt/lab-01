from findex.tokenize import tokenize

def test_tokenize_cyrillic_and_symbols():
    text = "М'ясо, віце-президент та 42 коти пішли у café!"
    tokens = list(tokenize(text))
    assert tokens == ["м'ясо", "віце-президент", "та", "42", "коти", "пішли", "у", "café"]

def test_unicode_normalization():
    # "cafe" + combining acute accent (\u0301)
    text1 = "cafe\u0301"
    # precomposed "café"
    text2 = "café"
    assert list(tokenize(text1)) == list(tokenize(text2))