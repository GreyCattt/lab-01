from findex.ast import And, Not, Or, Phrase, Term
from findex.search import parse_query


def test_parse_single_term():
    assert parse_query("hello") == Term("hello")

def test_parse_phrase():
    assert parse_query('"tarot club"') == Phrase(("tarot", "club"))

def test_parse_and():
    assert parse_query("tarot AND club") == And(Term("tarot"), Term("club"))

def test_parse_or_not():
    assert parse_query("tarot OR NOT club") == Or(Term("tarot"), Not(Term("club")))

def test_parse_complex_tree():
    """Тестуємо складне дерево з дужками: tarot AND (club OR fool)"""
    expected_tree = And(
        Term("tarot"),
        Or(Term("club"), Term("fool"))
    )
    assert parse_query("tarot AND (club OR fool)") == expected_tree

def test_parse_ignores_case_in_terms():
    """Токенізатор всередині парсера має приводити до нижнього регістру."""
    assert parse_query("TAROT") == Term("tarot")