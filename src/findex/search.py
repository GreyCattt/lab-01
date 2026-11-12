from __future__ import annotations

import argparse
import heapq
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from findex.ast import And, Not, Or, Phrase, QueryNode, Term
from findex.index import Index, open_index
from findex.models import SearchResult
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer
from findex.tokenize import tokenize
from findex.utils import timed


def parse_query(query: str) -> QueryNode:
    """
    Парсить рядок запиту в AST-дерево.
    Граматика:
      Expr    -> Term (OR Term)*
      Term    -> Factor (AND Factor)*
      Factor  -> NOT Factor | Primary
      Primary -> ( Expr ) | "phrase" | word
    """
    tokens = re.findall(r'\(|\)|AND|OR|NOT|"[^"]+"|[\w-]+', query)
    pos = 0

    def match(expected: str) -> bool:
        nonlocal pos
        if pos < len(tokens) and tokens[pos] == expected:
            pos += 1
            return True
        return False

    def parse_expr() -> QueryNode:
        node = parse_term()
        while match("OR"):
            node = Or(node, parse_term())
        return node

    def parse_term() -> QueryNode:
        node = parse_factor()
        while match("AND"):
            node = And(node, parse_factor())
        return node

    def parse_factor() -> QueryNode:
        if match("NOT"):
            return Not(parse_factor())
        return parse_primary()

    def parse_primary() -> QueryNode:
        nonlocal pos
        if pos >= len(tokens) or tokens[pos] in ("AND", "OR", "NOT", ")"):
            raise ValueError(
                f"Неочікуваний токен: {tokens[pos] if pos < len(tokens) else 'кінець рядка'}"
            )
        if match("("):
            node = parse_expr()
            if not match(")"):
                raise ValueError("Незакрита дужка")
            return node

        token = tokens[pos]
        pos += 1

        if token.startswith('"') and token.endswith('"'):
            words = tuple(tokenize(token))
            return Phrase(words)
        words = list(tokenize(token))
        return Term(words[0]) if words else Term(token)

    result = parse_expr()
    if pos < len(tokens):
        raise ValueError(f"Неочікуваний токен: {tokens[pos]}")
    return result


class SearchEngine:
    def __init__(self, index: Index, scorer: Scorer) -> None:
        self.index = index
        self.scorer = scorer

    def extract_raw_words(self, node: QueryNode) -> set[str]:
        """Рекурсивно витягує всі базові слова з AST для підсвітки та підрахунку балів."""
        if isinstance(node, Term):
            return {node.word}
        if isinstance(node, Phrase):
            return set(node.words)
        if isinstance(node, (And, Or)):
            return self.extract_raw_words(node.left) | self.extract_raw_words(node.right)
        if isinstance(node, Not):
            return self.extract_raw_words(node.child)
        return set()

    def search_legacy(self, query_str: str, top_k: int = 5) -> list[SearchResult]:
        """Базова реалізація з Лаби 3 (цикл по об'єктах Python + heapq.nlargest)."""
        ast = parse_query(query_str)
        matched_docs = ast.evaluate(self.index)
        query_words = self.extract_raw_words(ast)

        results: list[SearchResult] = []
        for doc_id in matched_docs:
            doc_len = self.index.get_meta(doc_id).length
            score = 0.0

            for word in query_words:
                postings = self.index[word]
                tf = next((p.frequency for p in postings if p.doc_id == doc_id), 0)
                if tf > 0:
                    df = self.index.df(word)
                    score += self.scorer.score(tf, df, doc_len)

            results.append(SearchResult(doc_id, score))

        return heapq.nlargest(top_k, results)

    @timed
    @lru_cache(maxsize=128)  # noqa: B019
    def search(self, query_str: str, top_k: int = 5) -> list[SearchResult]:
        """
        Векторизований пошук через NumPy int32 масиви постінгів та np.argpartition (Лаба 8).
        Порядок результатів повністю збігається з реалізацією Лаби 3 (search_legacy).
        """
        ast = parse_query(query_str)
        matched_docs = ast.evaluate(self.index)
        if not matched_docs or top_k <= 0:
            return []

        query_words = self.extract_raw_words(ast)
        doc_lengths = self.index.doc_lengths
        total_scores = np.zeros(len(doc_lengths), dtype=np.float64)

        for word in sorted(query_words):
            doc_ids, freqs = self.index.get_postings_arrays(word)
            if doc_ids.size > 0:
                df = int(doc_ids.size)
                dl = doc_lengths[doc_ids]
                total_scores[doc_ids] += self.scorer.score_array(freqs, df, dl)

        matched_arr = np.fromiter(matched_docs, dtype=np.int32, count=len(matched_docs))
        candidate_scores = total_scores[matched_arr]

        k = min(top_k, int(matched_arr.size))
        if matched_arr.size <= k:
            order = np.argsort(-candidate_scores, kind="stable")
        else:
            idx_range = np.arange(matched_arr.size, dtype=np.int32)
            keys = np.rec.fromarrays([-candidate_scores, idx_range], names="neg_score,idx")
            part = np.argpartition(keys, k - 1)[:k]
            sub_order = np.lexsort((idx_range[part], -candidate_scores[part]))
            order = part[sub_order]

        return [
            SearchResult(int(matched_arr[int(i)]), float(candidate_scores[int(i)]))
            for i in order
        ]

    def generate_snippet(self, text: str, query_words: set[str], window: int = 80) -> str:
        """Повертає вікно тексту навколо першого знайденого слова із запиту."""
        lower_text = text.casefold()
        best_idx = -1

        for word in query_words:
            idx = lower_text.find(word)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx

        if best_idx == -1:
            return text[: window * 2] + "..."

        start = max(0, best_idx - window)
        end = min(len(text), best_idx + window)
        snippet = text[start:end]

        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(lambda m: f"\033[93m\033[1m{m.group(0)}\033[0m", snippet)

        return f"... {snippet} ..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced Search Engine")
    parser.add_argument("index_file", type=Path, help="Path to index.pkl")
    parser.add_argument("query", type=str, help="Search query (e.g. 'tarot AND club')")
    parser.add_argument(
        "--scorer", choices=["bm25", "tfidf"], default="bm25", help="Ranking algorithm"
    )
    args = parser.parse_args()

    with open_index(args.index_file) as index:
        scorer = BM25Scorer(index) if args.scorer == "bm25" else TfIdfScorer(index)
        engine = SearchEngine(index, scorer)

        print(f"\n--- Query: {args.query} | Scorer: {args.scorer.upper()} ---")

        top_results = engine.search(args.query)
        engine.search(args.query)

        query_words = engine.extract_raw_words(parse_query(args.query))

        for rank, res in enumerate(top_results, 1):
            meta = index.get_meta(res.doc_id)
            text = Path(meta.path).read_text(encoding="utf-8")
            snippet = engine.generate_snippet(text, query_words)

            print(f"\n{rank}. {meta.path} (Score: {res.score:.4f})")
            print(snippet)


if __name__ == "__main__":
    main()
