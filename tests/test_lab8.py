from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.index import Index, build_partial
from findex.ranking import BM25Scorer, TfIdfScorer
from findex.search import SearchEngine
from findex.semantic import SemanticIndex, encode_text, reciprocal_rank_fusion
from findex.web.app import Settings, create_app, get_index

runner = CliRunner()


def _create_sample_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = [
        "tarot club gathering above the gray fog with the fool and justice",
        "klein moretti used pendulum divination and spirit vision to predict fate",
        "crimson moon and scarlet moonlight shone over the cathedral of evernight",
        "audrey hall and her golden retriever susie studied spectator psychology",
        "alger wilson sailed the ghost ship across the stormy ocean waves",
        "inspector dunn smith and the nighthawks investigated the murder crime",
        "emperor roselle gustav wrote secret diary pages in simplified chinese symbols",
        "revolver brass bullets and firearm holster used for marksmanship practice",
    ]
    for i, text in enumerate(docs):
        (corpus / f"doc_{i}.txt").write_text(text, encoding="utf-8")
    return corpus


def test_numpy_int32_postings_and_doc_lengths(tmp_path: Path) -> None:
    corpus = _create_sample_corpus(tmp_path)
    index = Index()
    index.build(corpus)

    assert index.doc_lengths.dtype == np.int32
    assert index.doc_lengths.shape == (8,)

    doc_ids, freqs = index.get_postings_arrays("tarot")
    assert doc_ids.dtype == np.int32
    assert freqs.dtype == np.int32
    assert doc_ids.tolist() == [0]
    assert freqs.tolist() == [1]


@pytest.mark.parametrize("scorer_cls", [BM25Scorer, TfIdfScorer])
@pytest.mark.parametrize(
    "query",
    [
        "tarot",
        "the",
        "tarot OR crimson OR ocean",
        "the AND (tarot OR crimson OR ship)",
        "NOT tarot",
    ],
)
def test_numpy_search_matches_lab3_order_exactly(
    tmp_path: Path,
    scorer_cls: type[BM25Scorer | TfIdfScorer],
    query: str,
) -> None:
    corpus = _create_sample_corpus(tmp_path)
    index = Index()
    index.build(corpus)
    scorer = scorer_cls(index)
    engine = SearchEngine(index, scorer)

    legacy_results = engine.search_legacy(query, top_k=5)
    numpy_results = engine.search(query, top_k=5)

    assert [r.doc_id for r in numpy_results] == [r.doc_id for r in legacy_results]
    for np_res, leg_res in zip(numpy_results, legacy_results, strict=True):
        assert np_res.score == pytest.approx(leg_res.score, rel=1e-9)


def test_semantic_and_hybrid_rrf_search(tmp_path: Path) -> None:
    corpus = _create_sample_corpus(tmp_path)
    npy_path = tmp_path / "embeddings.npy"

    # CLI embed command
    res = runner.invoke(cli_app, ["embed", str(corpus), "--out", str(npy_path)])
    assert res.exit_code == 0
    assert npy_path.is_file()

    sem_idx = SemanticIndex.load(npy_path)
    assert sem_idx.embeddings.dtype == np.float32
    # Check L2 normalization
    norms = np.linalg.norm(sem_idx.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Paraphrase query without exact words from doc_1 ("klein moretti used pendulum divination...")
    paraphrase = "astrology fortune telling prophecy clairvoyance"
    sem_results = sem_idx.search(paraphrase, top_k=3)
    assert len(sem_results) > 0
    assert sem_results[0].doc_id == 1

    # RRF fusion test
    index = Index()
    index.build(corpus)
    bm25_engine = SearchEngine(index, BM25Scorer(index))
    bm25_res = bm25_engine.search("tarot", top_k=5)
    fused = reciprocal_rank_fusion(bm25_res, sem_results, top_k=5)
    assert len(fused) >= 2
    assert {r.doc_id for r in fused}.issuperset({0, 1})

    # Verify vector normalization helper directly
    q_vec = encode_text(paraphrase)
    assert float(np.linalg.norm(q_vec)) == pytest.approx(1.0, rel=1e-5)


def test_web_api_semantic_and_hybrid_modes(tmp_path: Path) -> None:
    corpus = _create_sample_corpus(tmp_path)
    index = Index()
    index.build(corpus)

    application = create_app(Settings(index_path=tmp_path / "missing.pkl"))
    application.dependency_overrides[get_index] = lambda: index

    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        sem_resp = client.get(
            "/search",
            params={"q": "fortune telling prophecy", "mode": "semantic", "k": 3},
        )
        hyb_resp = client.get(
            "/search",
            params={"q": "tarot OR prophecy", "mode": "hybrid", "k": 3},
        )

    assert sem_resp.status_code == 200
    sem_body: dict[str, Any] = sem_resp.json()
    assert sem_body["mode"] == "semantic"
    assert sem_body["results"][0]["doc_id"] == 1

    assert hyb_resp.status_code == 200
    hyb_body: dict[str, Any] = hyb_resp.json()
    assert hyb_body["mode"] == "hybrid"
    assert len(hyb_body["results"]) > 0


def test_benchmark_build_index(tmp_path: Path, benchmark: Any) -> None:
    corpus = _create_sample_corpus(tmp_path)
    paths = sorted(corpus.glob("*.txt"))
    bench_fn: Callable[..., Index] = benchmark
    result = bench_fn(build_partial, paths, 0)
    assert result.num_docs == 8


def test_benchmark_search(tmp_path: Path, benchmark: Any) -> None:
    corpus = _create_sample_corpus(tmp_path)
    index = Index()
    index.build(corpus)
    engine = SearchEngine(index, BM25Scorer(index))

    def _run_search() -> int:
        SearchEngine.search.__wrapped__.cache_clear()  # type: ignore[attr-defined]
        return len(engine.search("the AND (tarot OR crimson)", top_k=5))

    bench_fn: Callable[..., int] = benchmark
    count = bench_fn(_run_search)
    assert count >= 1
