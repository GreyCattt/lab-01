from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.index import Index
from findex.models import DocMeta
from findex.web.app import Settings, create_app, get_index


def _build_in_memory_index(tmp_path: Path) -> Index:
    (tmp_path / "doc0.txt").write_text("tarot club and the fool", encoding="utf-8")
    (tmp_path / "doc1.txt").write_text("tarot divination above the gray fog", encoding="utf-8")
    index = Index()
    index.build(tmp_path)
    return index


def _make_app(tmp_path: Path, *, async_search: bool = False) -> tuple[FastAPI, Index]:
    index = _build_in_memory_index(tmp_path)
    application = create_app(
        Settings(index_path=tmp_path / "nonexistent.pkl"),
        async_search=async_search,
    )
    application.dependency_overrides[get_index] = lambda: index
    return application, index


def test_search_returns_paginated_results(tmp_path: Path) -> None:
    application, _ = _make_app(tmp_path)
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        page1 = client.get("/search", params={"q": "tarot", "k": 1, "page": 1, "scorer": "bm25"})
        page2 = client.get("/search", params={"q": "tarot", "k": 1, "page": 2, "scorer": "bm25"})

    assert page1.status_code == 200
    body1: dict[str, Any] = page1.json()
    assert body1["total"] == 2
    assert body1["pages"] == 2
    assert body1["page"] == 1
    assert len(body1["results"]) == 1

    assert page2.status_code == 200
    body2: dict[str, Any] = page2.json()
    assert body2["page"] == 2
    assert len(body2["results"]) == 1
    assert body1["results"][0]["doc_id"] != body2["results"][0]["doc_id"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": "tarot", "k": 0},
        {"q": "tarot", "k": 101},
        {"q": "", "k": 5},
        {"q": "tarot", "scorer": "invalid"},
        {"q": "tarot", "page": 0},
        {"q": "tarot AND OR club"},
    ],
)
def test_search_validates_parameters(tmp_path: Path, params: dict[str, str | int]) -> None:
    application, _ = _make_app(tmp_path)
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/search", params=params)
    assert response.status_code == 422


def test_document_found_and_unknown_not_found(tmp_path: Path) -> None:
    application, _ = _make_app(tmp_path)
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        ok_resp = client.get("/docs/0")
        missing_resp = client.get("/docs/999")

    assert ok_resp.status_code == 200
    doc_body: dict[str, Any] = ok_resp.json()
    assert doc_body["doc_id"] == 0
    assert "tarot" in doc_body["content"]
    assert missing_resp.status_code == 404


def test_stats_endpoint(tmp_path: Path) -> None:
    application, _ = _make_app(tmp_path)
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/stats")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["documents"] == 2
    assert data["terms"] > 0
    assert data["average_document_length"] > 0


def test_health_is_unavailable_without_loaded_index(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "missing.pkl"))
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/health")
    assert response.status_code == 503


def test_unhandled_exception_returns_500_without_stack_and_logs_request_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    application, _ = _make_app(tmp_path)

    def _broken_index() -> Index:
        raise RuntimeError("secret internal failure details")

    application.dependency_overrides[get_index] = _broken_index
    with caplog.at_level(logging.ERROR, logger="findex.web"), TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/stats")

    assert response.status_code == 500
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert "secret internal failure details" not in response.text
    assert "Traceback" not in response.text
    assert any(request_id in record.message for record in caplog.records)


def test_home_page_renders_highlighted_snippets_and_doc_viewer(tmp_path: Path) -> None:
    application, _ = _make_app(tmp_path)
    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/", params={"q": "tarot", "doc_id": 0})

    assert response.status_code == 200
    assert "<mark>tarot</mark>" in response.text.lower()
    assert "Документ #0" in response.text


def test_search_snippet_fallback_uses_filename_for_windows_style_path(tmp_path: Path) -> None:
    application, index = _make_app(tmp_path)
    meta = index.get_meta(0)
    index._metadata[0] = DocMeta(doc_id=meta.doc_id, length=meta.length, path="nested\\doc0.txt")  # type: ignore[attr-defined]
    (tmp_path / "doc0.txt").unlink()

    with TestClient(application) as raw_client:
        client = cast(httpx.Client, raw_client)
        response = client.get("/search", params={"q": "club", "k": 5, "scorer": "bm25"})

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["results"]
    assert "doc0.txt" in body["results"][0]["snippet"]


def test_20_concurrent_search_requests_def_vs_async_def(tmp_path: Path) -> None:
    sync_app, _ = _make_app(tmp_path, async_search=False)
    async_app, _ = _make_app(tmp_path, async_search=True)

    def _run_20(application: FastAPI) -> tuple[float, list[int]]:
        with TestClient(application) as raw_client:
            client = cast(httpx.Client, raw_client)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                futures = [
                    pool.submit(client.get, "/search", params={"q": "tarot", "k": 5})
                    for _ in range(20)
                ]
                statuses = [f.result().status_code for f in futures]
            elapsed = time.perf_counter() - start
            return elapsed, statuses

    def_elapsed, def_statuses = _run_20(sync_app)
    async_elapsed, async_statuses = _run_20(async_app)

    assert def_statuses == [200] * 20
    assert async_statuses == [200] * 20
    assert def_elapsed > 0
    assert async_elapsed > 0
