from __future__ import annotations

import html
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, ConfigDict, Field

from findex.index import Index, open_index
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer
from findex.search import SearchEngine, parse_query

logger = logging.getLogger("findex.web")
WEB_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))
STATIC_DIR = WEB_DIR / "static"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _highlight_html(text: str, words: Iterable[str]) -> Markup:
    """Safely escape text and wrap matched query words in <mark> tags."""
    escaped = html.escape(text)
    clean_words = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    if not clean_words:
        return Markup(escaped)
    pattern = re.compile(
        "(" + "|".join(re.escape(html.escape(w)) for w in clean_words) + ")",
        re.IGNORECASE,
    )
    highlighted = pattern.sub(r"<mark>\1</mark>", escaped)
    return Markup(highlighted)


TEMPLATES.env.filters["highlight"] = _highlight_html  # type: ignore[assignment]


def _default_index_path() -> Path:
    return Path(os.getenv("INDEX_PATH", "index.pkl"))


def _default_page_size() -> int:
    raw = os.getenv("PAGE_SIZE", "10")
    try:
        return int(raw)
    except ValueError:
        return 10


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    model_config = ConfigDict(extra="ignore")

    index_path: Path = Field(default_factory=_default_index_path)
    page_size: int = Field(default_factory=_default_page_size, ge=1, le=100)
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> Settings:
        return cls.model_validate(
            {
                "index_path": os.getenv("INDEX_PATH", "index.pkl"),
                "page_size": os.getenv("PAGE_SIZE", "10"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            }
        )


class SearchItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    doc_id: int
    title: str
    score: float
    path: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: Literal["bm25", "tfidf"]
    page: int
    page_size: int
    total: int
    pages: int
    took_ms: float
    results: list[SearchItem]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    content: str


class StatsResponse(BaseModel):
    documents: int
    terms: int
    vocabulary: int
    average_document_length: float
    index_size_bytes: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "starting"]


@dataclass(slots=True)
class AppState:
    index: Index | None = None
    started_at: float = field(default_factory=time.monotonic)


def _resolve_document_path(raw_path: str) -> tuple[str, Path]:
    normalized = raw_path.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_file():
        return normalized, candidate
    fallback = DATA_DIR / candidate.name
    if fallback.is_file():
        return normalized, fallback
    return normalized, candidate


def _read_document(index: Index, doc_id: int) -> tuple[str, str]:
    try:
        meta = index.get_meta(doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    display_path, resolved_path = _resolve_document_path(meta.path)
    try:
        content = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not read document %s at %s", doc_id, resolved_path)
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return display_path, content


def _snippet(engine: SearchEngine, index: Index, doc_id: int, query_words: set[str]) -> str:
    _, content = _read_document(index, doc_id)
    raw_snippet = engine.generate_snippet(content, query_words)
    return ANSI_ESCAPE_RE.sub("", raw_snippet)


def _search(
    index: Index,
    query: str,
    k: int,
    scorer_name: Literal["bm25", "tfidf"],
    page: int,
) -> tuple[SearchResponse, set[str]]:
    t0 = time.perf_counter()
    if not query.strip():
        raise ValueError("Empty query")
    scorer: Scorer = BM25Scorer(index) if scorer_name == "bm25" else TfIdfScorer(index)
    engine = SearchEngine(index, scorer)
    ast = parse_query(query)
    query_words = engine.extract_raw_words(ast)
    max_docs = max(index.num_docs, 1)
    all_results = engine.search(query, top_k=max_docs)
    start = (page - 1) * k
    paged = all_results[start : start + k]
    total = len(all_results)
    pages = (total + k - 1) // k
    items: list[SearchItem] = []
    for result in paged:
        norm_path = index.get_meta(result.doc_id).path.replace("\\", "/")
        items.append(
            SearchItem(
                doc_id=result.doc_id,
                title=Path(norm_path).name,
                score=round(result.score, 6),
                path=norm_path,
                snippet=_snippet(engine, index, result.doc_id, query_words),
            )
        )
    took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    response = SearchResponse(
        query=query,
        scorer=scorer_name,
        page=page,
        page_size=k,
        total=total,
        pages=pages,
        took_ms=took_ms,
        results=items,
    )
    return response, query_words


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "findex", AppState()).index
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return index


def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return Settings.from_env()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = getattr(application.state, "settings", Settings.from_env())
    application.state.findex = AppState(started_at=time.monotonic())
    try:
        with open_index(settings.index_path) as index:
            application.state.findex.index = index
            logger.info(
                "Loaded index from %s (%d docs, %d terms)",
                settings.index_path,
                index.num_docs,
                len(index),
            )
            yield
    except Exception:
        logger.exception("Could not load index from %s", settings.index_path)
        yield
    finally:
        application.state.findex.index = None


def create_app(
    settings: Settings | None = None,
    *,
    async_search: bool = False,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    application = FastAPI(title="Findex", version="0.7.0", lifespan=lifespan)
    application.state.settings = runtime_settings
    if STATIC_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled exception request_id=%s", request_id)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        q: str = Query(default="", max_length=500),
        scorer: Literal["bm25", "tfidf"] = Query(default="bm25"),
        k: Annotated[int | None, Query(ge=1, le=100)] = None,
        page: int = Query(default=1, ge=1),
        doc_id: Annotated[int | None, Query(ge=0)] = None,
        index: Index = Depends(get_index),  # noqa: B008
        app_settings: Settings = Depends(get_settings),  # noqa: B008
    ) -> Response:
        page_size = k if k is not None else app_settings.page_size
        results: SearchResponse | None = None
        query_words: set[str] = set()
        error: str | None = None
        selected_doc: DocumentResponse | None = None

        if q.strip():
            try:
                results, query_words = _search(index, q, page_size, scorer, page)
            except ValueError:
                error = "Некоректний синтаксис запиту. Перевірте оператори AND, OR, NOT та дужки."

        if doc_id is not None:
            try:
                doc_path, doc_content = _read_document(index, doc_id)
                selected_doc = DocumentResponse(
                    doc_id=doc_id,
                    title=Path(doc_path).name,
                    path=doc_path,
                    content=doc_content,
                )
            except HTTPException:
                error = f"Документ #{doc_id} не знайдено."

        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "q": q,
                "scorer": scorer,
                "k": page_size,
                "results": results,
                "query_words": sorted(query_words),
                "selected_doc": selected_doc,
                "error": error,
                "stats": {
                    "documents": index.num_docs,
                    "terms": len(index),
                    "avg_len": round(index.avg_doc_length, 1),
                },
            },
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(
        request: Request,
        doc_id: int,
        q: str = Query(default="", max_length=500),
        index: Index = Depends(get_index),  # noqa: B008
    ) -> Response:
        path, content = _read_document(index, doc_id)
        query_words: set[str] = set()
        if q.strip():
            try:
                engine = SearchEngine(index, BM25Scorer(index))
                query_words = engine.extract_raw_words(parse_query(q))
            except ValueError:
                query_words = set()
        doc = DocumentResponse(
            doc_id=doc_id,
            title=Path(path).name,
            path=path,
            content=content,
        )
        return TEMPLATES.TemplateResponse(
            request=request,
            name="doc.html",
            context={"doc": doc, "q": q, "query_words": sorted(query_words)},
        )

    if async_search:

        @application.get("/search", response_model=SearchResponse)
        async def search_async(
            q: Annotated[str, Query(min_length=1, max_length=500)],
            k: Annotated[int | None, Query(ge=1, le=100)] = None,
            scorer: Literal["bm25", "tfidf"] = "bm25",
            page: Annotated[int, Query(ge=1)] = 1,
            index: Index = Depends(get_index),  # noqa: B008
            app_settings: Settings = Depends(get_settings),  # noqa: B008
        ) -> SearchResponse:
            page_size = k if k is not None else app_settings.page_size
            try:
                response, _ = _search(index, q, page_size, scorer, page)
                return response
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Invalid query syntax") from exc

    else:

        @application.get("/search", response_model=SearchResponse)
        def search(
            q: Annotated[str, Query(min_length=1, max_length=500)],
            k: Annotated[int | None, Query(ge=1, le=100)] = None,
            scorer: Literal["bm25", "tfidf"] = "bm25",
            page: Annotated[int, Query(ge=1)] = 1,
            index: Index = Depends(get_index),  # noqa: B008
            app_settings: Settings = Depends(get_settings),  # noqa: B008
        ) -> SearchResponse:
            page_size = k if k is not None else app_settings.page_size
            try:
                response, _ = _search(index, q, page_size, scorer, page)
                return response
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Invalid query syntax") from exc

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def document(doc_id: int, index: Index = Depends(get_index)) -> DocumentResponse:  # noqa: B008
        path, content = _read_document(index, doc_id)
        return DocumentResponse(doc_id=doc_id, title=Path(path).name, path=path, content=content)

    @application.get("/stats", response_model=StatsResponse)
    def stats(
        request: Request,
        index: Index = Depends(get_index),  # noqa: B008
        app_settings: Settings = Depends(get_settings),  # noqa: B008
    ) -> StatsResponse:
        state: AppState = getattr(request.app.state, "findex", AppState())
        uptime = max(0.0, time.monotonic() - state.started_at)
        try:
            size_bytes = app_settings.index_path.stat().st_size
        except OSError:
            size_bytes = 0
        return StatsResponse(
            documents=index.num_docs,
            terms=len(index),
            vocabulary=len(index),
            average_document_length=index.avg_doc_length,
            index_size_bytes=size_bytes,
            uptime_seconds=round(uptime, 2),
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        index = getattr(request.app.state, "findex", AppState()).index
        if index is None:
            raise HTTPException(status_code=503, detail="Index is not loaded")
        return HealthResponse(status="ok")

    return application


app = create_app()
