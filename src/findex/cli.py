from __future__ import annotations

import asyncio
import json
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table

from findex.crawler import crawl
from findex.index import Index, build_partial, open_index
from findex.models import SearchResult
from findex.ranking import BM25Scorer, TfIdfScorer
from findex.search import SearchEngine, parse_query
from findex.semantic import SemanticIndex, reciprocal_rank_fusion
from findex.tokenize import tokenize

console = Console()
err_console = Console(stderr=True)
app = typer.Typer(add_completion=False, help="Findex: Full-text search engine CLI")


def setup_logging(verbose_count: int) -> None:
    if verbose_count == 0:
        level = logging.WARNING
    elif verbose_count == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=False, show_path=False)],
    )


@app.callback()
def main_callback(
    v: int = typer.Option(0, "-v", count=True, help="Підняти рівень логів (-v INFO, -vv DEBUG)"),
) -> None:
    setup_logging(v)


class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


@app.command("index")
def build_index(
    data_dir: Path = typer.Argument(..., help="Шлях до корпусу текстів", exists=True, file_okay=False),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="Шлях для збереження index.pkl"),  # noqa: B008
    workers: int = typer.Option(1, "--workers", help="Кількість паралельних воркерів"),
    executor: ExecutorType = typer.Option(ExecutorType.serial, "--executor", help="Тип паралелізму"),  # noqa: B008
) -> None:
    console.print(f"Збираємо індекс з [cyan]{data_dir}[/cyan]...")
    paths = sorted(data_dir.glob("**/*.txt"))
    if not paths:
        err_console.print("[red]Документи не знайдені![/red]")
        raise typer.Exit(code=1)

    chunk_size = math.ceil(len(paths) / workers) if workers > 0 else len(paths)
    chunks: list[tuple[list[Path], int]] = []
    start_id = 0
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i : i + chunk_size]
        chunks.append((chunk, start_id))
        start_id += len(chunk)

    indices: list[Index] = []

    try:
        if executor == ExecutorType.serial:
            for chunk, s_id in chunks:
                indices.append(build_partial(chunk, s_id))

        elif executor == ExecutorType.threads:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build_partial, chunk, s_id) for chunk, s_id in chunks]
                for future in as_completed(futures):
                    indices.append(future.result())

        elif executor == ExecutorType.processes:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = [pool.submit(build_partial, chunk, s_id) for chunk, s_id in chunks]
                for future in as_completed(futures):
                    indices.append(future.result())

    except Exception as e:
        err_console.print(f"[red]Помилка у воркері під час збірки:[/red] {e}")
        raise typer.Exit(code=1) from e

    final_index = Index()
    for partial_idx in indices:
        final_index.merge(partial_idx)

    final_index.save(out)
    console.print(f"[green]Індекс успішно збережено у {out}[/green]!")
    console.print(f"Оброблено документів: [bold]{final_index.num_docs}[/bold]")


@app.command("embed")
def embed_corpus(
    data_dir: Path = typer.Argument(..., help="Шлях до корпусу текстів", exists=True, file_okay=False),  # noqa: B008
    out: Path = typer.Option(Path("embeddings.npy"), "--out", help="Шлях для збереження матриці .npy"),  # noqa: B008
    chunk_words: int = typer.Option(180, "--chunk-words", min=20, help="Розмір фрагмента у словах"),
) -> None:
    """Ділить документи на фрагменти, обчислює нормалізовані ембединги та зберігає матрицю у .npy."""
    console.print(f"Обчислюємо ембединги фрагментів для [cyan]{data_dir}[/cyan]...")
    sem_idx = SemanticIndex.from_corpus(data_dir, chunk_words=chunk_words)
    sem_idx.save(out)
    console.print(
        f"[green]Ембединги збережено у {out}[/green] "
        f"(фрагментів: [bold]{sem_idx.embeddings.shape[0]}[/bold], "
        f"розмірність: [bold]{sem_idx.embeddings.shape[1]}[/bold])!"
    )


@app.command("search")
def search_index(
    index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True, dir_okay=False),  # noqa: B008
    query: str = typer.Argument(..., help="Пошуковий запит"),
    scorer: str = typer.Option("bm25", help="Алгоритм ранжування (bm25, tfidf)"),
    mode: str = typer.Option("bm25", "--mode", help="Режим пошуку (bm25, tfidf, semantic, hybrid)"),
    embeddings: Path = typer.Option(Path("embeddings.npy"), "--embeddings", help="Шлях до .npy ембедингів"),  # noqa: B008
    top_k: int = typer.Option(5, "-k", "--top-k", min=1, max=100, help="Кількість результатів"),
    json_out: bool = typer.Option(False, "--json", help="Вивести результати у форматі JSON в stdout"),
) -> None:
    """Виконує лексичний (BM25/TF-IDF), семантичний або гібридний (RRF) пошук по індексу."""
    logger = logging.getLogger("findex.search")
    effective_mode = mode if mode != "bm25" else scorer

    ast = None
    if effective_mode in ("bm25", "tfidf"):
        try:
            ast = parse_query(query)
        except ValueError as exc:
            err_console.print("[red]Помилка:[/red] Зламаний синтаксис запиту.")
            raise typer.Exit(code=1) from exc

    try:
        with open_index(index_file) as index:
            scorer_obj = TfIdfScorer(index) if effective_mode == "tfidf" else BM25Scorer(index)
            engine = SearchEngine(index, scorer_obj)

            logger.info(f"Пошук: '{query}' | Режим: {effective_mode.upper()}")
            results: list[SearchResult]
            if effective_mode in ("bm25", "tfidf"):
                results = engine.search(query, top_k=top_k)
                query_words = engine.extract_raw_words(ast) if ast else set(tokenize(query))
            else:
                sem_idx = (
                    SemanticIndex.load(embeddings)
                    if embeddings.is_file()
                    else SemanticIndex.from_inverted_index(index)
                )
                sem_results = sem_idx.search(query, top_k=max(top_k * 4, 20))
                if effective_mode == "semantic":
                    results = sem_results[:top_k]
                else:
                    try:
                        bm25_results = engine.search(query, top_k=max(top_k * 4, 20))
                    except ValueError:
                        q_tokens = list(tokenize(query))
                        fallback_q = " OR ".join(q_tokens) if q_tokens else query
                        bm25_results = (
                            engine.search(fallback_q, top_k=max(top_k * 4, 20))
                            if q_tokens
                            else []
                        )
                    results = reciprocal_rank_fusion(bm25_results, sem_results, top_k=top_k)
                query_words = set(tokenize(query))

            if json_out:
                output_data = [
                    {
                        "doc_id": res.doc_id,
                        "path": str(index.get_meta(res.doc_id).path),
                        "score": round(res.score, 4),
                    }
                    for res in results
                ]
                print(json.dumps(output_data, indent=2))
            else:
                table = Table(
                    title=f"Результати пошуку ({effective_mode.upper()}): {query}",
                    show_lines=True,
                )
                table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
                table.add_column("Score", justify="right", style="magenta")
                table.add_column("Document", style="green")
                table.add_column("Snippet", style="white")

                for rank, res in enumerate(results, 1):
                    meta = index.get_meta(res.doc_id)
                    doc_path = Path(meta.path.replace("\\", "/"))
                    if doc_path.is_file():
                        text = doc_path.read_text(encoding="utf-8")
                        snippet = engine.generate_snippet(text, query_words)
                    else:
                        snippet = f"... {doc_path.name} ({meta.length} слів) ..."
                    table.add_row(str(rank), f"{res.score:.4f}", doc_path.name, snippet)

                Console().print(table)

    except typer.Exit:
        raise
    except Exception as e:
        err_console.print(f"[red]Помилка завантаження індексу:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command("stats")
def index_stats(
    index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True),  # noqa: B008
) -> None:
    """Виводить статистику побудованого індексу."""
    with open_index(index_file) as index:
        Console().print(f"[bold cyan]Статистика індексу {index_file}:[/bold cyan]")
        Console().print(f"Документів: [green]{index.num_docs}[/green]")
        Console().print(f"Унікальних термінів: [green]{len(index)}[/green]")
        Console().print(f"Середня довжина документа: [green]{index.avg_doc_length:.1f} слів[/green]")


async def run_crawler(start_urls: list[str], max_pages: int, concurrency: int) -> None:
    stats: dict[str, int] = {"pending_requests": 0, "errors": 0, "queue_depth": 0}
    start_time = time.monotonic()
    collected = 0

    with (
        open("documents.jsonl", "w", encoding="utf-8") as f,  # noqa: ASYNC230
        Live(refresh_per_second=4) as live,
    ):
        async for doc in crawl(start_urls, max_pages, concurrency, stats):
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            collected += 1

            elapsed = time.monotonic() - start_time
            pages_per_sec = collected / elapsed if elapsed > 0 else 0

            table = Table(title="Findex Crawler Status")
            table.add_column("Метрика", style="cyan")
            table.add_column("Значення", style="magenta")
            table.add_row("Зібрано сторінок", str(collected))
            table.add_row("Швидкість (стор/сек)", f"{pages_per_sec:.2f}")
            table.add_row("Запитів в очікуванні", str(stats["pending_requests"]))
            table.add_row("Помилок", str(stats["errors"]))
            table.add_row("Глибина черги", str(stats["queue_depth"]))

            live.update(table)


@app.command("crawl")
def crawl_cmd(
    url: str = typer.Argument(..., help="Start URL"),
    max_pages: int = typer.Option(200, "--max-pages", help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(20, "--concurrency", help="Кількість одночасних запитів"),
) -> None:
    """Асинхронний краулер для збору документів"""
    asyncio.run(run_crawler([url], max_pages, concurrency))


@app.command("serve")
def serve_cmd(
    port: int = typer.Option(
        8000, "--port", envvar="PORT", min=1, max=65535, help="Порт HTTP-сервера"
    ),
    workers: int = typer.Option(
        1, "--workers", envvar="WEB_CONCURRENCY", min=1, help="Кількість процесів uvicorn"
    ),
) -> None:
    """Запускає HTTP API та вебінтерфейс."""
    import uvicorn

    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)


if __name__ == "__main__":
    app()
