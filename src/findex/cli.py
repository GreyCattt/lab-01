import json
import logging
from pathlib import Path
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from findex.index import Index, open_index, build_partial
from findex.ranking import BM25Scorer, TfIdfScorer
from findex.search import SearchEngine, parse_query
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
console = Console()
err_console = Console(stderr=True)
app = typer.Typer(add_completion=False, help="Findex: Full-text search engine CLI")

def setup_logging(verbose_count: int):
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
        handlers=[RichHandler(console=err_console, rich_tracebacks=False, show_path=False)]
    )

@app.callback()
def main_callback(
    v: int = typer.Option(0, "-v", count=True, help="Підняти рівень логів (-v INFO, -vv DEBUG)")
):
    setup_logging(v)

class ExecutorType(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"

@app.command("index")
def build_index(
    data_dir: Path = typer.Argument(..., help="Шлях до корпусу текстів", exists=True, file_okay=False), # noqa: B008
    out: Path = typer.Option(..., "--out", help="Шлях для збереження index.pkl"), # noqa: B008
    workers: int = typer.Option(1, "--workers", help="Кількість паралельних воркерів"), # noqa: B008
    executor: ExecutorType = typer.Option(ExecutorType.serial, "--executor", help="Тип паралелізму") # noqa: B008
):
    
    console.print(f"Збираємо індекс з [cyan]{data_dir}[/cyan]...")
    paths = list(data_dir.glob("**/*.txt"))
    if not paths:
        err_console.print("[red]Документи не знайдені![/red]")
        raise typer.Exit(code=1)

    # Розбиваємо шляхи на шматки (chunks)
    chunk_size = math.ceil(len(paths) / workers) if workers > 0 else len(paths)
    chunks: list[tuple[list[Path], int]] = []
    start_id = 0
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
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
                    indices.append(future.result()) # Якщо воркер впав, помилка вилетить тут
                    
        elif executor == ExecutorType.processes:
            ctx = mp.get_context("spawn") # Безпечний запуск процесів
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = [pool.submit(build_partial, chunk, s_id) for chunk, s_id in chunks]
                for future in as_completed(futures):
                    indices.append(future.result())
                    
    except Exception as e:
        err_console.print(f"[red]Помилка у воркері під час збірки:[/red] {e}")
        raise typer.Exit(code=1)

    # Фаза Merge
    final_index = Index()
    for partial_idx in indices:
        final_index.merge(partial_idx)

    final_index.save(out)
    console.print(f"[green]Індекс успішно збережено у {out}[/green]!")
    console.print(f"Оброблено документів: [bold]{final_index.num_docs}[/bold]")


@app.command("search")
def search_index(
    index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True, dir_okay=False),# noqa: B008
    query: str = typer.Argument(..., help="Пошуковий запит"),
    scorer: str = typer.Option("bm25", help="Алгоритм ранжування (bm25, tfidf)"),
    json_out: bool = typer.Option(False, "--json", help="Вивести результати у форматі JSON в stdout")# noqa: B008
):
    """Виконує пошук по існуючому індексу."""
    logger = logging.getLogger("findex.search")
    
    # Вимога лаби: Очікувана помилка - один зрозумілий рядок, код виходу 1
    try:
        ast = parse_query(query)
    except ValueError:
        err_console.print("[red]Помилка:[/red] Зламаний синтаксис запиту.")
        raise typer.Exit(code=1)

    logger.debug(f"AST Дерево: {ast}")
    
    try:
        with open_index(index_file) as index:
            scorer_obj = BM25Scorer(index) if scorer == "bm25" else TfIdfScorer(index)
            engine = SearchEngine(index, scorer_obj)
            
            logger.info(f"Пошук: '{query}' | Алгоритм: {scorer.upper()}")
            results = engine.search(query)
            query_words = engine.extract_raw_words(ast)

            if json_out:
                # Вивід JSON суворо в stdout (sys.stdout)
                output_data = [
                    {"path": str(index.get_meta(res.doc_id).path), "score": round(res.score, 4)} 
                    for res in results
                ]
                print(json.dumps(output_data, indent=2))
            else:
                # Малюємо красиву таблицю rich для людини
                table = Table(title=f"Результати пошуку: {query}", show_lines=True)
                table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
                table.add_column("Score", justify="right", style="magenta")
                table.add_column("Document", style="green")
                table.add_column("Snippet", style="white")

                for rank, res in enumerate(results, 1):
                    meta = index.get_meta(res.doc_id)
                    text = Path(meta.path).read_text(encoding="utf-8")
                    snippet = engine.generate_snippet(text, query_words)
                    table.add_row(str(rank), f"{res.score:.4f}", Path(meta.path).name, snippet)

                Console().print(table)
                
    except Exception as e: # noqa: BLE001
        err_console.print(f"[red]Помилка завантаження індексу:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("stats")
def index_stats(index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True)): # noqa: B008
    """Виводить статистику побудованого індексу."""
    with open_index(index_file) as index:
        Console().print(f"[bold cyan]Статистика індексу {index_file}:[/bold cyan]")
        Console().print(f"Документів: [green]{index.num_docs}[/green]")
        Console().print(f"Унікальних термінів: [green]{len(index)}[/green]")
        Console().print(f"Середня довжина документа: [green]{index.avg_doc_length:.1f} слів[/green]")

if __name__ == "__main__":
    app()