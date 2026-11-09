import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Імпортуємо нашу логіку
from findex.index import Index, open_index
from findex.search import SearchEngine, parse_query
from findex.ranking import BM25Scorer, TfIdfScorer

# Всі діагностичні логи та помилки йдуть у stderr, щоб не ламати stdout (--json)
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

@app.command("index")
def build_index(
    data_dir: Path = typer.Argument(..., help="Шлях до корпусу текстів", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out", help="Шлях для збереження index.pkl")
):
    """Будує інвертований індекс із вказаної директорії."""
    logger = logging.getLogger("findex.index")
    logger.info(f"Розпочинаємо побудову індексу з директорії: {data_dir}")
    
    idx = Index()
    
    # Використовуємо rich.progress для красивої анімації завантаження
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console
    ) as progress:
        task = progress.add_task(f"Індексація файлів...", total=None)
        idx.build(data_dir)
        progress.update(task, description="[green]Індексацію завершено![/green]")

    logger.info(f"Збереження індексу у файл {out}")
    idx.save(out)
    err_console.print(f"[bold green]Успіх![/bold green] Індекс збережено (Слів: {len(idx)}).")


@app.command("search")
def search_index(
    index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True, dir_okay=False),
    query: str = typer.Argument(..., help="Пошуковий запит"),
    scorer: str = typer.Option("bm25", help="Алгоритм ранжування (bm25, tfidf)"),
    json_out: bool = typer.Option(False, "--json", help="Вивести результати у форматі JSON в stdout")
):
    """Виконує пошук по існуючому індексу."""
    logger = logging.getLogger("findex.search")
    
    # Вимога лаби: Очікувана помилка - один зрозумілий рядок, код виходу 1
    try:
        ast = parse_query(query)
    except Exception as e:
        err_console.print("[red]Помилка:[/red] Зламаний синтаксис запиту. Перевірте дужки та оператори.")
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
                
    except Exception as e:
        err_console.print(f"[red]Помилка завантаження індексу:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("stats")
def index_stats(index_file: Path = typer.Argument(..., help="Шлях до файлу index.pkl", exists=True)):
    """Виводить статистику побудованого індексу."""
    with open_index(index_file) as index:
        Console().print(f"[bold cyan]Статистика індексу {index_file}:[/bold cyan]")
        Console().print(f"Документів: [green]{index.num_docs}[/green]")
        Console().print(f"Унікальних термінів: [green]{len(index)}[/green]")
        Console().print(f"Середня довжина документа: [green]{index.avg_doc_length:.1f} слів[/green]")

if __name__ == "__main__":
    app()