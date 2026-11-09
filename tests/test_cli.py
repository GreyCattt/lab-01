from pathlib import Path
from typer.testing import CliRunner
from findex.cli import app

runner = CliRunner()

def test_cli_help():
    """Перевіряє, чи працює базова команда --help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Findex: Full-text search engine CLI" in result.stdout

def test_cli_index_build(tmp_path: Path):
    """Перевіряє, чи працює команда 'index' (створення індексу)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("tarot club", encoding="utf-8")
    
    out_file = tmp_path / "index.pkl"
    
    result = runner.invoke(app, ["index", str(data_dir), "--out", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    assert "Індекс збережено" in result.stderr

def test_cli_search_json(tmp_path: Path):
    """Перевіряє команду 'search' з прапорцем --json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("tarot club fool", encoding="utf-8")
    
    out_file = tmp_path / "index.pkl"
    runner.invoke(app, ["index", str(data_dir), "--out", str(out_file)])
    
    # Виконуємо пошук
    result = runner.invoke(app, ["search", str(out_file), "tarot", "--json"])
    assert result.exit_code == 0
    
    # Перевіряємо, чи вивід є валідним JSON і містить наш файл
    import json
    parsed = json.loads(result.stdout)
    assert len(parsed) == 1
    assert "doc1.txt" in parsed[0]["path"]

def test_cli_search_invalid_query(tmp_path: Path):
    """Вимога Лаби 4: Очікувана помилка при зламаному запиті має повертати код 1."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("tarot club", encoding="utf-8")
    
    out_file = tmp_path / "index.pkl"
    runner.invoke(app, ["index", str(data_dir), "--out", str(out_file)])
    
    # Зламаний запит
    result = runner.invoke(app, ["search", str(out_file), "tarot AND OR club"])
    assert result.exit_code == 1
    assert "Зламаний синтаксис запиту" in result.stderr