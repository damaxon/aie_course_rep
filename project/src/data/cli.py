from __future__ import annotations

import typer

from .core import (
    check_kaggle,
    check_kaggle_config,
    delete_all,
    download_all,
    full_pipeline,
    organize_all,
    prepare_all,
)

app = typer.Typer(help="CLI для управления данными проекта (download / organize / clean)")

@app.command()
def download() -> None:
    """
    Скачивание raw-датасетов.
    """

    if not check_kaggle():
        typer.echo("Kagglehub не установлен. Установите его с помощью 'pip install kagglehub'")
        raise typer.Exit(code=1)

    if not check_kaggle_config():
        typer.echo("configs/kaggle.json не найден")
        raise typer.Exit(code=1)
    
    results = download_all()

    for name,success in results.items():
        status = "OK" if success else "FAIL"
        typer.echo(f"{name}: {status}")
    
    if not all(results.values()):
        raise typer.Exit(code=1)
    
@app.command()
def organize() -> None:
    """
    Организация структуры данных.
    """

    results = organize_all()

    for name,success in results.items():
        status = "OK" if success else "NOT FOUND"
        typer.echo(f"{name}: {status}")

@app.command()
def clean() -> None:
    """
    Удаление raw-данных.
    """

    count = delete_all()
    typer.echo(f"Удалено {count} файлов.")

@app.command()
def prepare() -> None:
    """
    Подготовить proccessed данные
    """

    results = prepare_all()

    typer.echo("\n=== DETECTION ===")
    typer.echo(f"Copied images: {results['detection']['copied_images']}")
    typer.echo(f"Annotations created: {results['detection']['annotations_created']}")
    typer.echo(f"Number of annotation rows: {results['detection']['num_annotations']}")

@app.command()
def full() -> None:
    """
    Полный pipeline подготовки данных: download -> organize -> prepare
    """
    if not check_kaggle():
        typer.echo("Kagglehub не установлен. Установите его с помощью 'pip install kagglehub'")
        raise typer.Exit(code=1)

    if not check_kaggle_config():
        typer.echo("configs/kaggle.json не найден")
        raise typer.Exit(code=1)

    results = full_pipeline()

    typer.echo("=== DOWNLOAD ===")
    for name, success in results["download"].items():
        status = "OK" if success else "FAIL"
        typer.echo(f"{name}: {status}")

    if not all(results["download"].values()):
        raise typer.Exit(code=1)

    typer.echo("\n=== ORGANIZE ===")
    for name, success in results["organize"].items():
        status = "OK" if success else "NOT FOUND"
        typer.echo(f"{name}: {status}")

    typer.echo("\n=== PREPARE ===")

    det = results["prepare"]["detection"]

    typer.echo("\n=== DETECTION ===")
    typer.echo(f"Copied images: {det['copied_images']}")
    typer.echo(f"Annotations created: {det['annotations_created']}")
    typer.echo(f"Number of annotation rows: {det['num_annotations']}")

if __name__ == "__main__":
    app()
