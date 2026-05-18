from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from src.data.paths import PROJECT_DIR

app = typer.Typer(help="CLI для управления detection-сервисом")


ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "models" / "best_detector.pt"
META_PATH = ARTIFACTS_DIR / "metrics" / "best_detector_meta.json"


@app.command()
def run_api(
    host: str = typer.Option("127.0.0.1", help="Хост для запуска API"),
    port: int = typer.Option(8000, help="Порт для запуска API"),
    reload: bool = typer.Option(True, help="Автоперезагрузка при изменениях кода"),
) -> None:
    """
    Запустить FastAPI detection-сервис через uvicorn.
    """
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def check_artifacts() -> None:
    """
    Проверить наличие основных артефактов модели.
    """
    typer.echo(f"Model path: {MODEL_PATH}")
    typer.echo(f"Exists: {MODEL_PATH.exists()}")

    typer.echo(f"\nMeta path: {META_PATH}")
    typer.echo(f"Exists: {META_PATH.exists()}")

    if MODEL_PATH.exists() and META_PATH.exists():
        typer.echo("\nArtifacts look OK")
    else:
        typer.echo("\nArtifacts are missing")
        raise typer.Exit(code=1)


@app.command()
def paths() -> None:
    """
    Показать ключевые пути проекта.
    """
    typer.echo(f"PROJECT_DIR: {PROJECT_DIR}")
    typer.echo(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")
    typer.echo(f"MODEL_PATH: {MODEL_PATH}")
    typer.echo(f"META_PATH: {META_PATH}")


if __name__ == "__main__":
    app()