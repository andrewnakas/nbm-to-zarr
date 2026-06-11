"""CLI entry point for NBM to Zarr."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from nbm_to_zarr.base.dataset import Dataset
from nbm_to_zarr.noaa.nbm.analysis import NbmAnalysisDataset
from nbm_to_zarr.noaa.nbm.forecast import NbmForecastDataset

app = typer.Typer(name="nbm", help="NBM to Zarr reformatter", add_completion=False)
console = Console()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Registry of available datasets (id -> factory).
DATASETS: dict[str, type[Dataset]] = {
    "noaa-nbm-conus-forecast": NbmForecastDataset,
    "noaa-nbm-conus-analysis": NbmAnalysisDataset,
}


def _get_dataset(dataset_id: str, include_std: bool) -> Dataset:
    if dataset_id not in DATASETS:
        rprint(f"[red]Error: dataset {dataset_id} not found[/red]")
        rprint(f"Available: {', '.join(DATASETS)}")
        raise typer.Exit(1)
    return DATASETS[dataset_id](include_std=include_std)  # type: ignore[call-arg]


@app.command()
def list_datasets() -> None:
    """List available datasets."""
    table = Table(title="Available NBM Datasets")
    table.add_column("ID", style="cyan")
    table.add_column("Variant", style="magenta")
    table.add_column("Description", style="green")
    for dataset_id, cls in DATASETS.items():
        attrs = cls().template_config.dataset_attributes
        table.add_row(dataset_id, attrs.variant, attrs.title)
    console.print(table)


@app.command()
def info(dataset_id: str = typer.Argument(..., help="Dataset ID")) -> None:
    """Show information about a dataset."""
    ds = _get_dataset(dataset_id, include_std=False)
    tc = ds.template_config
    attrs = tc.dataset_attributes
    rprint(f"\n[bold cyan]{attrs.title}[/bold cyan]\n[dim]{attrs.description}[/dim]\n")
    table = Table(show_header=False, box=None)
    table.add_row("ID", attrs.id)
    table.add_row("Provider / Model", f"{attrs.provider} / {attrs.model}")
    table.add_row("Variant", attrs.variant)
    table.add_row("Dimensions", str(tc.dimensions))
    table.add_row("Append dimension", f"{tc.append_dim} (freq {tc.append_dim_freq})")
    table.add_row("Variables", ", ".join(v.name for v in tc.data_vars))
    console.print(table)


@app.command()
def update_template(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    output_dir: Path = typer.Option(Path("./templates"), "--output", "-o"),
    end: str = typer.Option("2025-12-31", help="Append-dim end date (YYYY-MM-DD)"),
) -> None:
    """Materialize the empty template Zarr (schema only)."""
    ds = _get_dataset(dataset_id, include_std=False)
    end_dt = datetime.fromisoformat(end)
    template = ds.template_config.get_template(end_dt)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{dataset_id}.zarr"
    template.to_zarr(out, mode="w", zarr_format=2)
    rprint(f"[green]✓ Template written to {out}[/green]")
    rprint(f"  dims={dict(template.sizes)} vars={list(template.data_vars)}")


@app.command()
def operational_update(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    output_path: Path = typer.Option(..., "--output", "-o"),
    include_std: bool = typer.Option(False, "--include-std"),
) -> None:
    """Fetch the latest cycle/day and (over)write a compact store."""
    ds = _get_dataset(dataset_id, include_std=include_std)
    ds.operational_update(output_path)
    rprint(f"[green]✓ Operational update complete: {output_path}[/green]")


@app.command()
def backfill(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    output_path: Path = typer.Option(..., "--output", "-o"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    include_std: bool = typer.Option(False, "--include-std"),
) -> None:
    """Backfill a historical date range, growing the store along the append dim."""
    ds = _get_dataset(dataset_id, include_std=include_std)
    ds.backfill(output_path, datetime.fromisoformat(start), datetime.fromisoformat(end))
    rprint(f"[green]✓ Backfill complete: {output_path}[/green]")


if __name__ == "__main__":
    app()
