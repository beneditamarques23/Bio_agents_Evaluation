"""
Run a benchmark matrix defined by a YAML config.

Usage:
    uv run python pipelines/run_benchmark.py --config configs/quick_smoke.yaml
    uv run python pipelines/run_benchmark.py --config configs/full_benchmark.yaml \
        --dry-run
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Override output directory"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print matrix without running"
    ),
) -> None:
    from bio_agents.evaluation.runner import BenchmarkConfig, BenchmarkRunner

    cfg = BenchmarkConfig.from_yaml(config)

    if dry_run:
        table = Table(title=f"Benchmark matrix — {config.name}")
        table.add_column("Task", style="cyan")
        table.add_column("Framework", style="magenta")
        table.add_column("Model", style="green")
        for t in cfg.tasks:
            for f in cfg.frameworks:
                for m in cfg.models:
                    table.add_row(t, f, m)
        console.print(table)
        n = len(cfg.tasks) * len(cfg.frameworks) * len(cfg.models)
        console.print(f"[bold]{n}[/bold] total runs")
        return

    runner = BenchmarkRunner(cfg)
    console.print(f"[bold]Starting benchmark[/bold] — config: {config}")

    results = runner.run()
    out_file = runner.save(results, output_dir=output)

    console.print(f"\n[green]Done.[/green] Results saved to [bold]{out_file}[/bold]")
    console.print(
        f"Passed: {sum(r.eval_result.passed for r in results)}/{len(results)}"
    )


if __name__ == "__main__":
    app()
