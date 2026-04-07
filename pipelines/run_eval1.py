"""
Run the Biomni-Eval1 benchmark dataset against any framework/model.

Usage:
    uv run python pipelines/run_eval1.py --config configs/biomni_eval1_smoke.yaml
    uv run python pipelines/run_eval1.py --config configs/biomni_eval1_full.yaml \
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
    from bio_agents.evaluation.eval1_runner import Eval1Config, Eval1Runner

    cfg = Eval1Config.from_yaml(config)
    runner = Eval1Runner(cfg)

    if dry_run:
        plan = runner.dry_run()
        table = Table(title=f"Eval1 matrix — {config.name}")
        table.add_column("Task", style="cyan")
        table.add_column("Instance ID", style="yellow")
        table.add_column("Framework", style="magenta")
        table.add_column("Model", style="green")
        for row in plan:
            table.add_row(
                row["eval1_task_name"],
                str(row["task_instance_id"]),
                row["framework"],
                row["model"],
            )
        console.print(table)
        console.print(f"[bold]{len(plan)}[/bold] total runs")
        return

    console.print(f"[bold]Starting Eval1 benchmark[/bold] — config: {config}")

    results = runner.run()
    out_file = runner.save(results, output_dir=output)

    console.print(f"\n[green]Done.[/green] Results saved to [bold]{out_file}[/bold]")
    passed = sum(r.eval_result.passed for r in results)
    avg_score = (
        sum(r.eval_result.score for r in results) / len(results) if results else 0.0
    )
    console.print(f"Passed: {passed}/{len(results)}  |  Avg score: {avg_score:.3f}")


if __name__ == "__main__":
    app()
