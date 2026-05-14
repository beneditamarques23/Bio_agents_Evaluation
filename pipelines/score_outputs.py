"""
Score external outputs against the Biomni-Eval1 benchmark.

This script takes a file of agent outputs collected from any source outside
bio-agents-playground (e.g., another tool, a third-party API, manual runs)
and evaluates them with the same BiomniEval1 exact-match scorer used by
run_eval1.py.

Input file requirements (JSONL or CSV):
    task_name         (str)  — one of the 10 Biomni-Eval1 task types
    task_instance_id  (int)  — row ID within that task type in biomni/Eval1
    output            (str)  — the agent response to score

    Any extra columns (framework, model, run_id, …) are preserved as-is.

Usage:
    # Score a JSONL file
    uv run python pipelines/score_outputs.py \\
        --input path/to/external_outputs.jsonl \\
        --output results/my_scored_run

    # Score a CSV file
    uv run python pipelines/score_outputs.py \\
        --input path/to/external_outputs.csv \\
        --output results/my_scored_run

    # Dry-run: preview input records without scoring
    uv run python pipelines/score_outputs.py \\
        --input path/to/external_outputs.jsonl \\
        --dry-run
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


def _load_records(path: Path) -> list[dict]:
    """Load records from a JSONL or CSV file."""
    import csv
    import json

    suffix = path.suffix.lower()
    records = []

    if suffix == ".jsonl":
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    else:
        raise typer.BadParameter(
            f"Input must be a .jsonl or .csv file, got: {suffix!r}",
            param_hint="--input",
        )

    return records


@app.command()
def main(
    input: Path = typer.Option(
        ..., "--input", "-i", help="Path to input JSONL or CSV file"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output directory for results.jsonl"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview input records without scoring"
    ),
) -> None:
    records = _load_records(input)
    console.print(
        f"[bold]Loaded {len(records)} record(s)[/bold] from [cyan]{input}[/cyan]"
    )

    if dry_run:
        table = Table(title=f"Input preview — {input.name}")
        table.add_column("task_name", style="cyan")
        table.add_column("task_instance_id", style="yellow")
        table.add_column("output (first 80 chars)", style="white")
        for r in records:
            table.add_row(
                str(r.get("task_name", "")),
                str(r.get("task_instance_id", "")),
                str(r.get("output", ""))[:80],
            )
        console.print(table)
        console.print(
            f"[bold]{len(records)}[/bold] record(s) — dry-run, nothing scored"
        )
        return

    if output is None:
        raise typer.BadParameter(
            "--output is required when not using --dry-run", param_hint="--output"
        )

    from bio_agents.evaluation.standalone_scorer import StandaloneEval1Scorer

    scorer = StandaloneEval1Scorer()

    console.print("Scoring...")
    results = scorer.score_batch(records)
    out_file = scorer.save(results, output_dir=output)
    stats = scorer.summary(results)

    console.print(f"\n[green]Done.[/green] Results saved to [bold]{out_file}[/bold]")
    console.print(
        f"Overall: {stats['n_passed']}/{stats['n_total']} passed  |  "
        f"Avg score: {stats['avg_score']:.3f}"
    )

    if stats["per_task"]:
        table = Table(title="Score by task type")
        table.add_column("task_name", style="cyan")
        table.add_column("n_total", justify="right")
        table.add_column("n_passed", justify="right", style="green")
        table.add_column("pass_rate", justify="right", style="bold")
        for task, s in sorted(stats["per_task"].items()):
            table.add_row(
                task,
                str(s["n_total"]),
                str(s["n_passed"]),
                f"{s['pass_rate']:.0%}",
            )
        console.print(table)


if __name__ == "__main__":
    app()
