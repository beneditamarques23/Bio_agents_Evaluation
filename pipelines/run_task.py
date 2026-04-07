"""
Run a single task interactively.

Usage:
    uv run python pipelines/run_task.py literature_search \
        --framework anthropic_sdk --model claude-sonnet-4-6
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    task: str = typer.Argument(..., help="Task name (must be in TASK_REGISTRY)"),
    framework: str = typer.Option(..., "--framework", "-f", help="Framework name"),
    model: str = typer.Option(..., "--model", "-m", help="Model key"),
) -> None:
    from bio_agents.evaluation.runner import BenchmarkConfig, BenchmarkRunner

    cfg = BenchmarkConfig(
        tasks=[task],
        frameworks=[framework],
        models=[model],
        eval=["success_rate", "cost_latency"],
    )
    runner = BenchmarkRunner(cfg)
    results = runner.run()
    r = results[0]

    console.print(
        Panel(
            r.agent_result.output, title=f"{task} | {framework} | {model}", expand=False
        )
    )
    console.print(f"Score:    [bold]{r.eval_result.score:.2f}[/bold]")
    console.print(f"Passed:   [bold]{'yes' if r.eval_result.passed else 'no'}[/bold]")
    console.print(f"Duration: [bold]{r.duration_s:.1f}s[/bold]")
    if r.agent_result.tool_calls:
        console.print(
            f"Tools called: {[tc.get('name') for tc in r.agent_result.tool_calls]}"
        )


if __name__ == "__main__":
    app()
