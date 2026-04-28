"""
Run a single task interactively.

Usage:
    uv run python pipelines/run_task.py assay_generation \
        --framework robin --model llama3

    # Pass framework-specific kwargs with --kwarg KEY=VALUE (repeatable)
    uv run python pipelines/run_task.py assay_generation \
        --framework robin --model llama3 \
        --kwarg lite_mode=true \
        --kwarg num_assays=2

    # Biomni kwargs
    uv run python pipelines/run_task.py literature_search \
        --framework biomni --model llama3 \
        --kwarg use_tool_retriever=true \
        --kwarg timeout_seconds=300
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False)
console = Console()


def _parse_value(raw: str) -> bool | int | float | str:
    """Coerce a raw string value to bool / int / float / str."""
    if raw.lower() in ("true", "yes", "1"):
        return True
    if raw.lower() in ("false", "no", "0"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@app.command()
def main(
    task: str = typer.Argument(..., help="Task name (must be in TASK_REGISTRY)"),
    framework: str = typer.Option(..., "--framework", "-f", help="Framework name"),
    model: str = typer.Option(..., "--model", "-m", help="Model key"),
    kwarg: list[str] = typer.Option(
        [],
        "--kwarg",
        "-k",
        help=(
            "Extra kwarg forwarded to the framework runner, as KEY=VALUE. "
            "Repeatable. Values are auto-cast to bool / int / float / str. "
            "Example: --kwarg lite_mode=true --kwarg num_assays=2"
        ),
    ),
) -> None:
    from bio_agents.evaluation.runner import BenchmarkConfig, BenchmarkRunner

    runner_kwargs: dict = {}
    for kv in kwarg:
        if "=" not in kv:
            raise typer.BadParameter(
                f"--kwarg must be KEY=VALUE, got: {kv!r}", param_hint="--kwarg"
            )
        key, _, value = kv.partition("=")
        runner_kwargs[key.strip()] = _parse_value(value.strip())

    cfg = BenchmarkConfig(
        tasks=[task],
        frameworks=[framework],
        models=[model],
        eval=["success_rate", "cost_latency"],
        runner_kwargs=runner_kwargs,
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
    if runner_kwargs:
        console.print(f"Runner kwargs: {runner_kwargs}")


if __name__ == "__main__":
    app()
