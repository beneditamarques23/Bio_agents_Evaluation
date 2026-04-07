import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bio_agents.frameworks.base import AgentResult, AgentRunner
from bio_agents.tasks.base import BioTask, EvalResult


@dataclass
class BenchmarkConfig:
    tasks: list[str]
    frameworks: list[str]
    models: list[str]
    eval: list[str]
    output_dir: str = "results"
    # Optional per-run kwargs forwarded to AgentRunner.run()
    runner_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        known = {"tasks", "frameworks", "models", "eval", "output_dir"}
        with open(path) as f:
            data = yaml.safe_load(f)
        runner_kwargs = {k: v for k, v in data.items() if k not in known}
        base = {k: v for k, v in data.items() if k in known}
        return cls(**base, runner_kwargs=runner_kwargs)


@dataclass
class RunResult:
    task: str
    framework: str
    model: str
    agent_result: AgentResult
    eval_result: EvalResult
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "framework": self.framework,
            "model": self.model,
            "output": self.agent_result.output,
            "tool_calls": self.agent_result.tool_calls,
            "score": self.eval_result.score,
            "passed": self.eval_result.passed,
            "metrics": self.eval_result.metrics,
            "duration_s": self.duration_s,
            "agent_metadata": self.agent_result.metadata,
        }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path) -> "BenchmarkRunner":
        return cls(BenchmarkConfig.from_yaml(path))

    def run(self) -> list[RunResult]:
        results = []
        for task_name in self.config.tasks:
            task = self._load_task(task_name)
            for framework_name in self.config.frameworks:
                runner = self._load_runner(framework_name)
                for model in self.config.models:
                    result = self._run_single(task, runner, model)
                    results.append(result)
        return results

    def save(
        self, results: list[RunResult], output_dir: str | Path | None = None
    ) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "results.jsonl"
        with open(output_file, "a") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        return output_file

    def _run_single(self, task: BioTask, runner: AgentRunner, model: str) -> RunResult:
        task_input = task.get_input()
        tools = task.get_tools()

        start = time.perf_counter()
        agent_result = runner.run(
            task_input.prompt, tools, model, **self.config.runner_kwargs
        )
        duration = time.perf_counter() - start

        eval_result = task.evaluate(agent_result)

        return RunResult(
            task=task.name,
            framework=runner.framework_name,
            model=model,
            agent_result=agent_result,
            eval_result=eval_result,
            duration_s=round(duration, 3),
        )

    def _load_task(self, name: str) -> BioTask:
        from bio_agents.tasks import TASK_REGISTRY

        if name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task: '{name}'. Available: {list(TASK_REGISTRY)}"
            )
        return TASK_REGISTRY[name]()

    def _load_runner(self, name: str) -> AgentRunner:
        from bio_agents.frameworks import FRAMEWORK_REGISTRY

        if name not in FRAMEWORK_REGISTRY:
            raise ValueError(
                f"Unknown framework: '{name}'. Available: {list(FRAMEWORK_REGISTRY)}"
            )
        return FRAMEWORK_REGISTRY[name]()
