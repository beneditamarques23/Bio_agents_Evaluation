"""
Eval1Runner — runs the Biomni-Eval1 benchmark dataset against any framework/model.

Parallel to BenchmarkRunner but designed for dataset-based evaluation:
  - Loads instances from biomni/Eval1 (HuggingFace)
  - Expands task_names × instances_per_task into individual BiomniEval1Task objects
  - Loops: instances × frameworks × models
  - Saves JSONL compatible with BenchmarkRunner output (+ eval1-specific fields)

Usage:
  runner = Eval1Runner.from_config("configs/biomni_eval1_smoke.yaml")
  results = runner.run()
  runner.save(results)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bio_agents.frameworks.base import AgentResult, AgentRunner
from bio_agents.tasks.base import EvalResult
from bio_agents.tasks.biomni_eval1.task import EVAL1_TASK_NAMES, BiomniEval1Task


@dataclass
class Eval1Config:
    task_names: list[str]  # task types to run, or ["all"] for all 10
    frameworks: list[str]
    models: list[str]
    instances_per_task: int = 1  # how many instances per task type
    output_dir: str = "results"
    runner_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Eval1Config":
        known = {
            "task_names",
            "frameworks",
            "models",
            "instances_per_task",
            "output_dir",
        }
        with open(path) as f:
            data = yaml.safe_load(f)
        runner_kwargs = {k: v for k, v in data.items() if k not in known}
        base = {k: v for k, v in data.items() if k in known}
        return cls(**base, runner_kwargs=runner_kwargs)

    def resolved_task_names(self) -> list[str]:
        if self.task_names == ["all"]:
            return EVAL1_TASK_NAMES
        return self.task_names


@dataclass
class Eval1RunResult:
    eval1_task_name: str
    task_instance_id: int
    framework: str
    model: str
    agent_result: AgentResult
    eval_result: EvalResult
    duration_s: float

    def to_dict(self) -> dict:
        return {
            # Eval1-specific identifiers
            "eval1_task_name": self.eval1_task_name,
            "task_instance_id": self.task_instance_id,
            # Standard benchmark fields (compatible with BenchmarkRunner output)
            "task": f"eval1_{self.eval1_task_name}",
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


class Eval1Runner:
    def __init__(self, config: Eval1Config):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path) -> "Eval1Runner":
        return cls(Eval1Config.from_yaml(path))

    def run(self) -> list[Eval1RunResult]:
        instances = self._load_instances()
        results = []
        for task in instances:
            for framework_name in self.config.frameworks:
                runner = self._load_runner(framework_name)
                for model in self.config.models:
                    result = self._run_single(task, runner, model)
                    results.append(result)
        return results

    def dry_run(self) -> list[dict]:
        """Return the planned run matrix without executing anything."""
        instances = self._load_instances()
        plan = []
        for task in instances:
            for framework_name in self.config.frameworks:
                for model in self.config.models:
                    plan.append(
                        {
                            "eval1_task_name": task._task_name,
                            "task_instance_id": task._task_instance_id,
                            "framework": framework_name,
                            "model": model,
                        }
                    )
        return plan

    def save(
        self, results: list[Eval1RunResult], output_dir: str | Path | None = None
    ) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "results.jsonl"
        with open(output_file, "a") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        return output_file

    def _load_instances(self) -> list[BiomniEval1Task]:
        from datasets import load_dataset  # type: ignore[import]

        dataset = load_dataset("biomni/Eval1", split="test")
        task_names = self.config.resolved_task_names()
        instances: list[BiomniEval1Task] = []

        for task_name in task_names:
            dataset_rows: list[dict] = list(dataset)  # type: ignore[arg-type]
            rows = [r for r in dataset_rows if r["task_name"] == task_name]
            if not rows:
                raise ValueError(
                    f"No instances found for task '{task_name}' in biomni/Eval1. "
                    f"Available: {EVAL1_TASK_NAMES}"
                )
            count = min(self.config.instances_per_task, len(rows))
            for row in rows[:count]:
                instances.append(
                    BiomniEval1Task(
                        task_name=task_name,
                        task_instance_id=row["task_instance_id"],
                        prompt=row["prompt"],
                    )
                )

        return instances

    def _run_single(
        self, task: BiomniEval1Task, runner: AgentRunner, model: str
    ) -> Eval1RunResult:
        task_input = task.get_input()
        tools = task.get_tools()

        start = time.perf_counter()
        agent_result = runner.run(
            task_input.prompt, tools, model, **self.config.runner_kwargs
        )
        duration = time.perf_counter() - start

        eval_result = task.evaluate(agent_result)

        return Eval1RunResult(
            eval1_task_name=task._task_name,
            task_instance_id=task._task_instance_id,
            framework=runner.framework_name,
            model=model,
            agent_result=agent_result,
            eval_result=eval_result,
            duration_s=round(duration, 3),
        )

    def _load_runner(self, name: str) -> AgentRunner:
        from bio_agents.frameworks import FRAMEWORK_REGISTRY

        if name not in FRAMEWORK_REGISTRY:
            raise ValueError(
                f"Unknown framework: '{name}'. Available: {list(FRAMEWORK_REGISTRY)}"
            )
        return FRAMEWORK_REGISTRY[name]()
