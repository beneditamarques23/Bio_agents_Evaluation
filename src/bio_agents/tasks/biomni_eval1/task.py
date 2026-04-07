"""
BiomniEval1Task — wraps a single instance from the biomni/Eval1 HuggingFace dataset.

Not registered in TASK_REGISTRY. Instantiated dynamically by Eval1Runner.

Dataset: https://huggingface.co/datasets/biomni/Eval1
  - 433 test instances across 10 biological task types
  - Scoring: BiomniEval1().evaluate(task_name, task_instance_id, response) → 0.0 or 1.0

Requires:
  - biomni extra installed: uv sync --extra biomni (or: make setup-biomni)
"""

from __future__ import annotations

from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

# All 10 task types present in the dataset
EVAL1_TASK_NAMES = [
    "gwas_causal_gene_opentargets",
    "gwas_variant_prioritization",
    "gwas_causal_gene_abc",
    "gwas_causal_gene_ot_genetics",
    "screen_gene_retrieval",
    "lab_bench_seqqa",
    "rare_disease_diagnosis",
    "crispr_delivery",
    "patient_gene_detection",
    "lab_bench_cloningqa",
]


class BiomniEval1Task(BioTask):
    """A single instance from the Biomni-Eval1 benchmark dataset."""

    def __init__(self, task_name: str, task_instance_id: int, prompt: str) -> None:
        self._task_name = task_name
        self._task_instance_id = task_instance_id
        self._prompt = prompt

    @property
    def name(self) -> str:
        return f"eval1_{self._task_name}_{self._task_instance_id}"

    @property
    def description(self) -> str:
        return (
            f"Biomni-Eval1 instance: task={self._task_name}, "
            f"instance_id={self._task_instance_id}"
        )

    def get_input(self) -> TaskInput:
        return TaskInput(
            prompt=self._prompt,
            context={
                "task_name": self._task_name,
                "task_instance_id": self._task_instance_id,
                "dataset": "biomni/Eval1",
            },
        )

    def get_tools(self) -> list:
        return []

    def evaluate(self, result: AgentResult) -> EvalResult:
        try:
            from biomni.eval.biomni_eval1 import BiomniEval1  # type: ignore[import]

            evaluator = BiomniEval1()
            score = float(
                evaluator.evaluate(
                    self._task_name, self._task_instance_id, result.output
                )
            )
        except Exception as exc:
            # Gracefully degrade — record the error and score 0
            return EvalResult(
                score=0.0,
                passed=False,
                metrics={
                    "eval_error": str(exc),
                    "task_name": self._task_name,
                    "task_instance_id": self._task_instance_id,
                },
            )

        return EvalResult(
            score=score,
            passed=score > 0.0,
            metrics={
                "task_name": self._task_name,
                "task_instance_id": self._task_instance_id,
                "output_length": len(result.output or ""),
            },
        )
