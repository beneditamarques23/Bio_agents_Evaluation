"""
Assay Generation task.

Given a disease name, generate and rank experimental assays that could
be used to screen therapeutic candidates.

Input  (prompt): disease name
Output:          top assay description / candidate-generation goal string

Evaluation:
  - Structural: output must be a non-trivial string (> 50 chars)
  - Tool usage: at least one Crow search must have been executed
  - Future:     LLM-as-judge scoring against domain-expert rubric
"""

from __future__ import annotations

from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

# Default disease used when no disease_name is provided at instantiation.
_DEFAULT_DISEASE = "dry age-related macular degeneration"


class AssayGenerationTask(BioTask):
    """
    Bio task: propose and rank experimental assays for a given disease.

    Instantiate with a specific disease to benchmark or use the default.
    """

    def __init__(self, disease_name: str = _DEFAULT_DISEASE) -> None:
        self.disease_name = disease_name

    @property
    def name(self) -> str:
        return "assay_generation"

    @property
    def description(self) -> str:
        return (
            "Given a disease name, use literature search to propose and "
            "rank experimental assays for therapeutic candidate screening."
        )

    def get_input(self) -> TaskInput:
        return TaskInput(
            prompt=self.disease_name,
            context={"disease_name": self.disease_name},
        )

    def get_tools(self) -> list:
        """Crow is the primary external tool for this task."""
        from bio_agents.tools.futurehouse.crow import crow_search_sync

        return [crow_search_sync]

    def evaluate(self, result: AgentResult) -> EvalResult:
        output = result.output or ""
        num_tool_calls = len(result.tool_calls)

        # Structural checks (no ground truth available yet)
        has_output = len(output) > 50
        used_tools = num_tool_calls > 0

        passed = has_output
        score = sum([has_output, used_tools]) / 2

        return EvalResult(
            score=round(score, 2),
            passed=passed,
            metrics={
                "output_length": len(output),
                "num_tool_calls": num_tool_calls,
                "has_output": has_output,
                "used_tools": used_tools,
            },
        )
