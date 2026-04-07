"""
Candidate Ranking task.

Given an assay-generation goal (output of AssayGenerationTask), generate
and rank therapeutic candidates using literature review and pairwise
LLM comparison.

Input  (prompt): candidate-generation goal string (or disease name as fallback)
Output:          ranked candidates summary; CSV written to output_dir

Evaluation:
  - Structural: non-trivial output string
  - File check: ranked_therapeutic_candidates.csv exists in output_dir
  - Tool usage: Crow + Falcon calls recorded
  - Future:     LLM-as-judge / domain-expert rubric
"""

from __future__ import annotations

import os

from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

_DEFAULT_DISEASE = "dry age-related macular degeneration"


class CandidateRankingTask(BioTask):
    """
    Bio task: generate and rank therapeutic candidates for an assay goal.

    Instantiate with a pre-computed candidate_goal (from AssayGenerationTask)
    or just a disease_name to run stand-alone.
    """

    def __init__(
        self,
        disease_name: str = _DEFAULT_DISEASE,
        candidate_goal: str = "",
    ) -> None:
        self.disease_name = disease_name
        self.candidate_goal = candidate_goal

    @property
    def name(self) -> str:
        return "candidate_ranking"

    @property
    def description(self) -> str:
        return (
            "Generate and rank therapeutic candidates for a given assay goal "
            "using literature search and pairwise LLM comparison."
        )

    def get_input(self) -> TaskInput:
        return TaskInput(
            prompt=self.candidate_goal or self.disease_name,
            context={
                "disease_name": self.disease_name,
                "candidate_goal": self.candidate_goal,
            },
        )

    def get_tools(self) -> list:
        """Crow (search) + Falcon (deep reports) are both used here."""
        from bio_agents.tools.futurehouse.crow import crow_search_sync
        from bio_agents.tools.futurehouse.falcon import falcon_report_sync

        return [crow_search_sync, falcon_report_sync]

    def evaluate(self, result: AgentResult) -> EvalResult:
        output = result.output or ""
        output_dir = result.metadata.get("output_dir", "")
        num_tool_calls = len(result.tool_calls)

        # Check if Robin wrote the ranked candidates CSV
        candidates_csv = (
            os.path.join(output_dir, "ranked_therapeutic_candidates.csv")
            if output_dir
            else ""
        )
        has_csv = bool(candidates_csv) and os.path.exists(candidates_csv)
        has_output = len(output) > 50
        used_tools = num_tool_calls > 0

        # Score: 0.33 each for output, tools, CSV file
        score = sum([has_output, used_tools, has_csv]) / 3
        passed = has_output  # minimum bar

        return EvalResult(
            score=round(score, 2),
            passed=passed,
            metrics={
                "output_length": len(output),
                "num_tool_calls": num_tool_calls,
                "has_output": has_output,
                "used_tools": used_tools,
                "has_candidates_csv": has_csv,
                "output_dir": output_dir,
            },
        )
