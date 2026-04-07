import re

from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

_PROMPT = (
    "Search the recent literature on CAR-T cell therapy resistance mechanisms "
    "in solid tumors. Summarize the key findings and list the most cited papers."
)

# Matches (2018), (2019), ..., (2025) or a DOI pattern
_REF_PATTERN = re.compile(r"\(20\d{2}\)|10\.\d{4,}/\S+")


class LiteratureSearchTask(BioTask):
    @property
    def name(self) -> str:
        return "literature_search"

    @property
    def description(self) -> str:
        return (
            "Retrieve and summarize recent literature on a biomedical topic, "
            "with paper references."
        )

    def get_input(self) -> TaskInput:
        return TaskInput(prompt=_PROMPT)

    def get_tools(self) -> list:
        return []

    def evaluate(self, result: AgentResult) -> EvalResult:
        text = result.output or ""
        refs = _REF_PATTERN.findall(text)
        n_refs = len(set(refs))

        if n_refs >= 2:
            score = 1.0
        elif len(text) > 200:
            score = 0.5
        else:
            score = 0.0

        return EvalResult(
            score=score,
            passed=score >= 0.5,
            metrics={"references_found": n_refs, "output_length": len(text)},
        )
