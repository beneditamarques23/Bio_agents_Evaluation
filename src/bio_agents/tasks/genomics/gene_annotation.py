from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

_GENES = ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"]

_PROMPT = (
    "Annotate the following human genes and describe their known biological functions, "
    "associated diseases, and pathway memberships: " + ", ".join(_GENES) + "."
)


class GeneAnnotationTask(BioTask):
    @property
    def name(self) -> str:
        return "gene_annotation"

    @property
    def description(self) -> str:
        return (
            "Annotate a set of human genes with biological functions, "
            "associated diseases, and pathway memberships."
        )

    def get_input(self) -> TaskInput:
        return TaskInput(prompt=_PROMPT, context={"genes": _GENES})

    def get_tools(self) -> list:
        return []

    def evaluate(self, result: AgentResult) -> EvalResult:
        text = result.output or ""
        mentioned = [g for g in _GENES if g in text]
        score = round(len(mentioned) / len(_GENES), 2)

        return EvalResult(
            score=score,
            passed=score >= 0.6,
            metrics={
                "genes_mentioned": mentioned,
                "genes_missing": [g for g in _GENES if g not in mentioned],
                "output_length": len(text),
            },
        )
