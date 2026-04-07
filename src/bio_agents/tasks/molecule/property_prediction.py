from bio_agents.frameworks.base import AgentResult
from bio_agents.tasks.base import BioTask, EvalResult, TaskInput

_DRUGS = ["aspirin", "ibuprofen"]
_ADMET_TERMS = ["bioavailability", "absorption", "toxicity", "metabolism", "excretion"]

_PROMPT = (
    "Predict and compare the key ADMET properties (absorption, distribution, "
    "metabolism, excretion, toxicity) of aspirin (acetylsalicylic acid) and "
    "ibuprofen. Which has better oral bioavailability?"
)


class MoleculePropertyTask(BioTask):
    @property
    def name(self) -> str:
        return "molecule_property"

    @property
    def description(self) -> str:
        return (
            "Predict and compare ADMET properties of small molecules "
            "and assess oral bioavailability."
        )

    def get_input(self) -> TaskInput:
        return TaskInput(prompt=_PROMPT, context={"drugs": _DRUGS})

    def get_tools(self) -> list:
        return []

    def evaluate(self, result: AgentResult) -> EvalResult:
        text = (result.output or "").lower()
        drugs_found = [d for d in _DRUGS if d in text]
        admet_found = [t for t in _ADMET_TERMS if t in text]

        both_drugs = len(drugs_found) == 2
        admet_count = len(admet_found)

        if both_drugs and admet_count >= 2:
            score = 1.0
        elif both_drugs or admet_count >= 1:
            score = 0.5
        else:
            score = 0.0

        return EvalResult(
            score=score,
            passed=score >= 0.5,
            metrics={
                "drugs_mentioned": drugs_found,
                "admet_terms_found": admet_found,
                "output_length": len(text),
            },
        )
