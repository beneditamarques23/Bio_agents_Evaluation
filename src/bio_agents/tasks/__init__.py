from bio_agents.tasks.base import BioTask, EvalResult, TaskInput
from bio_agents.tasks.drug_discovery.assay_generation import AssayGenerationTask
from bio_agents.tasks.drug_discovery.candidate_ranking import CandidateRankingTask
from bio_agents.tasks.genomics.gene_annotation import GeneAnnotationTask
from bio_agents.tasks.literature.search import LiteratureSearchTask
from bio_agents.tasks.molecule.property_prediction import MoleculePropertyTask

# Registry maps task name → BioTask subclass.
TASK_REGISTRY: dict[str, type[BioTask]] = {
    "assay_generation": AssayGenerationTask,
    "candidate_ranking": CandidateRankingTask,
    "literature_search": LiteratureSearchTask,
    "gene_annotation": GeneAnnotationTask,
    "molecule_property": MoleculePropertyTask,
}

__all__ = [
    "BioTask",
    "EvalResult",
    "TaskInput",
    "TASK_REGISTRY",
    "AssayGenerationTask",
    "CandidateRankingTask",
    "LiteratureSearchTask",
    "GeneAnnotationTask",
    "MoleculePropertyTask",
]
