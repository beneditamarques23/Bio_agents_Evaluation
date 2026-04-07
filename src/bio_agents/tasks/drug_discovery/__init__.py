"""Drug-discovery task definitions."""

from bio_agents.tasks.drug_discovery.assay_generation import AssayGenerationTask
from bio_agents.tasks.drug_discovery.candidate_ranking import CandidateRankingTask

__all__ = ["AssayGenerationTask", "CandidateRankingTask"]
