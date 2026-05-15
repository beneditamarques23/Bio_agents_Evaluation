from bio_agents.frameworks.base import AgentResult, AgentRunner
from bio_agents.frameworks.biomni.runner import BiomniRunner
from bio_agents.frameworks.robin.rag_runner import RobinRAGRunner
from bio_agents.frameworks.robin.runner import RobinRunner

# Registry maps framework name → AgentRunner subclass.
FRAMEWORK_REGISTRY: dict[str, type[AgentRunner]] = {
    "robin": RobinRunner,
    "robin-rag": RobinRAGRunner,
    "biomni": BiomniRunner,
}

__all__ = [
    "AgentRunner",
    "AgentResult",
    "FRAMEWORK_REGISTRY",
    "RobinRunner",
    "RobinRAGRunner",
    "BiomniRunner",
]
