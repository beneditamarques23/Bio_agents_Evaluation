from bio_agents.frameworks.base import AgentResult, AgentRunner
from bio_agents.frameworks.biomni.runner import BiomniRunner
from bio_agents.frameworks.robin.runner import RobinRunner

# Registry maps framework name → AgentRunner subclass.
FRAMEWORK_REGISTRY: dict[str, type[AgentRunner]] = {
    "robin": RobinRunner,
    "biomni": BiomniRunner,
}

__all__ = [
    "AgentRunner",
    "AgentResult",
    "FRAMEWORK_REGISTRY",
    "RobinRunner",
    "BiomniRunner",
]
