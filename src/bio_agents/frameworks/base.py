from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    output: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Populated by runners: tokens_used, cost_usd, latency_s, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRunner(ABC):
    """Common interface every framework adapter must implement."""

    @property
    @abstractmethod
    def framework_name(self) -> str: ...

    @abstractmethod
    def run(self, prompt: str, tools: list, model: str, **kwargs) -> AgentResult: ...
