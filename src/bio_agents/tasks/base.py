from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from bio_agents.frameworks.base import AgentResult


@dataclass
class TaskInput:
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    score: float  # 0.0–1.0
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class BioTask(ABC):
    """Common interface every bio task must implement."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def get_input(self) -> TaskInput: ...

    @abstractmethod
    def get_tools(self) -> list: ...

    @abstractmethod
    def evaluate(self, result: AgentResult) -> EvalResult: ...
