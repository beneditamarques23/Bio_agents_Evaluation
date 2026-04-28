"""
Biomni framework adapter.

Wraps the Stanford snap-stanford Biomni general-purpose bio agent
(https://github.com/snap-stanford/Biomni) as a bio-agents AgentRunner.

The agent uses a ReAct loop with access to 78 datasets, 45+ Python packages,
12 R packages, and 25+ CLI tools. It accepts any biomedical research prompt.

Input  (prompt): any biomedical research question
Output (AgentResult):
  - output:     final assistant message from the conversation log
  - tool_calls: intermediate execution steps extracted from the log
  - metadata:   model, log_length, data_path

Requires:
  - biomni optional group installed: uv sync --extra biomni (or: make setup-biomni)
  - At least one LLM API key in .env (or Ollama running locally)
"""

from __future__ import annotations

from typing import Any

from bio_agents.config import settings
from bio_agents.frameworks.base import AgentResult, AgentRunner


class BiomniRunner(AgentRunner):
    """Runs the Biomni A1 agent for any biomedical research prompt."""

    @property
    def framework_name(self) -> str:
        return "biomni"

    def run(self, prompt: str, tools: list, model: str, **kwargs: Any) -> AgentResult:
        """
        Execute the Biomni agent synchronously.

        Args:
            prompt:  Any biomedical research question.
            tools:   Ignored — Biomni manages its own tools internally.
            model:   Registry key (e.g. "llama3", "llama-3.3-70b", "claude-sonnet-4-6").
            kwargs:
                use_tool_retriever (bool, default True)  — smart tool selection.
                timeout_seconds    (int,  default 600)   — per-tool execution timeout.
                commercial_mode (bool, default False) — filter non-commercial datasets.
        """
        _require_dependencies()
        _sync_env_vars()

        from biomni.agent import A1  # type: ignore[import]

        from bio_agents.models.registry import REGISTRY

        info = REGISTRY.get(model, {})
        model_id = info.get("model_id", model)
        provider = info.get("provider", "openai")

        # langchain_openai (used internally by Biomni) appends /chat/completions to
        # base_url, so Ollama's OpenAI-compatible endpoint must include the /v1 prefix.
        # ollama_cloud models (:<size>-cloud tags) are routed to Ollama's remote GPU
        # infrastructure by the local Ollama daemon — same base_url as local.
        # Requires: `ollama serve` + `ollama signin`
        if provider in ("local", "ollama_cloud"):
            base_url: str | None = f"{settings.ollama_base_url.rstrip('/')}/v1"
        else:
            base_url = None

        agent = A1(
            path=settings.biomni_data_path,
            llm=model_id,
            base_url=base_url,
            api_key=_get_api_key(provider),
            use_tool_retriever=kwargs.get("use_tool_retriever", True),
            timeout_seconds=kwargs.get("timeout_seconds", 600),
            commercial_mode=kwargs.get("commercial_mode", False),
        )

        # go() returns (log, final_message_content)
        # log is a list of pretty-printed strings (one per agent step)
        # final_message_content is the last assistant message as a plain string
        log, final_output = agent.go(prompt)

        tool_calls = [
            {"name": f"step_{i}", "output": entry} for i, entry in enumerate(log)
        ]

        return AgentResult(
            output=final_output,
            tool_calls=tool_calls,
            metadata={
                "model": model,
                "log_length": len(log),
                "data_path": settings.biomni_data_path,
            },
        )


def _get_api_key(provider: str) -> str | None:
    key_map = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.google_api_key,
        "groq": settings.groq_api_key,
    }
    return key_map.get(provider) or None


def _sync_env_vars() -> None:
    """Push settings into os.environ so Biomni's internal os.getenv() calls resolve."""
    import os

    pairs = {
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
        "GOOGLE_API_KEY": settings.google_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
        "BIOMNI_DATA_PATH": settings.biomni_data_path,
    }
    for key, value in pairs.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def _require_dependencies() -> None:
    try:
        import biomni  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise ImportError("biomni is not installed. Run: make setup-biomni") from exc
