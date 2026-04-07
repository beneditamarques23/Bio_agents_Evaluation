"""
Robin framework adapter.

Wraps the Future-House Robin multi-agent drug-discovery pipeline
(https://github.com/Future-House/robin) as a bio-agents AgentRunner.

Pipeline (two stages, both run per call):
  1. experimental_assay  — literature search + assay proposal + pairwise ranking
  2. therapeutic_candidates — candidate generation + Falcon reports + ranking

Input  (prompt): disease name  e.g. "dry age-related macular degeneration"
Output (AgentResult):
  - output:      candidate_generation_goal string (top assay description)
  - tool_calls:  [{name, input, output}, …] for each Robin stage
  - metadata:    output_dir, disease_name, model used

Requires:
  - FUTUREHOUSE_API_KEY in .env
  - robin optional group installed: uv sync --extra robin
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bio_agents.config import settings
from bio_agents.frameworks.base import AgentResult, AgentRunner


class RobinRunner(AgentRunner):
    """Runs the full Robin pipeline for a given disease name."""

    @property
    def framework_name(self) -> str:
        return "robin"

    def run(self, prompt: str, tools: list, model: str, **kwargs: Any) -> AgentResult:
        """
        Execute the Robin pipeline synchronously.

        Args:
            prompt:  Disease name (e.g. "dry age-related macular degeneration").
            tools:   Ignored — Robin manages its own tools internally.
            model:   Registry key (e.g. "llama3", "claude-sonnet-4-6", "o4-mini").
            kwargs:
                num_assays     (int, default 3)  — assays to generate and rank.
                num_candidates (int, default 5)  — candidates to generate and rank.
                num_queries    (int, default 5)  — literature search queries per stage.
        """
        return asyncio.run(self._run_async(prompt, model, **kwargs))

    async def _run_async(
        self, disease_name: str, model: str, **kwargs: Any
    ) -> AgentResult:
        _require_dependencies()
        _sync_env_vars()  # Robin uses os.getenv() directly

        from futurehouse_client import JobNames  # type: ignore[import]
        from robin import (  # type: ignore[import]
            RobinConfiguration,
            experimental_assay,
            therapeutic_candidates,
        )
        from robin.configuration import AgentConfig  # type: ignore[import]

        from bio_agents.models.registry import REGISTRY, get_litellm_id

        litellm_model = get_litellm_id(model) if model in REGISTRY else model
        provider = REGISTRY[model]["provider"] if model in REGISTRY else "openai"
        llm_config = _build_llm_config(litellm_model, provider)

        output_dir = Path("results") / "robin" / disease_name.lower().replace(" ", "_")

        from futurehouse_client import FutureHouseClient  # type: ignore[import]

        fh_client = FutureHouseClient(
            api_key=settings.futurehouse_api_key,
            service_uri=settings.futurehouse_api_url,
        )

        # Route all FutureHouse agent calls through Phoenix (the only job this
        # API key has access to). CROW and FALCON require a higher-tier key.
        phoenix_config = AgentConfig(
            assay_lit_search_agent=JobNames.PHOENIX,
            assay_hypothesis_report_agent=JobNames.PHOENIX,
            candidate_lit_search_agent=JobNames.PHOENIX,
            candidate_hypothesis_report_agent=JobNames.PHOENIX,
        )

        config = RobinConfiguration(
            disease_name=disease_name,
            llm_name=litellm_model,
            llm_config=llm_config,
            futurehouse_api_key=settings.futurehouse_api_key,
            run_folder_name=str(output_dir),
            num_assays=kwargs.get("num_assays", 3),
            num_candidates=kwargs.get("num_candidates", 5),
            num_queries=kwargs.get("num_queries", 5),
            agent_settings=phoenix_config,
        )
        # Inject our pre-built client so Robin uses the correct API URL
        config._fh_client = fh_client

        tool_calls: list[dict[str, Any]] = []

        # Stage 1 — assay generation & ranking
        candidate_goal: str | None = await experimental_assay(config)
        tool_calls.append(
            {
                "name": "experimental_assay",
                "input": disease_name,
                "output": candidate_goal,
            }
        )

        # Stage 2 — therapeutic candidate generation & ranking
        goal_str: str = candidate_goal or ""
        await therapeutic_candidates(goal_str, config)
        tool_calls.append(
            {
                "name": "therapeutic_candidates",
                "input": goal_str,
                "output": str(output_dir / "ranked_therapeutic_candidates.csv"),
            }
        )

        return AgentResult(
            output=goal_str,
            tool_calls=tool_calls,
            metadata={
                "disease_name": disease_name,
                "output_dir": str(output_dir),
                "model": model,
            },
        )


def _sync_env_vars() -> None:
    """
    Robin and its dependencies call os.getenv() directly rather than using
    our pydantic-settings object. Push our loaded settings into os.environ
    so those calls resolve correctly.
    """
    import os

    pairs = {
        "FUTUREHOUSE_API_KEY": settings.futurehouse_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
        "GOOGLE_API_KEY": settings.google_api_key,
    }
    for key, value in pairs.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def _build_llm_config(litellm_model: str, provider: str) -> dict:
    """
    Build the LiteLLM router config Robin expects.

    Robin uses LiteLLMModel(name=llm_name, config=llm_config) internally.
    The router requires the model to appear in model_list with matching
    litellm_params — including api_base for local providers like Ollama.
    """
    litellm_params: dict[str, Any] = {"model": litellm_model, "timeout": 300}

    if provider == "local":
        litellm_params["api_base"] = settings.ollama_base_url
    elif provider == "anthropic":
        litellm_params["api_key"] = settings.anthropic_api_key
    elif provider == "openai":
        litellm_params["api_key"] = settings.openai_api_key
    elif provider == "gemini":
        litellm_params["api_key"] = settings.google_api_key

    return {
        "model_list": [
            {
                "model_name": litellm_model,
                "litellm_params": litellm_params,
            }
        ]
    }


def _require_dependencies() -> None:
    if not settings.futurehouse_api_key:
        raise RuntimeError(
            "FUTUREHOUSE_API_KEY is not set. "
            "Get a key at https://platform.futurehouse.org and add it to .env."
        )
    try:
        import robin  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("robin is not installed. Run: uv sync --extra robin") from exc
