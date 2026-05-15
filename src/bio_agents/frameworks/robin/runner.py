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
  - FUTUREHOUSE_API_KEY in .env  (not required when lite_mode=True)
  - robin optional group installed: uv sync --extra robin

Lite mode (lite_mode=True)
  Replaces the two FutureHouse literature-search + hypothesis-report calls
  per stage with PubMed (NCBI E-utilities) + local LLM summarisation.
  No FUTUREHOUSE_API_KEY needed — the full reasoning pipeline still runs.
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
            model:   Registry key for any supported provider — e.g. "llama3" (local),
                     "ministral-8b-cloud" (Ollama Cloud), "claude-sonnet-4-6"
                     (Anthropic), "o4-mini" (OpenAI), "gemini-2.0-flash" (Google),
                     "llama-3.3-70b" (Groq).
            kwargs:
                num_assays     (int,  default 3)    — assays to generate and rank.
                num_candidates (int,  default 5)    — candidates to generate and rank.
                num_queries    (int,  default 5)    — queries per stage.
                lite_mode      (bool, default False) — skip FutureHouse API; use
                               PubMed + local LLM for literature search and
                               hypothesis reports. No FUTUREHOUSE_API_KEY needed.
        """
        return asyncio.run(self._run_async(prompt, model, **kwargs))

    async def _run_async(
        self, disease_name: str, model: str, **kwargs: Any
    ) -> AgentResult:
        lite_mode: bool = kwargs.get("lite_mode", False)

        _require_dependencies(lite_mode=lite_mode)
        _sync_env_vars()  # Robin uses os.getenv() directly

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

        config = RobinConfiguration(
            disease_name=disease_name,
            llm_name=litellm_model,
            llm_config=llm_config,
            # Pass a non-empty placeholder so RobinConfiguration accepts the field;
            # in lite mode the real key is never used (call_platform is patched).
            futurehouse_api_key=settings.futurehouse_api_key or "lite_mode",  # type: ignore[call-arg]
            run_folder_name=str(output_dir),
            num_assays=kwargs.get("num_assays", 3),
            num_candidates=kwargs.get("num_candidates", 5),
            num_queries=kwargs.get("num_queries", 5),
        )

        if lite_mode:
            # Inject a sentinel so RobinConfiguration.fh_client property never
            # tries to instantiate a real FutureHouseClient (which would fail
            # without a key). Our patched call_platform ignores fh_client anyway.
            config._fh_client = object()  # type: ignore[assignment]
        else:
            from futurehouse_client import (  # type: ignore[import]
                FutureHouseClient,
                JobNames,
            )

            fh_client = FutureHouseClient(
                api_key=settings.futurehouse_api_key,
                service_uri=settings.futurehouse_api_url,
            )
            config._fh_client = fh_client  # type: ignore[assignment]

            # Route all FutureHouse agent calls through Phoenix (the only job this
            # API key has access to). CROW and FALCON require a higher-tier key.
            phoenix_config = AgentConfig(
                assay_lit_search_agent=JobNames.PHOENIX,
                assay_hypothesis_report_agent=JobNames.PHOENIX,
                candidate_lit_search_agent=JobNames.PHOENIX,
                candidate_hypothesis_report_agent=JobNames.PHOENIX,
            )
            config.agent_settings = phoenix_config

        tool_calls: list[dict[str, Any]] = []

        if lite_mode:
            # Monkey-patch robin's call_platform in both pipeline modules with our
            # PubMed + local LLM replacement, then restore originals afterwards.
            import robin.assays as _robin_assays  # type: ignore[import]
            import robin.candidates as _robin_cands  # type: ignore[import]

            from bio_agents.frameworks.robin.lite import make_lite_call_platform

            _orig_assays = getattr(_robin_assays, "call_platform")
            _orig_cands = getattr(_robin_cands, "call_platform")
            lite_fn = make_lite_call_platform(config.llm_client)
            setattr(_robin_assays, "call_platform", lite_fn)
            setattr(_robin_cands, "call_platform", lite_fn)
            try:
                candidate_goal, tool_calls = await _run_pipeline(
                    disease_name,
                    output_dir,
                    config,
                    experimental_assay,
                    therapeutic_candidates,
                )
            finally:
                setattr(_robin_assays, "call_platform", _orig_assays)
                setattr(_robin_cands, "call_platform", _orig_cands)
        else:
            candidate_goal, tool_calls = await _run_pipeline(
                disease_name,
                output_dir,
                config,
                experimental_assay,
                therapeutic_candidates,
            )

        return AgentResult(
            output=candidate_goal,
            tool_calls=tool_calls,
            metadata={
                "disease_name": disease_name,
                "output_dir": str(output_dir),
                "model": model,
                "lite_mode": lite_mode,
            },
        )


async def _run_pipeline(
    disease_name: str,
    output_dir: Path,
    config: Any,
    experimental_assay: Any,
    therapeutic_candidates: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Run both Robin pipeline stages and return (goal_str, tool_calls)."""
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

    return goal_str, tool_calls


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
        "GROQ_API_KEY": settings.groq_api_key,
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

    Supported providers: anthropic · openai · gemini · groq · local · ollama_cloud
    """
    litellm_params: dict[str, Any] = {"model": litellm_model, "timeout": 300}

    if provider in ("local", "ollama_cloud"):
        # ollama_cloud models (:<size>-cloud tags) are routed to Ollama's remote
        # GPU infrastructure by the local Ollama daemon — same api_base as local.
        # Requires: `ollama serve` + `ollama signin`
        litellm_params["api_base"] = settings.ollama_base_url
    elif provider == "anthropic":
        litellm_params["api_key"] = settings.anthropic_api_key
    elif provider == "openai":
        litellm_params["api_key"] = settings.openai_api_key
    elif provider == "gemini":
        litellm_params["api_key"] = settings.google_api_key
    elif provider == "groq":
        litellm_params["api_key"] = settings.groq_api_key

    return {
        "model_list": [
            {
                "model_name": litellm_model,
                "litellm_params": litellm_params,
            }
        ]
    }


def _require_dependencies(lite_mode: bool = False) -> None:
    if not lite_mode and not settings.futurehouse_api_key:
        raise RuntimeError(
            "FUTUREHOUSE_API_KEY is not set. "
            "Get a key at https://platform.futurehouse.org and add it to .env, "
            "or pass lite_mode=True to run without it (uses PubMed + local LLM)."
        )
    try:
        import robin  # noqa: F401  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("robin is not installed. Run: uv sync --extra robin") from exc
