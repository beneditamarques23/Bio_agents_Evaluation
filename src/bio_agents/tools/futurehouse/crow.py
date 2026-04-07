"""
FutureHouse Crow — scientific literature search agent.

Crow answers natural-language questions by searching and synthesising
PubMed / bioRxiv papers via PaperQA2.  It can be used standalone as a
tool by any framework adapter, or internally by the Robin runner.

Requires:
    - FUTUREHOUSE_API_KEY in .env
    - robin optional dependency group: uv sync --extra robin
      (which pulls in futurehouse_client)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import NoReturn

from bio_agents.config import settings


@dataclass
class CrowResult:
    query: str
    answer: str
    formatted_answer: str = ""
    references: list[str] = field(default_factory=list)
    status: str = "success"
    trajectory_url: str = ""


async def crow_search(query: str, *, deep: bool = False) -> CrowResult:
    """
    Submit a query to the FutureHouse Crow literature-search agent.

    Args:
        query: Natural-language research question.
        deep:  Use the deeper (slower, more thorough) search variant.

    Returns:
        CrowResult with synthesised answer and metadata.

    Raises:
        RuntimeError:  FUTUREHOUSE_API_KEY is missing.
        ImportError:   futurehouse_client is not installed.
    """
    _require_api_key()
    client = _get_client()

    job = "job-futurehouse-paperqa2-deep" if deep else "job-futurehouse-paperqa2"

    try:
        from futurehouse_client import TaskRequest  # type: ignore[import]
    except ImportError as exc:
        _missing_client(exc)

    task = await asyncio.to_thread(
        client.run_task_until_done,  # type: ignore[attr-defined]
        TaskRequest(name=job, query=query),
    )

    return CrowResult(
        query=query,
        answer=getattr(task, "answer", "") or "",
        formatted_answer=getattr(task, "formatted_answer", "") or "",
        status=getattr(task, "status", "unknown"),
        trajectory_url=getattr(task, "trajectory_url", "") or "",
    )


def crow_search_sync(query: str, *, deep: bool = False) -> CrowResult:
    """Synchronous wrapper for use in non-async framework adapters."""
    return asyncio.run(crow_search(query, deep=deep))


# ---------------------------------------------------------------------------
# Helpers shared with falcon.py
# ---------------------------------------------------------------------------


def _require_api_key() -> None:
    if not settings.futurehouse_api_key:
        raise RuntimeError(
            "FUTUREHOUSE_API_KEY is not set. "
            "Get a key at https://platform.futurehouse.org and add it to .env."
        )


def _get_client():  # type: ignore[return]
    try:
        from futurehouse_client import FutureHouseClient  # type: ignore[import]
    except ImportError as exc:
        _missing_client(exc)
    return FutureHouseClient(
        api_key=settings.futurehouse_api_key,
        service_uri=settings.futurehouse_api_url,
    )


def _missing_client(exc: Exception) -> NoReturn:
    raise ImportError(
        "futurehouse_client is not installed. " "Run: uv sync --extra robin"
    ) from exc
