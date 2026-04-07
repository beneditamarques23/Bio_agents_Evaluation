"""
FutureHouse Falcon — deep scientific report agent.

Falcon produces long-form, citation-backed hypothesis reports by running
a thorough multi-step literature review (PaperQA2-deep).  Used internally
by Robin for therapeutic candidate generation, and available as a
standalone tool for other framework adapters.

Requires:
    - FUTUREHOUSE_API_KEY in .env
    - robin optional dependency group: uv sync --extra robin
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from bio_agents.tools.futurehouse.crow import (
    _get_client,
    _missing_client,
    _require_api_key,
)


@dataclass
class FalconResult:
    query: str
    report: str
    references: list[str] = field(default_factory=list)
    status: str = "success"
    trajectory_url: str = ""


async def falcon_report(query: str) -> FalconResult:
    """
    Generate a detailed scientific report using the FutureHouse Falcon agent.

    Args:
        query: Research question or hypothesis to investigate in depth.

    Returns:
        FalconResult with a long-form, citation-backed report.

    Raises:
        RuntimeError:  FUTUREHOUSE_API_KEY is missing.
        ImportError:   futurehouse_client is not installed.
    """
    _require_api_key()
    client = _get_client()

    try:
        from futurehouse_client import TaskRequest  # type: ignore[import]
    except ImportError as exc:
        _missing_client(exc)

    task = await asyncio.to_thread(
        client.run_task_until_done,  # type: ignore[attr-defined]
        TaskRequest(name="job-futurehouse-paperqa2-deep", query=query),
    )

    return FalconResult(
        query=query,
        report=getattr(task, "formatted_answer", "")
        or getattr(task, "answer", "")
        or "",
        status=getattr(task, "status", "unknown"),
        trajectory_url=getattr(task, "trajectory_url", "") or "",
    )


def falcon_report_sync(query: str) -> FalconResult:
    """Synchronous wrapper for use in non-async framework adapters."""
    return asyncio.run(falcon_report(query))
