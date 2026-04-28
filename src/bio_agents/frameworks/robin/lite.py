"""
Robin lite mode — FutureHouse-free drop-in replacement for call_platform().

Architecture
------------
Robin's pipeline calls `call_platform()` in four places:

  experimental_assay()       step 2 — literature search for assay ideas
  experimental_assay()       step 4 — deep hypothesis report per assay
  therapeutic_candidates()   step 2 — literature search for candidate ideas
  therapeutic_candidates()   step 4 — deep hypothesis report per candidate

In standard mode all four hit the FutureHouse API (Crow / Falcon jobs).
In lite mode we replace them with:

  Literature search calls  → PubMed (NCBI E-utilities) + local LLM summary
  Hypothesis report calls  → local LLM only (no external search)

Detection heuristic
-------------------
Robin builds the `queries` dict differently for each call type:

  Literature search  →  {query_text: query_text}   (keys == values)
  Hypothesis report  →  {short_name: long_prompt}  (keys != values)

We use `all(k == v for k, v in queries.items())` to tell them apart.

Usage
-----
Called by RobinRunner._run_async() when lite_mode=True:

    lite_fn = make_lite_call_platform(config.llm_client)
    robin.assays.call_platform     = lite_fn
    robin.candidates.call_platform = lite_fn
    try:
        ...pipeline...
    finally:
        robin.assays.call_platform     = original_fn
        robin.candidates.call_platform = original_fn
"""

from __future__ import annotations

import uuid
from typing import Any

from bio_agents.tools.pubmed import fetch_pubmed_abstracts


def make_lite_call_platform(llm_client: Any) -> Any:
    """
    Return an async drop-in for robin's call_platform() that uses
    PubMed + local LLM instead of the FutureHouse API.

    Args:
        llm_client: A robin LiteLLMModel instance (config.llm_client).
                    Captured in the closure so the returned function matches
                    call_platform's (queries, fh_client, job_name) signature.
    """

    async def _lite_call_platform(
        queries: dict[str, str],
        fh_client: Any,  # ignored — FutureHouse client not used in lite mode
        job_name: Any,  # ignored — job routing handled locally
    ) -> dict[str, Any]:
        # Detect call type by checking whether all keys equal their values.
        # Robin uses {q: q for q in query_list} for literature searches and
        # {short_name: long_prompt} for hypothesis / report generation calls.
        is_lit_search = all(k == v for k, v in queries.items())

        results: list[dict[str, Any]] = []
        for hypothesis, query in queries.items():
            if is_lit_search:
                answer, sources = await _pubmed_answer(query, llm_client)
            else:
                answer, sources = await _llm_report(query, llm_client)

            results.append(
                {
                    "hypothesis": hypothesis,
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                    "context": f"Query: {query}\nAnswer: {answer}",
                    "status": "success",
                    "task_run_id": str(uuid.uuid4()),
                }
            )

        return {"results": results, "count": len(results), "has_errors": False}

    return _lite_call_platform


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _pubmed_answer(query: str, llm_client: Any) -> tuple[str, str]:
    """Search PubMed for *query* and have the local LLM synthesise an answer."""
    from aviary.core import Message  # type: ignore[import]  # bundled with robin

    abstract_text, pmids = await fetch_pubmed_abstracts(query, max_results=5)

    if abstract_text:
        lit_context = abstract_text
        sources = "\n".join(f"PMID: {pmid}" for pmid in pmids)
    else:
        lit_context = "(No PubMed results found — answering from model knowledge.)"
        sources = "(No PubMed sources — generated from model knowledge.)"

    messages = [
        Message(
            role="system",
            content=(
                "You are a biomedical research scientist. "
                "Using the literature context below, provide a concise, "
                "evidence-based answer to the research question. "
                "Cite relevant findings and mechanisms."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Research question: {query}\n\n"
                f"Literature context:\n{lit_context}\n\n"
                "Synthesise the evidence into a comprehensive answer."
            ),
        ),
    ]
    response = await llm_client.call_single(messages)
    return str(response.text), sources


async def _llm_report(query: str, llm_client: Any) -> tuple[str, str]:
    """
    Generate a detailed research / hypothesis report using only the local LLM.

    Used for step-4 calls where Robin asks for a deep hypothesis report on
    a proposed assay or therapeutic candidate.
    """
    from aviary.core import Message  # type: ignore[import]  # bundled with robin

    messages = [
        Message(
            role="system",
            content=(
                "You are an expert biomedical researcher. "
                "Generate a detailed, structured research report based on your "
                "scientific knowledge. Be specific, mechanistic, and rigorous. "
                "Include relevant biological mechanisms, supporting evidence, "
                "potential limitations, and next experimental steps."
            ),
        ),
        Message(role="user", content=query),
    ]
    response = await llm_client.call_single(messages)
    source_note = "(Generated by local LLM — no external literature search)"
    return str(response.text), source_note
