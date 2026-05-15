"""
Robin RAG runner — PubMed-grounded Q&A for Biomni Eval tasks.

Uses Robin's PubMed infrastructure as a Retrieval-Augmented Generation (RAG)
layer to answer Biomni Eval biology questions with literature context.

Pipeline (3 steps, all run per call):
  1. query_extraction  — LLM extracts a clean PubMed search query from the question
  2. pubmed_search     — fetch abstracts via NCBI E-utilities (no API key needed)
  3. answer_synthesis  — LLM answers the original question using literature context
                         (falls back to LLM-only if PubMed returns no results)

Input  (prompt): Biomni Eval question text
Output (AgentResult):
  - output:      LLM-generated answer to the question
  - tool_calls:  [{name, input, output}, …] for each step
  - metadata:    model, pubmed_pmids, search_query, has_context

Requires:
  - An API key for the selected model provider (set in .env)
  - robin optional group installed: uv sync --extra robin
    (pulls in LiteLLM which this runner uses directly)
"""

from __future__ import annotations

import asyncio
from typing import Any

from bio_agents.frameworks.base import AgentResult, AgentRunner


class RobinRAGRunner(AgentRunner):
    """PubMed-grounded Q&A runner — answers Biomni Eval questions with literature
    context."""

    @property
    def framework_name(self) -> str:
        return "robin-rag"

    def run(self, prompt: str, tools: list, model: str, **kwargs: Any) -> AgentResult:
        """
        Execute the RAG pipeline synchronously.

        Args:
            prompt: Biomni Eval question text.
            tools:  Ignored — runner manages its own tools.
            model:  Registry key for any supported provider — e.g. "gemma34b-cloud",
                    "claude-sonnet-4-6", "llama-3.3-70b", "gemini-2.0-flash".
            kwargs:
                max_results (int, default 5) — PubMed abstracts to retrieve.
        """
        return asyncio.run(self._run_async(prompt, model, **kwargs))

    async def _run_async(self, prompt: str, model: str, **kwargs: Any) -> AgentResult:
        from bio_agents.frameworks.robin.runner import _build_llm_config, _sync_env_vars
        from bio_agents.models.registry import REGISTRY, get_litellm_id
        from bio_agents.tools.pubmed import fetch_pubmed_abstracts

        _sync_env_vars()  # push settings → os.environ so LiteLLM picks up API keys

        litellm_model = get_litellm_id(model) if model in REGISTRY else model
        provider = REGISTRY[model]["provider"] if model in REGISTRY else "openai"

        # Re-use _build_llm_config so api_base / api_key wiring is consistent
        # with the main RobinRunner; then extract the params dict for direct calls.
        llm_cfg = _build_llm_config(litellm_model, provider)
        # litellm_params contains model + api_key/api_base/timeout — drop "model"
        # because litellm.acompletion takes it as a positional kwarg.
        call_params: dict[str, Any] = {
            k: v
            for k, v in llm_cfg["model_list"][0]["litellm_params"].items()
            if k != "model"
        }

        max_results: int = kwargs.get("max_results", 5)
        tool_calls: list[dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Step 1 — extract a compact PubMed search query from the question
        # ------------------------------------------------------------------
        search_query, query_error = await _extract_search_query(
            prompt, litellm_model, call_params
        )
        tool_calls.append(
            {
                "name": "query_extraction",
                "input": prompt,
                "output": search_query,
                **({"error": query_error} if query_error else {}),
            }
        )

        # ------------------------------------------------------------------
        # Step 2 — PubMed search
        # ------------------------------------------------------------------
        abstracts, pmids = await fetch_pubmed_abstracts(
            search_query, max_results=max_results
        )
        tool_calls.append(
            {
                "name": "pubmed_search",
                "input": search_query,
                "output": (
                    f"Found {len(pmids)} abstract(s): {', '.join(pmids)}"
                    if pmids
                    else "No results found."
                ),
            }
        )

        # ------------------------------------------------------------------
        # Step 3 — answer synthesis (with or without context)
        # ------------------------------------------------------------------
        answer, answer_error = await _synthesize_answer(
            prompt, abstracts, litellm_model, call_params
        )
        tool_calls.append(
            {
                "name": "answer_synthesis",
                "input": prompt,
                "output": answer,
                **({"error": answer_error} if answer_error else {}),
            }
        )

        return AgentResult(
            output=answer,
            tool_calls=tool_calls,
            metadata={
                "model": model,
                "search_query": search_query,
                "pubmed_pmids": pmids,
                "has_context": bool(abstracts),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _extract_search_query(
    question: str,
    model: str,
    call_params: dict[str, Any],
) -> tuple[str, str]:
    """
    Ask the LLM to distil the question into a compact PubMed keyword query.

    Returns:
        (search_query, error_repr) — error_repr is "" on success.
        On failure falls back to the first 200 characters of the question.
    """
    import litellm  # type: ignore[import]

    system = (
        "You are a biomedical literature search expert. "
        "Given a biology question, produce a concise PubMed keyword query "
        "(3–8 terms) that will retrieve the most relevant papers. "
        "Return ONLY the search query — no explanation, no punctuation other "
        "than the query itself."
    )
    user = f"Question:\n{question}"

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=100,
            **call_params,
        )
        raw = (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]
        return (raw if raw else question[:200]), ""
    except Exception as exc:
        return question[:200], repr(exc)


async def _synthesize_answer(
    question: str,
    abstracts: str,
    model: str,
    call_params: dict[str, Any],
) -> tuple[str, str]:
    """
    Ask the LLM to answer the question, optionally grounded in PubMed abstracts.

    Returns:
        (answer, error_repr) — error_repr is "" on success.
    """
    import litellm  # type: ignore[import]

    if abstracts:
        system = (
            "You are a biomedical expert. Answer the question below using the "
            "provided PubMed abstracts as your primary evidence. "
            "Be concise and precise. "
            "If the expected answer is a single option letter (A/B/C/D/E) or a "
            "gene/protein name, state it clearly at the very start of your response."
        )
        user = f"PubMed abstracts:\n{abstracts}\n\nQuestion:\n{question}"
    else:
        # No PubMed results — fall back to LLM knowledge
        system = (
            "You are a biomedical expert. Answer the question below based on your "
            "knowledge. Be concise and precise. "
            "If the expected answer is a single option letter (A/B/C/D/E) or a "
            "gene/protein name, state it clearly at the very start of your response."
        )
        user = f"Question:\n{question}"

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            **call_params,
        )
        answer = (response.choices[0].message.content or "").strip()  # type: ignore[union-attr]
        return answer, ""
    except Exception as exc:
        return "", repr(exc)
