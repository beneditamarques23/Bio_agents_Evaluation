"""
PubMed literature search via NCBI E-utilities.

No API key required for basic usage (rate-limited to 3 req/s).
Set NCBI_API_KEY in .env to raise the limit to 10 req/s.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_USER_AGENT = (
    "bio-agents-playground/0.1 (https://github.com/LokaHQ/bio-agents-playground)"
)


async def fetch_pubmed_abstracts(
    query: str, max_results: int = 5
) -> tuple[str, list[str]]:
    """
    Search PubMed for *query* and fetch the abstract text.

    Returns:
        (abstract_text, pmids) — raw efetch text and the list of PMIDs found.
        Returns ("", []) on any network or parsing failure so callers can
        gracefully fall back to LLM-only knowledge.

    Args:
        query:       Natural-language or keyword search query.
        max_results: Maximum number of abstracts to retrieve (default 5).
    """
    try:
        return await asyncio.to_thread(_fetch_sync, query, max_results)
    except Exception as exc:
        logger.warning("PubMed search failed for %r: %s", query, exc)
        return "", []


# ---------------------------------------------------------------------------
# Sync implementation (run in a thread via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _fetch_sync(query: str, max_results: int) -> tuple[str, list[str]]:
    ncbi_key = os.environ.get("NCBI_API_KEY", "")
    api_key_param = f"&api_key={ncbi_key}" if ncbi_key else ""

    # --- Step 1: esearch — retrieve PMIDs --------------------------------
    search_url = (
        f"{_NCBI_BASE}/esearch.fcgi"
        f"?db=pubmed"
        f"&term={urllib.parse.quote(query)}"
        f"&retmax={max_results}"
        f"&retmode=json"
        f"&sort=relevance"
        f"{api_key_param}"
    )
    req = urllib.request.Request(search_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return "", []

    # --- Step 2: efetch — retrieve abstracts as plain text ---------------
    fetch_url = (
        f"{_NCBI_BASE}/efetch.fcgi"
        f"?db=pubmed"
        f"&id={','.join(pmids)}"
        f"&rettype=abstract"
        f"&retmode=text"
        f"{api_key_param}"
    )
    req = urllib.request.Request(fetch_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    return text, pmids
