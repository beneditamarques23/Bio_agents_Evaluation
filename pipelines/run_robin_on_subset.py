"""
Run Robin-RAG on the fixed Biomni Eval1 subset using Gemma 3 12B Cloud.

This script evaluates the Robin-RAG framework on the same fixed subset used
for the other biomedical agents.

Framework:
    robin-rag

Model:
    gemma312b-cloud

Input:
    data_subset/project_subset.csv

Output:
    data_subset/output_robin_gemma312b_cloud.csv


What Robin-RAG does
-------------------

This script does NOT run the full FutureHouse Robin agent. It runs the local
Robin-RAG implementation available in this project through:

    FRAMEWORK_REGISTRY["robin-rag"]

The runner used internally is RobinRAGRunner.

Robin-RAG is a retrieval-augmented biomedical question-answering pipeline.
For each Biomni Eval query, the framework generally follows three steps:

    1. Query extraction
       The model receives the original benchmark question and converts it into
       a shorter PubMed-style search query.

    2. Literature retrieval
       The extracted query is used to retrieve PubMed abstracts. The number of
       retrieved abstracts is controlled by max_results.

    3. Answer synthesis
       The model receives the original question plus the retrieved literature
       context and produces a final answer.



Why this script is plain Python
-------------------------------

This is intentionally a normal Python script and not a Jupyter notebook.

LiteLLM and the Robin-RAG runner use asyncio internally. In Jupyter, there is
already an active event loop, so calling:

    asyncio.run(main())

inside a notebook can raise:

    asyncio.run() cannot be called from a running event loop

Running this file from PowerShell avoids that problem:

    uv run python pipelines/run_robin_on_subset.py


Output columns
--------------

The CSV contains both a raw output column and a cleaned output column.

    output_reasoning

        This stores the full raw text returned by Robin-RAG.

        It is called output_reasoning for consistency with the other benchmark
        scripts, but it should not be interpreted as the model's private hidden
        chain-of-thought. It is simply the complete visible response returned
        by the framework before post-processing.


    output

        This stores the cleaned final answer extracted from output_reasoning.

        It is the column used by the automatic scoring script. For example:

            - multiple-choice tasks should become A, B, C, D, or E;
            - CRISPR delivery tasks should become A-F;
            - gene retrieval tasks should become a gene symbol;
            - GWAS variant tasks should become an rsID;
            - rare disease diagnosis tasks should become the final predicted
              answer format.

        The extractor is heuristic. It tries to find explicit [ANSWER] tags,
        <solution> tags, multiple-choice letters, candidate genes, variants,
        or the last meaningful line.

    tool_calls

        Stores tool call information returned by the framework, if available.

    metadata

        Stores extra metadata returned by the framework, if available.

    run_error

        Stores the full traceback if a query fails. If the query completes
        normally, this field is empty.


Resume behaviour
----------------

This script is append-only by default.

That means:

    - it does not delete the existing CSV;
    - it does not rewrite previous rows;
    - it checks which sample_index values already exist in the CSV;
    - it only runs queries whose sample_index is not already present;
    - after each query finishes, the result is immediately appended to the CSV.

This makes it safer to interrupt the script with Ctrl+C or shut down the
computer, because completed rows remain saved.

If you really want to force a full rerun from zero, set:

    overwrite_existing = True

below.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from bio_agents.frameworks import FRAMEWORK_REGISTRY
from bio_agents.tasks.biomni_eval1.task import BiomniEval1Task


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def strip_solution_tags(text: str) -> str:
    match = re.search(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_candidates_from_query(input_query: str) -> list[str]:
    candidates: list[str] = []

    variant_match = re.search(r"Variants:\s*(.+)", input_query, flags=re.I | re.S)
    if variant_match:
        variants_text = variant_match.group(1)
        candidates.extend(re.findall(r"\brs\d+\b", variants_text))

    genes_braced = re.findall(r"\{([^{}]+)\}", input_query)
    candidates.extend([g.strip() for g in genes_braced if g.strip()])

    candidate_match = re.search(
        r"Candidate genes:\s*(.+?)(?:Output only|$)",
        input_query,
        flags=re.I | re.S,
    )
    if candidate_match:
        for item in candidate_match.group(1).split(","):
            gene = item.strip().strip("[]'\" ")
            if gene:
                candidates.append(gene)

    seen = set()
    clean_candidates = []

    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            clean_candidates.append(candidate)

    return clean_candidates


def extract_final_answer(raw_output: str, task_name: str, input_query: str) -> str:
    text = str(raw_output or "").strip()

    if not text:
        return ""

    answer_tag = re.findall(
        r"\[ANSWER\]\s*([A-Za-z0-9_.:/\-{}'\", ]+?)\s*\[/ANSWER\]",
        text,
        flags=re.I | re.S,
    )
    if answer_tag:
        return answer_tag[-1].strip()

    solution_text = strip_solution_tags(text)

    answer_tag_solution = re.findall(
        r"\[ANSWER\]\s*([A-Za-z0-9_.:/\-{}'\", ]+?)\s*\[/ANSWER\]",
        solution_text,
        flags=re.I | re.S,
    )
    if answer_tag_solution:
        return answer_tag_solution[-1].strip()

    if task_name in {"lab_bench_dbqa", "lab_bench_seqqa"}:
        valid_letters = "A-E"
    elif task_name == "crispr_delivery":
        valid_letters = "A-F"
    else:
        valid_letters = ""

    if valid_letters:
        patterns = [
            rf"\[ANSWER\]\s*([{valid_letters}])\s*\[/ANSWER\]",
            rf"(?:answer|correct answer|option|method)\s*(?:is|:)?\s*\**([{valid_letters}])\**\b",
            rf"\(([{valid_letters}])\)",
            rf"\*\*([{valid_letters}])\*\*",
            rf"\b([{valid_letters}])\b\s*$",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, solution_text, flags=re.I)
            if matches:
                return matches[-1].upper()

    if task_name == "rare_disease_diagnosis":
        dict_matches = re.findall(r"\{.*?\}", solution_text, flags=re.S)
        if dict_matches:
            return dict_matches[-1].strip()

    candidates = extract_candidates_from_query(input_query)

    if candidates:
        found = []

        for candidate in candidates:
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(candidate) + r"(?![A-Za-z0-9_])"

            for match in re.finditer(pattern, solution_text):
                found.append((match.start(), candidate))

        if found:
            found.sort()
            return found[-1][1].strip()

    lines = [line.strip() for line in solution_text.splitlines() if line.strip()]

    if lines:
        return lines[-1].strip()

    return solution_text.strip()


def get_completed_sample_indices(output_csv: Path) -> set[str]:
    """
    Return sample_index values that already exist in the output CSV.

    A query is considered completed if there is already any row for that
    sample_index, regardless of whether the output is correct, incorrect,
    empty, or difficult to score.

    This function never modifies the CSV.
    """
    completed: set[str] = set()

    if not output_csv.exists():
        return completed

    try:
        with open(output_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or "sample_index" not in reader.fieldnames:
                print("WARNING: Existing CSV does not have sample_index column.")
                return completed

            for row_number, row in enumerate(reader, start=2):
                try:
                    sample_index = str(row.get("sample_index", "")).strip()
                    if sample_index:
                        completed.add(sample_index)
                except Exception as exc:
                    print(f"WARNING: Skipping malformed row {row_number}: {exc}")

    except Exception as exc:
        print("WARNING: Could not read existing CSV safely.")
        print(exc)

    return completed


input_csv = Path("data_subset/project_subset.csv")
output_csv = Path("data_subset/output_robin_gemma312b_cloud.csv")

framework = "robin-rag"
model = "gemma312b-cloud"

max_results = 5
max_attempts = 3
overwrite_existing = False

output_csv.parent.mkdir(parents=True, exist_ok=True)

if overwrite_existing and output_csv.exists():
    output_csv.unlink()

df = pd.read_csv(input_csv)
df["sample_index"] = df["sample_index"].astype(str)

completed_sample_indices = get_completed_sample_indices(output_csv)

remaining_df = df[
    ~df["sample_index"].isin(completed_sample_indices)
].copy()

print(f"Running {framework} with model={model}")
print("Input:", input_csv)
print("Output:", output_csv)
print(f"Total rows in subset: {len(df)}")
print(f"Already completed with any row in CSV: {len(completed_sample_indices)}")
print(f"Remaining to run: {len(remaining_df)}")
print(f"PubMed max_results per query: {max_results}")
print(f"Max attempts per query if raw output is empty: {max_attempts}")


async def main() -> None:
    if remaining_df.empty:
        print("\nNo remaining queries to run.")
        print("Saved to:", output_csv)
        return

    runner = FRAMEWORK_REGISTRY[framework]()
    results = []

    for count, (_, row) in enumerate(remaining_df.iterrows(), start=1):
        print(f"\n========== Query {count}/{len(remaining_df)} ==========")
        print("Sample index:", row["sample_index"])
        print("Task:", row["task_name"])
        print("Instance:", row["task_instance_id"])

        task = BiomniEval1Task(
            task_name=str(row["task_name"]),
            task_instance_id=int(str(row["task_instance_id"])),
            prompt=str(row["input_query"]),
        )

        task_input = task.get_input()

        start = time.perf_counter()

        try:
            agent_result = None
            raw_output = ""

            for attempt in range(1, max_attempts + 1):
                print(f"Attempt {attempt}/{max_attempts}")

                agent_result = await runner._run_async(  # type: ignore[attr-defined]
                    task_input.prompt,
                    model,
                    max_results=max_results,
                )

                raw_output = str(agent_result.output or "").strip()

                if raw_output:
                    break

                print("WARNING: Empty raw Robin output. Retrying...")

            duration_s = round(time.perf_counter() - start, 2)

            if agent_result is None:
                raise RuntimeError("Robin-RAG did not return an agent_result.")

            output_reasoning = raw_output
            output_text = extract_final_answer(
                raw_output=raw_output,
                task_name=str(row["task_name"]),
                input_query=str(row["input_query"]),
            )

            tool_calls = agent_result.tool_calls
            metadata = agent_result.metadata
            run_error = ""

            if not output_text.strip():
                print("WARNING: Empty clean output")
                print("Raw output:", raw_output)
                print("Metadata:", safe_json(metadata))
                print("Tool calls:", safe_json(tool_calls))
            else:
                print("Clean output:", output_text)

        except Exception:
            duration_s = round(time.perf_counter() - start, 2)

            output_reasoning = ""
            output_text = ""
            tool_calls = []
            metadata = {}
            run_error = traceback.format_exc()

            print("ERROR:")
            print(run_error)

        result_row = {
            "sample_index": row["sample_index"],
            "framework": framework,
            "model": model,
            "task_name": row["task_name"],
            "task_instance_id": row["task_instance_id"],
            "input_query": row["input_query"],
            "dataset_eval1_answer": row["dataset_eval1_answer"],
            "output_reasoning": output_reasoning,
            "output": output_text,
            "duration_s": duration_s,
            "tool_calls_count": len(tool_calls),
            "tool_calls": safe_json(tool_calls),
            "metadata": safe_json(metadata),
            "run_error": run_error,
        }

        results.append(result_row)

        pd.DataFrame([result_row]).to_csv(
            output_csv,
            mode="a",
            header=not output_csv.exists(),
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
        )

        print("Saved this query to CSV.")

    print("\nSaved to:", output_csv)
    print("Rows added in this run:", len(results))


asyncio.run(main())