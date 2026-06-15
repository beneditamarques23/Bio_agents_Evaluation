"""
Run Biomni on the fixed Eval1 subset using qwen3.

Input:
    data_subset/project_subset.csv

Output:
    data_subset/output_biomni_qwen3.csv

This version uses the full Biomni runner with tools enabled.
It separates:
    - output_reasoning: full raw Biomni response
    - output: cleaned final answer for automatic evaluation

Append-only resume mode:
    - Never deletes or rewrites the existing CSV.
    - Reads existing rows only to identify completed sample_index values.
    - Runs only queries whose output is missing/empty.
    - Appends each new result immediately after each query finishes.
"""

from __future__ import annotations

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
            (
                rf"(?:answer|correct answer|option|method)"
                rf"\s*(?:is|:)?\s*\**([{valid_letters}])\**\b"
            ),
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
    Read existing output CSV and return sample_index values that already have
    any row in the CSV.

    This function never modifies the CSV.

    A query is considered completed if it has any row in the CSV, regardless
    of whether output is correct, empty, invalid, or incomplete.
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
output_csv = Path("data_subset/output_biomni_qwen3.csv")

framework = "biomni"
model = "qwen3"

commercial_mode = False
timeout_seconds = 600

output_csv.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_csv)
df["sample_index"] = df["sample_index"].astype(str)

completed_sample_indices = get_completed_sample_indices(output_csv)

remaining_df = df[~df["sample_index"].isin(list(completed_sample_indices))].copy()

print(f"Running {framework} with model={model}")
print("Input:", input_csv)
print("Output:", output_csv)
print(f"Total queries in subset: {len(df)}")
print(f"Already completed with non-empty output: {len(completed_sample_indices)}")
print(f"Remaining to run: {len(remaining_df)}")

if remaining_df.empty:
    print("\nNo remaining queries to run.")
    print("Saved to:", output_csv)
    raise SystemExit(0)


runner = FRAMEWORK_REGISTRY[framework]()


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

    # Full Biomni mode
    tools = task.get_tools()

    start = time.perf_counter()

    try:
        agent_result = runner.run(
            task_input.prompt,
            tools,
            model,
            commercial_mode=commercial_mode,
            timeout_seconds=timeout_seconds,
        )

        duration_s = round(time.perf_counter() - start, 2)

        raw_output = str(agent_result.output or "")
        output_reasoning = raw_output
        output_text = extract_final_answer(
            raw_output=raw_output,
            task_name=str(row["task_name"]),
            input_query=str(row["input_query"]),
        )

        tool_calls = agent_result.tool_calls
        metadata = agent_result.metadata
        run_error = ""

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
