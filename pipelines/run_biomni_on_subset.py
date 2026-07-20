"""
Run Biomni on the fixed Biomni Eval1 subset

This script evaluates the Biomni framework on the same fixed subset used for
the other biomedical agents.

Framework:
    biomni

Model:
    qwen3

Input:
    data_subset/project_subset.csv

Output:
    data_subset/output_biomni_qwen3.csv

-----------------
What Biomni does
-----------------

This script runs the Biomni framework through:

    FRAMEWORK_REGISTRY["biomni"]

For each Biomni Eval query, the script builds a BiomniEval1Task object and
passes the task prompt plus the task-specific tools to the Biomni runner.

In contrast to Robin-RAG, which is mainly a PubMed retrieval-and-answering
pipeline, Biomni is an agentic biomedical framework. It can receive a set of
tools associated with each benchmark task and attempt to solve the question by
reasoning over the prompt, selecting tools, executing tool calls when needed,
and producing a final answer.

The general workflow is:

    1. Task loading
       The script reads each row from data_subset/project_subset.csv and creates
       a BiomniEval1Task using the task name, task instance ID, and input query.

    2. Tool selection
       For each task, the script calls:

           task.get_tools()

       This returns the tools made available to Biomni for that specific query.

    3. Agent execution
       The Biomni runner receives the original prompt, the task-specific tools,
       the selected model, and the timeout configuration.

    4. Answer extraction
       Biomni may return a long response containing explanations, XML-like tags,
       tool-related text, or final-answer formatting. The script stores this
       complete response in output_reasoning and then applies a heuristic
       extractor to produce a shorter answer in output.

Biomni is therefore closer to a tool-using biomedical agent than to a simple
question-answering model. Because of this, failures can happen at different
levels: model generation, tool selection, tool execution, timeout, parsing, or
final-answer extraction.

-----------------
Why qwen3 is used
-----------------

The model is set to:

    qwen3

This uses the local/Ollama model configuration available in the project. In the
previous tests, qwen3 was selected mainly because it was more stable with the
Biomni agent loop than several other tested models.

Some cloud models produced integration errors, empty answers, internal server
errors, or message-format incompatibilities during Biomni runs. qwen3 may still
produce low-quality or incorrectly formatted answers, but it is useful for
obtaining a complete set of runs with fewer execution-level failures.

-----------------
Configuration
-----------------

commercial_mode = False

This means the script runs Biomni in academic mode and allows the use of all
datasets made available by the framework, including non-commercial resources.

timeout_seconds = 600

Each query is allowed up to 600 seconds. This is useful because Biomni can be
slow, especially when it performs tool retrieval, tool calls, or multi-step
agentic reasoning.

-----------------
Output columns
-----------------

The CSV contains both a raw output column and a cleaned output column.

    output_reasoning

        This stores the full raw text returned by Biomni.

        It is called output_reasoning for consistency with the other benchmark
        scripts, but it should not be interpreted as the model's private hidden
        chain-of-thought. It is simply the complete visible response returned
        by the framework before post-processing.

        This column is useful for qualitative analysis because it preserves the
        full generated answer, including explanations, tool-related text,
        formatting artefacts, XML-like tags, uncertainty, and any extra content
        produced by the agent.

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
        dictionary-like answers, or the last meaningful line.

        Because Biomni can sometimes return verbose or badly formatted text,
        output may still contain imperfect extra text. In that case,
        output_reasoning should be used to inspect what the model actually
        produced.

    tool_calls

        Stores tool call information returned by Biomni, if available.

    metadata

        Stores extra metadata returned by Biomni, if available.

    run_error

        Stores the full traceback if a query fails. If the query completes
        normally, this field is empty.

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

FINAL_COLUMNS = [
    "sample_index",
    "framework",
    "model",
    "task_name",
    "task_instance_id",
    "input_query",
    "dataset_eval1_answer",
    "output_reasoning",
    "output",
    "duration_s",
    "tool_calls_count",
    "tool_calls",
    "metadata",
    "run_error",
]


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


def read_csv_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader)


def validate_existing_csv_schema(output_csv: Path) -> None:
    if not output_csv.exists():
        return

    existing_header = read_csv_header(output_csv)

    if existing_header != FINAL_COLUMNS:
        raise ValueError(
            "\nExisting output CSV has an incompatible schema.\n"
            f"File: {output_csv}\n\n"
            f"Expected columns:\n{FINAL_COLUMNS}\n\n"
            f"Found columns:\n{existing_header}\n\n"
            "To avoid shifted/misaligned columns, the script stopped.\n"
            "Fix the existing CSV, rename it, delete it, or set "
            "overwrite_existing = True to start from zero."
        )


def get_completed_sample_indices(output_csv: Path) -> set[str]:
    # Return sample_index values that already exist in the output CSV.

    completed: set[str] = set()

    if not output_csv.exists():
        return completed

    with open(output_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "sample_index" not in reader.fieldnames:
            raise ValueError(
                f"Existing CSV does not have a sample_index column: {output_csv}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                sample_index = str(row.get("sample_index", "")).strip()
                if sample_index:
                    completed.add(sample_index)
            except Exception as exc:
                print(f"WARNING: Skipping malformed row {row_number}: {exc}")

    return completed


input_csv = Path("data_subset/project_subset.csv")
output_csv = Path("data_subset/output_biomni_qwen3.csv")

framework = "biomni"
model = "qwen3"

commercial_mode = False
timeout_seconds = 600
overwrite_existing = False

output_csv.parent.mkdir(parents=True, exist_ok=True)

if overwrite_existing and output_csv.exists():
    output_csv.unlink()

validate_existing_csv_schema(output_csv)

df = pd.read_csv(input_csv)
df["sample_index"] = df["sample_index"].astype(str)

completed_sample_indices = get_completed_sample_indices(output_csv)

remaining_df = df[~df["sample_index"].isin(list(completed_sample_indices))].copy()

print(f"Running {framework} with model={model}")
print("Input:", input_csv)
print("Output:", output_csv)
print(f"Total queries in subset: {len(df)}")
print(f"Already completed with any row in CSV: {len(completed_sample_indices)}")
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

    clean_row = {col: result_row.get(col, "") for col in FINAL_COLUMNS}

    pd.DataFrame([clean_row], columns=FINAL_COLUMNS).to_csv(
        output_csv,
        mode="a",
        header=not output_csv.exists(),
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    print("Saved this query to CSV.")


print("\nSaved to:", output_csv)
