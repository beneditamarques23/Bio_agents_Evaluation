"""
Run Robin RAG on the fixed Eval1 subset.

Uses the RobinRAGRunner: PubMed-grounded Q&A that answers each Biomni Eval
question by (1) extracting a search query, (2) fetching PubMed abstracts, and
(3) synthesising an answer with literature context.

This is intentionally a plain Python script (not a notebook) because LiteLLM
uses asyncio internally — calling asyncio.run() inside Jupyter raises:

    asyncio.run() cannot be called from a running event loop

Input:
    data_subset/project_subset.csv

Output:
    data_subset/output_robin.csv
"""

import asyncio
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from bio_agents.frameworks import FRAMEWORK_REGISTRY
from bio_agents.tasks.biomni_eval1.task import BiomniEval1Task


def safe_json(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


input_csv = Path("data_subset/project_subset.csv")
output_csv = Path("data_subset/output_robin.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)
if output_csv.exists():
    output_csv.unlink()  # Remove stale results to avoid appending to a previous run

framework = "robin-rag"
# RobinRAGRunner calls the LLM twice per question (query extraction + answer
# synthesis) and does a PubMed search in between.  Any model in the registry
# works; free-tier Ollama cloud models are a good default:
#   ministral-3b-cloud, ministral-8b-cloud, ministral-14b-cloud  (free)
#   gemma34b-cloud, gemma312b-cloud                              (free)
model = "gemma312b-cloud"

df = pd.read_csv(input_csv)

print(f"Running {framework} with model={model}")
print("Input:", input_csv)
print("Output:", output_csv)
print("Rows:", len(df))


async def main() -> None:
    # A single event loop is shared across all rows so that LiteLLM's internal
    # asyncio LoggingWorker queue stays bound to the correct loop. Calling
    # asyncio.run() once per row creates a new loop each time and causes:
    #   RuntimeError: <Queue> is bound to a different event loop
    runner = FRAMEWORK_REGISTRY[framework]()
    results = []

    for count, (idx, row) in enumerate(df.iterrows(), start=1):
        print(f"\n========== Query {count}/{len(df)} ==========")
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
            # Call _run_async directly so we stay in the same event loop.
            # runner.run() wraps this in asyncio.run(), which would create a
            # new loop on every iteration and break LiteLLM's logging queue.
            agent_result = await runner._run_async(  # type: ignore[attr-defined]
                task_input.prompt,
                model,
                max_results=5,
            )

            duration_s = round(time.perf_counter() - start, 2)
            output_text = agent_result.output
            tool_calls = agent_result.tool_calls
            metadata = agent_result.metadata
            run_error = ""

            print("Output:", output_text)

        except Exception:
            duration_s = round(time.perf_counter() - start, 2)
            output_text = ""
            tool_calls = []
            metadata = {}
            run_error = traceback.format_exc()

            print("ERROR:")
            print(run_error)

        results.append(
            {
                "sample_index": row["sample_index"],
                "framework": framework,
                "model": model,
                "task_name": row["task_name"],
                "task_instance_id": row["task_instance_id"],
                "input_query": row["input_query"],
                "dataset_eval1_answer": row["dataset_eval1_answer"],
                "output": output_text,
                "duration_s": duration_s,
                "tool_calls_count": len(tool_calls),
                "tool_calls": safe_json(tool_calls),
                "metadata": safe_json(metadata),
                "run_error": run_error,
            }
        )
        # Incremental save — each row is flushed to disk immediately so a mid-run
        # crash or kill does not lose already-completed results.
        pd.DataFrame([results[-1]]).to_csv(
            output_csv,
            mode="a",
            header=not output_csv.exists(),
            index=False,
            encoding="utf-8-sig",
        )

    print("\nSaved to:", output_csv)
    print("Rows:", len(results))


asyncio.run(main())
