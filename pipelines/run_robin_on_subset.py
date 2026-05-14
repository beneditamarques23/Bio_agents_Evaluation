"""
Run Robin on the fixed Eval1 subset.

This script is used to test the Robin framework on the same Eval1 subset used
for Biomni, Phylo, and Edison.

It is intentionally implemented as a Python script instead of a notebook because
Robin uses asyncio internally, and calling it inside Jupyter can raise:

    asyncio.run() cannot be called from a running event loop

Input:
    data_subset/project_subset.csv

Output:
    data_subset/output_robin.csv

Current issue:
    Robin does not run successfully in the current local setup.
    Even with lite_mode=True, execution may fail due to authentication or timeout issues depending on the installed Robin/FutureHouse/Edison dependencies.
"""

from pathlib import Path
import json
import time
import traceback
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

framework = "robin"
model = "qwen3"

df = pd.read_csv(input_csv)

runner = FRAMEWORK_REGISTRY[framework]()
results = []

print(f"Running {framework} with model={model}")
print("Input:", input_csv)
print("Output:", output_csv)
print("Rows:", len(df))

for idx, row in df.iterrows():
    print(f"\n========== Query {idx + 1}/{len(df)} ==========")
    print("Task:", row["task_name"])
    print("Instance:", row["task_instance_id"])

    task = BiomniEval1Task(
        task_name=row["task_name"],
        task_instance_id=row["task_instance_id"],
        prompt=row["input_query"],
    )

    task_input = task.get_input()
    tools = task.get_tools()

    start = time.perf_counter()

    try:
        agent_result = runner.run(
            task_input.prompt,
            tools,
            model,
            commercial_mode=False,
            timeout_seconds=600,
            lite_mode=True,
            num_assays=1,
            num_candidates=1,
            num_queries=1,
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

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\nSaved to:", output_csv)
print("Rows:", len(results_df))