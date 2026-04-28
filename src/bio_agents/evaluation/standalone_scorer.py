"""
StandaloneEval1Scorer — score external outputs against the Biomni-Eval1 benchmark.

Use this when you have agent outputs gathered from *outside* this framework
(e.g., from another tool, a manual run, or a third-party API) and want to
evaluate them with the same exact-match scoring used by Eval1Runner.

Each record needs only three fields:
    task_name         (str)  — one of the 10 Biomni-Eval1 task types
    task_instance_id  (int)  — row ID within that task type in biomni/Eval1
    output            (str)  — the agent response to score

Any extra fields (framework, model, run_id, cost, …) are passed through
untouched in the output records.

Scoring is binary: 0.0 (incorrect) or 1.0 (correct), via:
    BiomniEval1().evaluate(task_name, task_instance_id, output)

Requires: uv sync --extra biomni  (or: make setup-biomni)

Example::

    scorer = StandaloneEval1Scorer()

    # Score a single output
    result = scorer.score_one(
        task_name="gwas_causal_gene_opentargets",
        task_instance_id=767,
        output="The causal gene is HNF1A.",
        model="gpt-4o",          # extra fields are passed through
    )
    print(result["score"])   # 1.0
    print(result["passed"])  # True

    # Score a batch loaded from a file
    results = scorer.score_from_jsonl("path/to/outputs.jsonl")
    out_file = scorer.save(results, output_dir="results/my_scored_run")
    stats = scorer.summary(results)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Fields every input record must contain
REQUIRED_FIELDS = {"task_name", "task_instance_id", "output"}


class StandaloneEval1Scorer:
    """Score externally-collected outputs against the Biomni-Eval1 benchmark.

    The scorer wraps ``BiomniEval1().evaluate()`` from the biomni package and
    adds conveniences for batch loading, file I/O, and summary statistics.
    It has no dependency on AgentRunner or BenchmarkRunner — usable in any
    script or notebook.
    """

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_one(
        self,
        task_name: str,
        task_instance_id: int,
        output: str,
        **extra: Any,
    ) -> dict:
        """Score a single output and return a result dict.

        Args:
            task_name: One of the 10 Biomni-Eval1 task types.
            task_instance_id: Row ID within that task type.
            output: The agent's free-text response.
            **extra: Any additional metadata to preserve in the result
                     (e.g., framework="my_tool", model="gpt-4o").

        Returns:
            Dict containing all input fields plus ``score``, ``passed``,
            and ``metrics``.
        """
        record: dict[str, Any] = {
            "task_name": task_name,
            "task_instance_id": task_instance_id,
            "output": output,
            **extra,
        }
        try:
            from biomni.eval.biomni_eval1 import BiomniEval1  # type: ignore[import]

            evaluator = BiomniEval1()
            score = float(evaluator.evaluate(task_name, task_instance_id, output))
            record["score"] = score
            record["passed"] = score > 0.0
            record["metrics"] = {
                "task_name": task_name,
                "task_instance_id": task_instance_id,
                "output_length": len(output or ""),
            }
        except Exception as exc:
            record["score"] = 0.0
            record["passed"] = False
            record["metrics"] = {
                "task_name": task_name,
                "task_instance_id": task_instance_id,
                "eval_error": str(exc),
            }
        return record

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_batch(self, records: list[dict]) -> list[dict]:
        """Score a list of records.

        Args:
            records: List of dicts. Each must have ``task_name``,
                     ``task_instance_id``, and ``output``. Extra keys
                     are preserved in the output.

        Returns:
            List of result dicts (same order) with ``score``, ``passed``,
            and ``metrics`` added.

        Raises:
            ValueError: If a record is missing a required field.
        """
        results = []
        for i, rec in enumerate(records):
            missing = REQUIRED_FIELDS - rec.keys()
            if missing:
                raise ValueError(
                    f"Record at index {i} is missing required fields: {missing}. "
                    f"Required: {REQUIRED_FIELDS}"
                )
            extra = {k: v for k, v in rec.items() if k not in REQUIRED_FIELDS}
            result = self.score_one(
                task_name=str(rec["task_name"]),
                task_instance_id=int(rec["task_instance_id"]),
                output=str(rec["output"] or ""),
                **extra,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def score_from_jsonl(self, path: str | Path) -> list[dict]:
        """Load records from a JSONL file and score them.

        Args:
            path: Path to a JSONL file (one JSON object per line).
                  Each object must have ``task_name``, ``task_instance_id``,
                  and ``output``.

        Returns:
            List of scored result dicts.
        """
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self.score_batch(records)

    def score_from_csv(self, path: str | Path) -> list[dict]:
        """Load records from a CSV file and score them.

        Args:
            path: Path to a CSV file with column headers.
                  Must include: ``task_name``, ``task_instance_id``, ``output``.
                  Extra columns are preserved in the output.

        Returns:
            List of scored result dicts.
        """
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return self.score_batch(records)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, results: list[dict], output_dir: str | Path) -> Path:
        """Save scored results to a JSONL file.

        Appends to ``results.jsonl`` inside *output_dir*, matching the
        format produced by ``BenchmarkRunner`` and ``Eval1Runner`` so
        the same pandas analysis cells work on all result files.

        Args:
            results: List of result dicts from any ``score_*`` method.
            output_dir: Directory to write into (created if absent).

        Returns:
            Path to the output file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / "results.jsonl"
        with open(output_file, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        return output_file

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def summary(self, results: list[dict]) -> dict:
        """Compute overall and per-task accuracy statistics.

        Args:
            results: List of scored result dicts.

        Returns:
            Dict with keys:
              - ``n_total``   — total records
              - ``n_passed``  — records with score > 0
              - ``avg_score`` — mean score
              - ``pass_rate`` — fraction passed
              - ``per_task``  — dict keyed by task_name, each with the
                                same four keys plus ``n_total``
        """
        if not results:
            return {
                "n_total": 0,
                "n_passed": 0,
                "avg_score": 0.0,
                "pass_rate": 0.0,
                "per_task": {},
            }

        # Accumulate per-task stats
        per_task: dict[str, dict] = {}
        for r in results:
            task = str(r.get("task_name", "unknown"))
            if task not in per_task:
                per_task[task] = {"scores": [], "n_passed": 0}
            per_task[task]["scores"].append(float(r.get("score", 0.0)))
            if r.get("passed", False):
                per_task[task]["n_passed"] += 1

        per_task_stats = {
            task: {
                "n_total": len(v["scores"]),
                "n_passed": v["n_passed"],
                "avg_score": sum(v["scores"]) / len(v["scores"]),
                "pass_rate": v["n_passed"] / len(v["scores"]),
            }
            for task, v in per_task.items()
        }

        all_scores = [float(r.get("score", 0.0)) for r in results]
        n_passed = sum(1 for r in results if r.get("passed", False))

        return {
            "n_total": len(results),
            "n_passed": n_passed,
            "avg_score": sum(all_scores) / len(all_scores),
            "pass_rate": n_passed / len(results),
            "per_task": per_task_stats,
        }
