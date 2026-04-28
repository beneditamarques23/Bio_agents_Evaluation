"""
Unit tests for StandaloneEval1Scorer.

All tests run without the biomni extra installed:
- summary(), save(), score_batch() validation, and file loaders are pure Python.
- score_one() / score_batch() are tested for graceful degradation when
  BiomniEval1 is unavailable (eval_error in metrics, score=0.0).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bio_agents.evaluation.standalone_scorer import (
    REQUIRED_FIELDS,
    StandaloneEval1Scorer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scorer() -> StandaloneEval1Scorer:
    return StandaloneEval1Scorer()


@pytest.fixture()
def fake_scored_results() -> list[dict]:
    """Pre-scored records (no biomni needed)."""
    return [
        {
            "task_name": "lab_bench_seqqa",
            "task_instance_id": 1,
            "output": "A",
            "score": 1.0,
            "passed": True,
            "metrics": {"task_name": "lab_bench_seqqa", "task_instance_id": 1},
        },
        {
            "task_name": "lab_bench_seqqa",
            "task_instance_id": 2,
            "output": "Z",
            "score": 0.0,
            "passed": False,
            "metrics": {"task_name": "lab_bench_seqqa", "task_instance_id": 2},
        },
        {
            "task_name": "crispr_delivery",
            "task_instance_id": 1,
            "output": "b",
            "score": 1.0,
            "passed": True,
            "metrics": {"task_name": "crispr_delivery", "task_instance_id": 1},
        },
    ]


# ---------------------------------------------------------------------------
# REQUIRED_FIELDS constant
# ---------------------------------------------------------------------------


def test_required_fields_constant() -> None:
    assert REQUIRED_FIELDS == {"task_name", "task_instance_id", "output"}


# ---------------------------------------------------------------------------
# score_batch — input validation
# ---------------------------------------------------------------------------


def test_score_batch_raises_on_missing_task_name(scorer: StandaloneEval1Scorer) -> None:
    with pytest.raises(ValueError, match="task_name"):
        scorer.score_batch([{"task_instance_id": 1, "output": "A"}])


def test_score_batch_raises_on_missing_task_instance_id(
    scorer: StandaloneEval1Scorer,
) -> None:
    with pytest.raises(ValueError, match="task_instance_id"):
        scorer.score_batch([{"task_name": "lab_bench_seqqa", "output": "A"}])


def test_score_batch_raises_on_missing_output(scorer: StandaloneEval1Scorer) -> None:
    with pytest.raises(ValueError, match="output"):
        scorer.score_batch([{"task_name": "lab_bench_seqqa", "task_instance_id": 1}])


def test_score_batch_error_includes_index(scorer: StandaloneEval1Scorer) -> None:
    """Error message must mention the record index so the caller can find it."""
    with pytest.raises(ValueError, match="index 2"):
        scorer.score_batch(
            [
                {"task_name": "lab_bench_seqqa", "task_instance_id": 1, "output": "A"},
                {"task_name": "lab_bench_seqqa", "task_instance_id": 2, "output": "B"},
                {"task_name": "lab_bench_seqqa", "output": "C"},  # missing id
            ]
        )


# ---------------------------------------------------------------------------
# score_one / score_batch — graceful degradation without biomni
# ---------------------------------------------------------------------------


def test_score_one_graceful_without_biomni(scorer: StandaloneEval1Scorer) -> None:
    """When BiomniEval1 is unavailable the scorer must not raise."""
    result = scorer.score_one(
        task_name="lab_bench_seqqa",
        task_instance_id=1,
        output="A",
    )
    # Required output fields are always present
    assert "score" in result
    assert "passed" in result
    assert "metrics" in result
    # Score is a float
    assert isinstance(result["score"], float)
    # Passed is consistent with score
    assert result["passed"] == (result["score"] > 0.0)


def test_score_one_preserves_extra_fields(scorer: StandaloneEval1Scorer) -> None:
    result = scorer.score_one(
        task_name="lab_bench_seqqa",
        task_instance_id=1,
        output="A",
        framework="my_tool",
        model="gpt-4o",
        run_id=42,
    )
    assert result["framework"] == "my_tool"
    assert result["model"] == "gpt-4o"
    assert result["run_id"] == 42


def test_score_batch_graceful_without_biomni(scorer: StandaloneEval1Scorer) -> None:
    records = [
        {"task_name": "lab_bench_seqqa", "task_instance_id": 1, "output": "A"},
        {"task_name": "crispr_delivery", "task_instance_id": 2, "output": "b"},
    ]
    results = scorer.score_batch(records)
    assert len(results) == 2
    for r in results:
        assert "score" in r
        assert "passed" in r
        assert "metrics" in r


def test_score_batch_preserves_extra_fields(scorer: StandaloneEval1Scorer) -> None:
    records = [
        {
            "task_name": "lab_bench_seqqa",
            "task_instance_id": 1,
            "output": "A",
            "model": "gpt-4o",
            "cost": 0.002,
        }
    ]
    results = scorer.score_batch(records)
    assert results[0]["model"] == "gpt-4o"
    assert results[0]["cost"] == 0.002


def test_score_batch_returns_same_length(scorer: StandaloneEval1Scorer) -> None:
    records = [
        {"task_name": "lab_bench_seqqa", "task_instance_id": i, "output": "A"}
        for i in range(5)
    ]
    results = scorer.score_batch(records)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_empty(scorer: StandaloneEval1Scorer) -> None:
    stats = scorer.summary([])
    assert stats["n_total"] == 0
    assert stats["n_passed"] == 0
    assert stats["avg_score"] == 0.0
    assert stats["pass_rate"] == 0.0
    assert stats["per_task"] == {}


def test_summary_counts(
    scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    stats = scorer.summary(fake_scored_results)
    assert stats["n_total"] == 3
    assert stats["n_passed"] == 2


def test_summary_avg_score(
    scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    stats = scorer.summary(fake_scored_results)
    assert abs(stats["avg_score"] - 2 / 3) < 1e-9


def test_summary_pass_rate(
    scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    stats = scorer.summary(fake_scored_results)
    assert abs(stats["pass_rate"] - 2 / 3) < 1e-9


def test_summary_per_task_keys(
    scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    stats = scorer.summary(fake_scored_results)
    assert set(stats["per_task"].keys()) == {"lab_bench_seqqa", "crispr_delivery"}


def test_summary_per_task_values(
    scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    stats = scorer.summary(fake_scored_results)
    seqqa = stats["per_task"]["lab_bench_seqqa"]
    assert seqqa["n_total"] == 2
    assert seqqa["n_passed"] == 1
    assert abs(seqqa["avg_score"] - 0.5) < 1e-9
    assert abs(seqqa["pass_rate"] - 0.5) < 1e-9

    crispr = stats["per_task"]["crispr_delivery"]
    assert crispr["n_total"] == 1
    assert crispr["n_passed"] == 1
    assert crispr["avg_score"] == 1.0
    assert crispr["pass_rate"] == 1.0


def test_summary_all_pass(scorer: StandaloneEval1Scorer) -> None:
    results = [{"score": 1.0, "passed": True, "task_name": "t"} for _ in range(4)]
    stats = scorer.summary(results)
    assert stats["n_passed"] == 4
    assert stats["avg_score"] == 1.0
    assert stats["pass_rate"] == 1.0


def test_summary_all_fail(scorer: StandaloneEval1Scorer) -> None:
    results = [{"score": 0.0, "passed": False, "task_name": "t"} for _ in range(3)]
    stats = scorer.summary(results)
    assert stats["n_passed"] == 0
    assert stats["avg_score"] == 0.0
    assert stats["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_creates_file(
    tmp_path: Path,
    scorer: StandaloneEval1Scorer,
    fake_scored_results: list[dict],
) -> None:
    out = scorer.save(fake_scored_results, output_dir=tmp_path)
    assert out.exists()
    assert out.name == "results.jsonl"


def test_save_creates_output_dir(
    tmp_path: Path, scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    nested = tmp_path / "a" / "b" / "c"
    scorer.save(fake_scored_results, output_dir=nested)
    assert nested.is_dir()


def test_save_writes_correct_line_count(
    tmp_path: Path, scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    out = scorer.save(fake_scored_results, output_dir=tmp_path)
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(fake_scored_results)


def test_save_valid_json_per_line(
    tmp_path: Path, scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    out = scorer.save(fake_scored_results, output_dir=tmp_path)
    for line in out.read_text().splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)


def test_save_appends_on_second_call(
    tmp_path: Path, scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    scorer.save(fake_scored_results[:1], output_dir=tmp_path)
    scorer.save(fake_scored_results[1:], output_dir=tmp_path)
    out = tmp_path / "results.jsonl"
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(fake_scored_results)


def test_save_round_trip(
    tmp_path: Path, scorer: StandaloneEval1Scorer, fake_scored_results: list[dict]
) -> None:
    out = scorer.save(fake_scored_results, output_dir=tmp_path)
    reloaded = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert reloaded[0]["task_name"] == fake_scored_results[0]["task_name"]
    assert reloaded[0]["score"] == fake_scored_results[0]["score"]


# ---------------------------------------------------------------------------
# score_from_jsonl()
# ---------------------------------------------------------------------------


def test_score_from_jsonl_loads_and_scores(
    tmp_path: Path, scorer: StandaloneEval1Scorer
) -> None:
    p = tmp_path / "inputs.jsonl"
    records = [
        {"task_name": "lab_bench_seqqa", "task_instance_id": 1, "output": "A"},
        {"task_name": "crispr_delivery", "task_instance_id": 2, "output": "b"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    results = scorer.score_from_jsonl(p)
    assert len(results) == 2
    for r in results:
        assert "score" in r


def test_score_from_jsonl_skips_blank_lines(
    tmp_path: Path, scorer: StandaloneEval1Scorer
) -> None:
    p = tmp_path / "inputs.jsonl"
    row_a = json.dumps(
        {"task_name": "lab_bench_seqqa", "task_instance_id": 1, "output": "A"}
    )
    row_b = json.dumps(
        {"task_name": "crispr_delivery", "task_instance_id": 2, "output": "b"}
    )
    p.write_text(row_a + "\n\n" + row_b + "\n")
    results = scorer.score_from_jsonl(p)
    assert len(results) == 2


def test_score_from_jsonl_preserves_extra_fields(
    tmp_path: Path, scorer: StandaloneEval1Scorer
) -> None:
    p = tmp_path / "inputs.jsonl"
    p.write_text(
        json.dumps(
            {
                "task_name": "lab_bench_seqqa",
                "task_instance_id": 1,
                "output": "A",
                "model": "gpt-4o",
            }
        )
        + "\n"
    )
    results = scorer.score_from_jsonl(p)
    assert results[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# score_from_csv()
# ---------------------------------------------------------------------------


def test_score_from_csv_loads_and_scores(
    tmp_path: Path, scorer: StandaloneEval1Scorer
) -> None:
    p = tmp_path / "inputs.csv"
    rows = [
        {"task_name": "lab_bench_seqqa", "task_instance_id": "1", "output": "A"},
        {"task_name": "crispr_delivery", "task_instance_id": "2", "output": "b"},
    ]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_name", "task_instance_id", "output"]
        )
        writer.writeheader()
        writer.writerows(rows)

    results = scorer.score_from_csv(p)
    assert len(results) == 2
    for r in results:
        assert "score" in r


def test_score_from_csv_preserves_extra_columns(
    tmp_path: Path, scorer: StandaloneEval1Scorer
) -> None:
    p = tmp_path / "inputs.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_name", "task_instance_id", "output", "model"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_name": "lab_bench_seqqa",
                "task_instance_id": "1",
                "output": "A",
                "model": "gpt-4o",
            }
        )

    results = scorer.score_from_csv(p)
    assert results[0]["model"] == "gpt-4o"
