# bio-agents-playground

A playground for benchmarking **bio-agent frameworks** across models, tasks, and evaluation metrics.

## Table of Contents

- [Overview](#overview)
- [Structure](#structure)
- [Setup](#setup)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Make Targets](#make-targets)
- [Results Format](#results-format)
- [Adding a Local Model (Ollama)](#adding-a-local-model-ollama)
- [Adding a Framework](#adding-a-framework)
- [Adding a Task](#adding-a-task)
- [Contributing](#contributing)

---

## Overview

The goal is to run a configurable matrix of:

```
tasks × frameworks × models → evaluation metrics
```

Results are saved as JSONL for downstream analysis in notebooks.

**Supported frameworks:**
`robin` · `biomni`

**Planned frameworks:**
`langchain` · `langgraph` · `anthropic_sdk` · `crewai` · `autogen` · `smolagents`

**Supported model providers:**
`anthropic` · `openai` · `google` · `groq` · `local (ollama)`

**Bio task domains:**
`drug_discovery` · `literature_search` · `gene_annotation` · `molecule_property` · `sequence_analysis` · `clinical_nlp` · `genomics`

---

## Structure

```
bio-agents-playground/
├── src/bio_agents/
│   ├── frameworks/      # AgentRunner adapters (one per framework)
│   │   ├── robin/       # FutureHouse/Edison Scientific Robin pipeline
│   │   └── biomni/      # Stanford Biomni general-purpose bio agent
│   ├── models/          # Model registry & provider info
│   ├── tasks/           # BioTask definitions (input, tools, eval)
│   │   ├── drug_discovery/  # assay_generation, candidate_ranking
│   │   ├── literature/      # literature_search
│   │   ├── genomics/        # gene_annotation
│   │   ├── molecule/        # molecule_property
│   │   └── biomni_eval1/    # BiomniEval1Task (parametric, dataset-driven)
│   ├── tools/           # Bio API wrappers (PubMed, BLAST, UniProt, FutureHouse, …)
│   ├── evaluation/      # BenchmarkRunner + metrics
│   └── config.py        # Settings (API keys via .env)
├── configs/             # YAML benchmark configs
├── pipelines/           # CLI entrypoints
│   ├── run_task.py      # single task runner
│   ├── run_benchmark.py # full matrix runner
│   └── run_eval1.py     # Biomni-Eval1 dataset runner
├── notebooks/           # Interactive Jupyter notebooks (mirror the pipelines)
│   ├── 01_run_task.ipynb
│   ├── 02_run_benchmark.ipynb
│   └── 03_run_eval1.ipynb
├── results/             # Benchmark outputs (JSONL)
└── tests/
```

---

## Setup

**Requirements:** [uv](https://docs.astral.sh/uv/) ≥ 0.4, Python 3.12+

```bash
# Install all extras (frameworks + model providers + dev tools)
make install

# Or selectively install only what you need
uv sync --extra robin --extra dev
uv sync --extra biomni --extra dev   # or: make setup-biomni
```

> **Biomni notes:**
> - First `agent.go()` call downloads ~11 GB of tools and datasets into `BIOMNI_DATA_PATH` (default `./data`). Run `make setup-biomni` to install the package upfront.
> - Biomni uses the OpenAI-compatible Ollama endpoint (`/v1`). Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3`) before running.

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### API Keys

| Provider | Key | Free tier |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | No |
| OpenAI | `OPENAI_API_KEY` | No |
| Google | `GOOGLE_API_KEY` | Yes — [aistudio.google.com](https://aistudio.google.com) |
| Groq | `GROQ_API_KEY` | Yes — [console.groq.com](https://console.groq.com) |
| Edison Scientific (FutureHouse) | `FUTUREHOUSE_API_KEY` | [platform.edisonscientific.com](https://platform.edisonscientific.com) |
| Local (Ollama) | — | Free — `ollama pull llama3` or `ollama pull qwen3` |

---

## Usage

### Run a single task interactively

```bash
# Robin — drug discovery pipeline
uv run python pipelines/run_task.py assay_generation \
  --framework robin \
  --model llama3

# Biomni — general-purpose bio agent
uv run python pipelines/run_task.py literature_search \
  --framework biomni \
  --model llama3
```

### Run a benchmark matrix

```bash
# Robin smoke test (requires FUTUREHOUSE_API_KEY + Ollama)
uv run python pipelines/run_benchmark.py --config configs/robin_smoke.yaml

# Biomni smoke test (requires Ollama or GROQ_API_KEY)
uv run python pipelines/run_benchmark.py --config configs/biomni_smoke.yaml

# Dry-run to preview the matrix
uv run python pipelines/run_benchmark.py --config configs/biomni_smoke.yaml --dry-run

# Full benchmark
make benchmark
```

### Benchmark config format

YAML configs define the full run matrix. Top-level keys not in `{tasks, frameworks, models, eval, output_dir}` are forwarded as kwargs to the framework runner:

Robin example (`configs/robin_smoke.yaml`):
```yaml
tasks:
  - assay_generation
frameworks:
  - robin
models:
  - llama3          # local Ollama — no key needed
  # - llama-3.3-70b # free tier — GROQ_API_KEY
eval:
  - success_rate
# Robin-specific kwargs
num_queries: 1
num_assays: 2
num_candidates: 2
output_dir: results/robin_smoke
```

Biomni example (`configs/biomni_smoke.yaml`):
```yaml
tasks:
  - literature_search
  - gene_annotation
  - molecule_property
frameworks:
  - biomni
models:
  - llama3          # local Ollama — no key needed
  # - llama-3.3-70b # free tier — GROQ_API_KEY
eval:
  - success_rate
# Biomni-specific kwargs
use_tool_retriever: true
timeout_seconds: 300
commercial_mode: false
output_dir: results/biomni_smoke
```

### Biomni-Eval1 dataset evaluation

[biomni/Eval1](https://huggingface.co/datasets/biomni/Eval1) is a curated benchmark of **433 test instances** across 10 biological reasoning task types (GWAS causal gene identification, rare disease diagnosis, CRISPR delivery, lab QA, and more). Answers are scored with binary exact match (0.0 or 1.0).

```bash
# Smoke: 3 task types × 1 instance (3 runs)
make eval1-smoke
# or:
uv run python pipelines/run_eval1.py --config configs/biomni_eval1_smoke.yaml --dry-run
uv run python pipelines/run_eval1.py --config configs/biomni_eval1_smoke.yaml

# Full: all 10 task types, all 433 instances (expensive)
make eval1-full
```

Config format (`configs/biomni_eval1_smoke.yaml`):
```yaml
task_names:
  - gwas_causal_gene_opentargets
  - lab_bench_seqqa
  - rare_disease_diagnosis
  # use [all] to run all 10 task types

instances_per_task: 1   # increase up to max available per task

frameworks:
  - biomni

models:
  - llama3

output_dir: results/biomni_eval1_smoke
```

### Notebook analysis

Results are saved to `results/<run>/results.jsonl`. Load them in a notebook:

```python
import pandas as pd
df = pd.read_json("results/quick_smoke/results.jsonl", lines=True)
df.groupby(["framework", "model"])["score"].mean()
```

---

## Notebooks

Three Jupyter notebooks in `notebooks/` mirror the CLI pipelines step-by-step, with explanations and analysis cells — suitable for interactive exploration or teaching.

| Notebook | Mirrors | What it covers |
|---|---|---|
| `01_run_task.ipynb` | `run_task.py` | Registry exploration, task inspection, single agent run, result breakdown |
| `02_run_benchmark.ipynb` | `run_benchmark.py` | Config loading, matrix dry-run, benchmark run, pandas analysis, JSONL reload |
| `03_run_eval1.ipynb` | `run_eval1.py` | Dataset preview, instance inspection, dry-run, per-task scoring, scaling up |

To open them:
```bash
uv run jupyter lab notebooks/
```

---

## Make Targets

| Target | What it does |
|---|---|
| `make install` | Install all extras + pre-commit hooks |
| `make setup-biomni` | Install only the Biomni extra |
| `make smoke` | Quick smoke test (robin + biomni, 1 task each) |
| `make benchmark` | Full benchmark matrix |
| `make eval1-smoke` | Biomni-Eval1 smoke (3 task types × 1 instance) |
| `make eval1-full` | Biomni-Eval1 full (all 433 instances) |
| `make notebooks` | Open Jupyter Lab in `notebooks/` |
| `make clean` | Remove notebook artifacts (`notebooks/results/`, `notebooks/data/`) |
| `make lint` | Run `ruff check` + `pyright` |
| `make test` | Run `pytest` |
| `make pre-commit` | Run all pre-commit hooks against every file |
| `make check` | Run `lint` + `test` + `pre-commit` (full CI gate) |

---

## Results Format

All runners write results to `results/<run_name>/results.jsonl` — one JSON object per line.

**`BenchmarkRunner` fields (`run_benchmark.py`, `run_task.py`):**

| Field | Type | Description |
|---|---|---|
| `task` | str | Task registry key |
| `framework` | str | Framework registry key |
| `model` | str | Model registry key |
| `score` | float | 0.0–1.0 evaluation score |
| `passed` | bool | Whether score ≥ passing threshold |
| `duration_s` | float | Wall-clock seconds |
| `output` | str | Agent's final text output |
| `tool_calls` | list | Intermediate agent steps |
| `metrics` | dict | Task-specific eval breakdown |

**`Eval1Runner` additional fields (`run_eval1.py`):**

| Field | Type | Description |
|---|---|---|
| `eval1_task_name` | str | One of the 10 Biomni-Eval1 task types |
| `task_instance_id` | int | Row ID in the `biomni/Eval1` HuggingFace dataset |

---

## Adding a Local Model (Ollama)

Any model available in [Ollama's library](https://ollama.com/library) can be used without an API key.

### 1. Pull the model

```bash
ollama pull qwen3          # example — works for any Ollama model name
```

### 2. Register it in the model registry

Add an entry to `src/bio_agents/models/registry.py`:

```python
"qwen3": {
    "provider": "local",
    "model_id": "qwen3",
    "litellm_id": "ollama_chat/qwen3",
},
```

The `model_id` must match the name used by `ollama pull`. The `litellm_id` always uses the `ollama_chat/` prefix.

### 3. Use it in a config or CLI

```bash
# CLI
uv run python pipelines/run_task.py literature_search \
  --framework biomni --model qwen3

# YAML config
models:
  - qwen3
```

> **Note:** Make sure Ollama is running before any local model run:
> ```bash
> ollama serve
> ```

---

## Adding a Framework

1. Create `src/bio_agents/frameworks/<name>/runner.py` implementing `AgentRunner`.
2. Register it in `src/bio_agents/frameworks/__init__.py`:

```python
from bio_agents.frameworks.my_framework.runner import MyRunner
FRAMEWORK_REGISTRY["my_framework"] = MyRunner
```

---

## Adding a Task

1. Create `src/bio_agents/tasks/<domain>/<task_name>.py` implementing `BioTask`.
2. Register it in `src/bio_agents/tasks/__init__.py`:

```python
from bio_agents.tasks.drug_discovery.assay import AssayGenerationTask
TASK_REGISTRY["assay_generation"] = AssayGenerationTask
```

---

## Contributing

```bash
# Before opening a PR, make sure all checks pass:
make check
```

This runs `ruff`, `pyright`, `pytest`, and all pre-commit hooks in one shot. The same gate runs in CI.
