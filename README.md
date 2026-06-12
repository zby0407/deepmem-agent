# DeepMem Agent

### Hermes-inspired Long-term Memory System for Conversational AI

A self-contained memory agent package for long-term conversational memory. Inspired by Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent) (MIT), DeepMem Agent adds cognitive-science-based retention scoring, hybrid semantic+keyword retrieval, and background memory consolidation ("dreaming") to give conversational AI agents persistent, evolving memory over long-running interactions.

---

## Architecture

### Three-Scope Cognitive Memory

Memories are classified into three scopes, each with its own decay constant modeled after cognitive science forgetting curves:

| Scope | Decay Constant (k) | Half-Life | Use Case |
|---|---|---|---|
| **Semantic** | 0.0005 | ~1386 days | Stable facts, identity, long-term preferences |
| **Procedural** | 0.005 | ~139 days | Habits, routines, behavioral patterns |
| **Episodic** | 0.02 | ~35 days | Recent events, short-term context |

### 2-Factor Retention Score

Each memory's retention is computed as the product of exponential time decay and a spaced-repetition confirmation boost:

```
retention = e^(-k * age_days) * min(1.8, 1 + ln(1 + confirmations) * 0.08)
```

- **Time decay**: Older memories fade, with the rate governed by the scope's decay constant.
- **Confirmation boost**: Each time a memory is reinforced (re-mentioned or confirmed by the user), its retention increases logarithmically, capped at 1.8x. This models the spacing effect from cognitive psychology.

### Hybrid Search Pipeline

Retrieval uses a multi-stage pipeline that combines the strengths of keyword and semantic search:

1. **FTS5 Pre-filter** -- SQLite full-text search for fast candidate generation
2. **Semantic Embedding Cosine** -- Dense vector similarity for meaning-based matching
3. **BM25 Sigmoid Normalization** -- Raw BM25 scores normalized to [0, 1] via adaptive sigmoid
4. **Entity Boost** -- Named-entity overlap between query and memory gives a bonus
5. **Additive Scoring** -- Signals combined: `(semantic + bm25 + entity_boost) / max_possible`
6. **Retention Decay** -- Final scores weighted by each memory's current retention factor

### Bi-Temporal Model

Every memory carries three timestamps: `valid_at`, `invalid_at`, and `expired_at`. Memories are **never hard-deleted**. When a fact becomes outdated (e.g., "user moved to a new city"), the old memory is marked with `invalid_at` and optionally linked to the new memory. This preserves a full temporal audit trail and allows the system to reason about what was true at any point in time.

### Background Consolidation ("Dreaming")

A background process runs periodically to consolidate, deduplicate, and prune the memory store. It is triggered when **both** conditions are met:

- At least **10 new memories** since the last consolidation
- At least **4 hours** have elapsed since the last run

The consolidation runs three phases:

1. **Deduction** -- Detect contradictions (new fact supersedes old), stale memories (invalidated by newer information), and generate inferences from existing facts.
2. **Induction** -- Identify cross-memory patterns (e.g., recurring themes, habit clusters) and create higher-order summary memories.
3. **Hindsight Pruning** -- Evaluate hindsight rules (always-inject procedural memories) by hit rate. Prune rules that have zero hits despite many injection attempts.

### Profile Anchoring

Identity-level memories (e.g., user's name, occupation, core preferences) are extracted and baked into a **system prompt prefix**. This prefix is sent with every LLM call, enabling prompt caching at the API level for significant latency and cost savings.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys (see Configuration section below)

# 3. Run the benchmark (see next section for full instructions)
python benchmarks/locomo_benchmark.py --benchmark locomo --limit 10
```

---

## LoCoMo Benchmark Reproduction

This package includes a benchmark runner that reproduces results on the **LoCoMo** (Long-Context Conversational Memory) dataset. Follow these steps to reproduce our numbers.

### Step 1: Get the LoCoMo Dataset

Clone or download the LoCoMo dataset from the official repository:

```bash
git clone https://github.com/adaptive-computing/locomo.git
```

Copy the dataset file `locomo10.json` into the `data/locomo/` directory:

```bash
mkdir -p data/locomo
cp locomo/locomo10.json data/locomo/
```

The file contains 10 conversation samples, each with a long multi-turn dialogue and a set of QA pairs for evaluation.

### Step 2: Configure Environment

Set your LLM provider API key. The default provider is DashScope (Alibaba Cloud / Qwen):

```bash
export DASHSCOPE_API_KEY=sk-your-key-here
```

Or edit the `.env` file directly (see [Configuration](#configuration) for all options).

### Step 3: Run the Benchmark

```bash
# Full run (10 samples, all questions)
python benchmarks/locomo_benchmark.py --benchmark locomo --limit 10

# Quick smoke test (1 sample)
python benchmarks/locomo_benchmark.py --benchmark locomo --limit 1

# Limit questions per sample
python benchmarks/locomo_benchmark.py --benchmark locomo --limit 10 --qa-limit 50
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--benchmark` | `locomo` | Benchmark dataset to use (`locomo` or `longmemeval`) |
| `--limit` | `0` (all) | Max number of conversation samples to process |
| `--qa-limit` | `0` (all) | Max number of QA pairs to evaluate per sample |
| `--max-turns` | `0` (all) | Max conversation turns to ingest per sample |
| `--concurrency` | `1` | Number of samples to process in parallel |
| `--sample-ids` | (all) | Comma-separated list of specific sample IDs to run |
| `--output` | `results_{timestamp}.json` | Output file path for detailed results |
| `--data-dir` | `data/` | Root directory for benchmark data files |

### Expected Results

Results from our reproduction run:

| Configuration | Samples | Questions | EM | F1 | Judge | Evidence Hit | Avg Latency |
|---|---|---|---|---|---|---|---|
| **Full Merged** | 10 | 1542 | 20.6% | 28.0% | 90.5% | 13.2% | ~2.9s |
| **Baseline** | 3 | 387 | 18.6% | 26.5% | 92.0% | 14.2% | -- |

The "Full Merged" row processes all 10 samples with the full pipeline (ingestion, consolidation, retrieval, and answering). The "Baseline" row uses 3 samples as a faster sanity check.

---

## Metrics Explained

- **EM (Exact Match)**: Percentage of answers that exactly match the ground truth. Strict but sensitive to paraphrasing issues.
- **F1 Score**: Token-level overlap between predicted and ground truth answers. More forgiving of minor wording differences.
- **Judge Score**: An LLM judge evaluates whether the generated answer is semantically correct given the question and ground truth. Scored as the percentage of questions judged correct. This is the most reliable metric for conversational QA.
- **Evidence Hit**: Percentage of questions where the retrieval pipeline returned at least one memory containing the ground truth answer. Measures retrieval recall independently of generation quality.

---

## Package Structure

```
deepmem-agent/
├── deepmem/                      # Core memory agent package
│   ├── __init__.py               # Public API exports
│   ├── memory_provider.py        # Abstract base class for memory backends
│   ├── memory_manager.py         # Orchestrator: coordinates extraction, storage, retrieval
│   ├── local_memory.py           # SQLite memory provider (the core implementation)
│   ├── memory_extract.py         # LLM-based fact extraction from conversation turns
│   ├── memory_consolidator.py    # Background "dreaming" consolidation process
│   ├── memory_history.py         # Audit trail for memory mutations
│   ├── scoring.py                # BM25 normalization + hybrid additive scoring
│   ├── embedding.py              # Embedding API client (OpenAI-compatible)
│   ├── llm.py                    # LLM API client (OpenAI-compatible)
│   └── harness.py                # Turn-by-turn orchestration for conversations
├── benchmarks/                   # Evaluation scripts
│   └── locomo_benchmark.py       # LoCoMo / LongMemEval benchmark runner
├── scripts/
│   └── inject_test_memories.py   # Inject synthetic memories for manual testing
├── tests/
│   └── test_scoring.py           # Unit tests for scoring module
├── data/                         # Benchmark data (not included, see above)
├── requirements.txt              # Python dependencies (minimal)
├── .env.example                  # Environment variable template
├── HERMES_LICENSE.txt            # Nous Research Hermes Agent attribution
├── LICENSE                       # MIT License
└── README.md
```

---

## Configuration

All configuration is via environment variables (or a `.env` file loaded by `python-dotenv`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `dashscope` | LLM provider: `dashscope`, `openai`, or custom |
| `DASHSCOPE_API_KEY` | -- | API key for DashScope (Alibaba Cloud / Qwen) |
| `OPENAI_API_KEY` | -- | API key for OpenAI |
| `LLM_BASE_URL` | (auto) | Custom OpenAI-compatible endpoint URL |
| `LLM_MODEL` | `qwen-flash` / `gpt-4o-mini` | Model name (default depends on provider) |
| `EMBEDDING_API_KEY` | (same as LLM) | Override embedding API key |
| `EMBEDDING_BASE_URL` | (same as LLM) | Override embedding endpoint URL |
| `EMBEDDING_MODEL` | `text-embedding-v3` | Embedding model name |

---

## License

MIT. See [LICENSE](LICENSE).

The memory architecture and cognitive-science-inspired design are adapted from Nous Research's [Hermes Agent](https://github.com/NousResearch/hermes-agent), also released under the MIT License. See [HERMES_LICENSE.txt](HERMES_LICENSE.txt) for the original license text.
