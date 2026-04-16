# KVBench

**KVBench** is a reproducible benchmark suite for studying the **latency, throughput, and memory tradeoffs** of KV-cache strategies in transformer inference.

This project focuses on a practical systems question:

> **When does KV cache become worth it, and which cache strategy should be preferred under real GPU memory constraints?**

The current benchmark setup targets **causal language models** and studies how inference behavior changes across:
- prompt length
- generation length
- cache strategy
- GPU memory pressure

---

## Why this project matters

KV cache is one of the key mechanisms that makes autoregressive generation practical in modern transformers.

During token-by-token decoding, recomputing attention states for the full previous context becomes increasingly expensive. KV cache avoids that repeated work by storing previously computed key/value tensors and reusing them during generation.

That creates an important tradeoff:

- **less recomputation**
- **faster decoding**
- **higher memory usage**

KVBench is designed to study that tradeoff in a controlled and reproducible way.

---

## Project goals

This project aims to answer questions such as:

- How much does KV cache improve decoding speed?
- How does GPU memory usage grow with context length?
- How do prompt length and generation length change the benefit of caching?
- When is `dynamic` cache preferable to `offloaded` cache?
- What happens when backend support prevents some cache strategies from running?

---

## Current scope

The project currently benchmarks:

- **No cache** (`use_cache=False`)
- **Dynamic cache**
- **Offloaded cache**
- **Static cache attempt** with failure logging when unsupported on the current setup

It also includes:
- single-run baseline benchmarking
- prompt-length sweeps
- multi-run prompt-length sweeps with warmup and aggregation
- cache-strategy sweeps
- raw and aggregated JSON outputs

---

## Hardware setup

Current experiments were developed and tested on:

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU
- **VRAM:** 6 GB
- **Platform:** Windows
- **Framework:** PyTorch + Hugging Face Transformers

This hardware constraint is part of the project motivation: the benchmark is meant to reflect realistic limited-VRAM experimentation, not only ideal large-server conditions.

---

## Repository structure

```text
kvbench/
├── README.md
├── PROJECT_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── check_gpu.py
├── test_model.py
├── configs/
├── docs/
│   └── progress_log.md
├── experiments/
├── notebooks/
├── results/
│   ├── raw/
│   └── figures/
└── src/
    ├── run_benchmark.py
    ├── run_prompt_sweep.py
    ├── run_prompt_sweep_multirun.py
    ├── run_cache_strategy_sweep.py
    └── run_generation_length_sweep.py
```

---

## Implemented benchmark stages

### 1. Environment validation
- CUDA/GPU detection
- PyTorch validation
- tokenizer/model loading
- first successful generation test

### 2. Baseline benchmark
- compare `use_cache=False` vs `use_cache=True`
- measure latency
- measure generated tokens
- measure tokens/sec
- measure peak GPU memory

### 3. Prompt-length sweep
- vary prompt length
- compare no-cache vs cached runs
- analyze how cache benefit evolves with context length

### 4. Multi-run prompt-length sweep
- warmup runs
- repeated trials
- mean/std aggregation
- raw and aggregated result files

### 5. Cache-strategy sweep
- compare `no_cache`, `dynamic`, and `offloaded`
- attempt `static` cache and log backend/toolchain failures
- analyze latency/memory tradeoffs between working strategies

### 6. Next stage
- generation-length sweep
- prefill vs decode decomposition
- break-even analysis
- decision guide for cache selection

---

## Key findings so far

Early results already show useful patterns:

- KV cache generally improves decoding speed compared with no-cache.
- The benefit becomes much clearer at longer contexts.
- `dynamic` cache is a strong default strategy on the current setup.
- `offloaded` cache becomes competitive for longer contexts and lower-VRAM conditions.
- `static` cache is not currently usable on this machine due to backend/toolchain support issues.

These observations are exactly the kind of real-world constraints this project is meant to capture.

---

## How to set up the environment

### 1. Create a virtual environment

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Validate the environment

```bash
python check_gpu.py
python test_model.py
```

---

## Example commands

### Baseline benchmark
```bash
python src/run_benchmark.py
```

### Prompt-length sweep
```bash
python src/run_prompt_sweep.py
```

### Multi-run prompt-length sweep
```bash
python src/run_prompt_sweep_multirun.py
```

### Cache-strategy sweep
```bash
python src/run_cache_strategy_sweep.py
```

### Generation-length sweep
```bash
python src/run_generation_length_sweep.py
```

---

## Output files

Benchmark scripts save structured outputs inside:

```text
results/raw/
```

Typical outputs include:
- raw per-trial JSON logs
- aggregated JSON summaries
- later: figures and analysis artifacts

---

## Project direction

The goal is not to build “just another benchmark script.”

The goal is to build a **small, serious inference study** that explains:

- when KV-cache strategies win
- why they win
- how memory constraints affect the decision
- which strategy should be chosen under practical deployment limitations

---

## Planned next steps

- add generation-length sweep analysis
- separate prefill and decode effects
- add break-even analysis
- add plotting and visualization
- write a concise report of findings
- turn benchmark results into a practical cache-selection guide

---

## Limitations

Current limitations include:
- small-model focus due to 6 GB VRAM constraints
- single-machine Windows environment
- some backend-specific features unavailable locally
- early-stage result coverage

These limitations are intentional to some extent: part of the project is to study inference tradeoffs under constrained hardware, not only under large production GPUs.

---

## Long-term vision

KVBench is being developed as more than a toy benchmark.

The long-term goal is to turn it into a practical and well-documented study of KV-cache strategy selection for transformer inference under real latency and memory constraints.

---

## Author

Built by **Abd El Jalil BZN** as an LLM systems / inference optimization project focused on KV-cache behavior in transformer decoding.
