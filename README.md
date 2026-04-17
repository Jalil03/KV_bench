# KVBench

KVBench is a lightweight, reproducible benchmark suite for studying the latency, throughput, and memory tradeoffs of KV cache strategies in transformer inference.

This project focuses on a practical question: when does KV cache materially help, how much memory does it cost, and which cache mode is the best default on constrained local hardware.

## What This Repo Shows

The current benchmark set covers:

- cache vs no-cache generation
- prompt-length scaling
- cache strategy comparison
- a decode-focused microbenchmark at long context

The experiments were intentionally scoped to remain stable on an RTX 3060 Laptop GPU with 6 GB VRAM. A broader generation-length sweep was avoided after heavier runs caused system instability and shutdowns.

## Main Findings

- KV cache becomes much more valuable at long context than at short context.
- `dynamic` cache is the strongest default on the current setup.
- `offloaded` cache is a useful fallback when memory pressure matters.
- `no_cache` breaks down for decode-heavy long-context inference.
- `static` cache could not be evaluated locally because of Triton/backend support issues.

From the saved prompt-length results:

- at `512` prompt tokens, `use_cache=False` averaged about `8.81 s`
- at `512` prompt tokens, `use_cache=True` averaged about `3.63 s`

That makes the long-context advantage of caching large enough to be visible even on a small local benchmark.

## Repo Layout

```text
KV_cache/
|-- README.md
|-- requirements.txt
|-- check_gpu.py
|-- test_model.py
|-- src/
|   |-- run_benchmark.py
|   |-- run_prompt_sweep.py
|   |-- run_prompt_sweep_multirun.py
|   |-- run_cache_strategy_sweep.py
|   |-- run_decode_microbenchmark.py
|   `-- plot_results.py
|-- results/
|   |-- raw/
|   `-- figures/
`-- notebooks/
```

## Environment

- GPU: NVIDIA RTX 3060 Laptop GPU
- VRAM: 6 GB
- Model: `HuggingFaceTB/SmolLM2-360M`
- Frameworks: PyTorch + Hugging Face Transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

Check GPU visibility:

```bash
python check_gpu.py
```

Run a simple model sanity check:

```bash
python test_model.py
```

## How To Reproduce

Baseline benchmark:

```bash
python src/run_benchmark.py
```

Prompt-length multirun benchmark:

```bash
python src/run_prompt_sweep_multirun.py
```

Cache strategy comparison:

```bash
python src/run_cache_strategy_sweep.py
```

Decode-focused microbenchmark:

```bash
python src/run_decode_microbenchmark.py
```

Regenerate plots:

```bash
python src/plot_results.py
```

## Outputs

Raw and aggregated experiment results are written to `results/raw/`.

Generated figures are written to `results/figures/`.

The plotting script now uses trial standard deviations as error bars to make variance visible instead of showing mean curves alone.

For detailed plot interpretation, see [Figure guide](C:/Users/JL/OneDrive/Desktop/KV_cache/notebooks/Figure%20guide.md).

## Benchmark Scope

### 1. Prompt-length scaling

Tests how latency, throughput, and peak GPU memory change as prompt length grows for:

- `use_cache=False`
- `use_cache=True`

### 2. Cache strategy comparison

Tests:

- `no_cache`
- `dynamic`
- `offloaded`
- `static` when backend support allows it

### 3. Decode-focused microbenchmark

Uses a fixed long prompt (`512` tokens) and a conservative decode length (`64` new tokens) to isolate the decode-side value of KV cache without risking unstable long runs on 6 GB hardware.

## Current Limitations

- Results come from a single laptop GPU environment.
- The tested model is intentionally small for safety and reproducibility.
- `static` cache failed locally because Triton/backend support was unavailable.
- The decode split uses a practical timing estimate rather than kernel-level instrumentation.
- A broad generation-length sweep was intentionally not run on this machine because previous heavier tests caused full system shutdowns.

## Why The Scope Is Conservative

The smaller decode benchmark is deliberate. On this hardware, pushing generation length higher is not just a slower run; it can make the machine unstable. The project therefore prioritizes:

- safe repeatability
- clear comparisons
- honest reporting of hardware constraints

This makes the benchmark more useful for real local-development scenarios, where safety and reproducibility matter as much as raw scale.

## Next Improvements

- add a compact summary table directly in the README
- refactor duplicated runner logic into shared utilities
- test one additional small model for cross-model comparison
- add batch-size sensitivity experiments if hardware allows
- compare local results on a higher-VRAM machine in a future extension
