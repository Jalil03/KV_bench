# Figure Guide

This guide explains what each figure in `results/figures/` is meant to show and how to interpret it in the context of this project.

## Prompt-length scaling

### `prompt_length_vs_latency.png`

This figure compares total generation latency for cache-enabled and no-cache inference as prompt length increases.

What it shows:

- At short prompts, the gap between the two modes is relatively modest.
- As prompt length grows, latency without cache rises much more sharply.
- The biggest separation appears at long context, especially around `512` prompt tokens.

Why it matters:

This is the clearest illustration that KV cache becomes much more valuable as context length grows. Without cache, the model repeatedly recomputes attention over the full prefix during decoding, which becomes increasingly expensive.

### `prompt_length_vs_throughput.png`

This figure shows throughput in tokens per second for cache and no-cache runs across prompt lengths.

What it shows:

- Cache-enabled throughput remains comparatively stable across the tested prompt range.
- No-cache throughput drops as prompt length increases.
- At long context, the throughput advantage of cache becomes large and easy to see.

Why it matters:

Latency and throughput tell the same story from different angles. This plot makes it clear that KV cache improves not just absolute runtime, but also sustained decoding efficiency.

### `prompt_length_vs_memory.png`

This figure compares peak GPU memory usage as prompt length increases.

What it shows:

- Memory usage rises with prompt length in both settings.
- Cache-enabled runs tend to use somewhat more memory.
- The added memory cost is modest compared with the long-context latency savings observed in the other plots.

Why it matters:

KV cache is not free. This plot shows the memory side of the tradeoff and helps justify the practical recommendation: on this setup, a small memory increase is often worth a large speed gain.

## Cache strategy comparison

### `cache_strategy_vs_latency.png`

This figure compares latency across cache modes over multiple prompt lengths.

What it shows:

- `dynamic` is the best overall default on the tested setup.
- `offloaded` stays competitive, especially when considering memory constraints.
- `no_cache` degrades much more severely at long prompts.
- `static` is absent from successful comparisons because it failed locally due to Triton/backend support issues.

Why it matters:

This plot turns the project from a simple cache-vs-no-cache study into a more practical systems comparison. It shows that implementation choice matters, not just whether cache is enabled.

### `cache_strategy_vs_throughput.png`

This figure compares tokens per second across the available cache strategies.

What it shows:

- `dynamic` generally provides the strongest throughput.
- `offloaded` is a reasonable fallback when memory pressure matters more than achieving the absolute fastest runtime.
- `no_cache` falls behind most clearly at longer prompts.

Why it matters:

This plot supports the practical recommendation in the project: use `dynamic` when memory is acceptable, and consider `offloaded` when VRAM is tighter.

### `cache_strategy_vs_memory.png`

This figure compares peak GPU memory across cache strategies.

What it shows:

- The working cache modes have measurable but not extreme memory differences on this benchmark.
- Memory differences are smaller than the most important latency differences.
- `offloaded` can remain attractive when trying to reduce pressure on device memory while preserving much of the decoding benefit.

Why it matters:

This plot helps frame the project as a tradeoff study rather than a pure speed contest. It shows why a slightly slower strategy may still be the right engineering choice in constrained environments.

## Decode-focused microbenchmark

### `decode_micro_total_time.png`

This figure compares total generation time for the decode-focused microbenchmark at prompt length `512` and generation length `64`.

What it shows:

- `dynamic` cache substantially reduces total generation time compared with `no_cache`.
- Even on a small local model, the long-context decode advantage is large.

Why it matters:

This is the most direct demonstration of the decode-side value of KV cache in the repo. It isolates a long-context setting where repeated reuse of past states has a visible effect.

### `decode_micro_decode_throughput.png`

This figure compares estimated decode throughput for the same microbenchmark.

What it shows:

- `dynamic` cache improves decode tokens/sec strongly relative to `no_cache`.
- The throughput gain aligns with the total-time reduction seen in the previous figure.

Why it matters:

This plot reinforces that the benefit is specifically tied to decoding efficiency, not just an incidental change in total runtime.

### `decode_micro_memory.png`

This figure compares peak GPU memory for the decode-focused microbenchmark.

What it shows:

- Cache-enabled decoding uses somewhat more memory than `no_cache`.
- The extra memory cost is small relative to the decode speedup in this benchmark.

Why it matters:

This figure completes the tradeoff picture for the microbenchmark: more memory is used, but the gain in decode performance is large enough to justify it in this scenario.

## Notes on Scope

- These figures were designed around an RTX 3060 Laptop GPU with `6 GB` VRAM.
- The decode benchmark was intentionally kept narrow to avoid unstable heavier runs on this machine.
- Error bars in the plots reflect trial standard deviation from repeated runs.
