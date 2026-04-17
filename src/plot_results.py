import json
from pathlib import Path

import matplotlib.pyplot as plt


PROMPT_SWEEP_PATH = Path("results/raw/prompt_sweep_multirun_agg.json")
CACHE_STRATEGY_PATH = Path("results/raw/cache_strategy_sweep_agg.json")
DECODE_MICROBENCHMARK_PATH = Path("results/raw/decode_microbenchmark_agg.json")
LEGACY_DECODE_PATH = Path("results/raw/generation_length_sweep_agg.json")

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_decode_microbenchmark_path() -> Path:
    if DECODE_MICROBENCHMARK_PATH.exists():
        return DECODE_MICROBENCHMARK_PATH
    return LEGACY_DECODE_PATH


def save_plot(filename: str):
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_prompt_sweep():
    data = load_json(PROMPT_SWEEP_PATH)
    cases = data["cases"]

    no_cache = sorted(
        [c for c in cases if c["use_cache"] is False],
        key=lambda x: x["prompt_length_tokens"],
    )
    cached = sorted(
        [c for c in cases if c["use_cache"] is True],
        key=lambda x: x["prompt_length_tokens"],
    )

    x_no = [c["prompt_length_tokens"] for c in no_cache]
    x_ca = [c["prompt_length_tokens"] for c in cached]

    # Latency
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x_no,
        [c["latency_mean_sec"] for c in no_cache],
        yerr=[c["latency_std_sec"] for c in no_cache],
        marker="o",
        capsize=4,
        label="no_cache",
    )
    plt.errorbar(
        x_ca,
        [c["latency_mean_sec"] for c in cached],
        yerr=[c["latency_std_sec"] for c in cached],
        marker="o",
        capsize=4,
        label="cache",
    )
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Latency (s)")
    plt.title("Prompt length vs latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("prompt_length_vs_latency.png")

    # Tokens/sec
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x_no,
        [c["tokens_per_sec_mean"] for c in no_cache],
        yerr=[c["tokens_per_sec_std"] for c in no_cache],
        marker="o",
        capsize=4,
        label="no_cache",
    )
    plt.errorbar(
        x_ca,
        [c["tokens_per_sec_mean"] for c in cached],
        yerr=[c["tokens_per_sec_std"] for c in cached],
        marker="o",
        capsize=4,
        label="cache",
    )
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Tokens/sec")
    plt.title("Prompt length vs throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("prompt_length_vs_throughput.png")

    # Memory
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x_no,
        [c["peak_gpu_memory_mb_mean"] for c in no_cache],
        yerr=[c["peak_gpu_memory_mb_std"] for c in no_cache],
        marker="o",
        capsize=4,
        label="no_cache",
    )
    plt.errorbar(
        x_ca,
        [c["peak_gpu_memory_mb_mean"] for c in cached],
        yerr=[c["peak_gpu_memory_mb_std"] for c in cached],
        marker="o",
        capsize=4,
        label="cache",
    )
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Prompt length vs peak GPU memory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("prompt_length_vs_memory.png")


def plot_cache_strategy_sweep():
    data = load_json(CACHE_STRATEGY_PATH)
    cases = [c for c in data["cases"] if c.get("status") == "ok"]

    modes = sorted(set(c["cache_mode"] for c in cases))
    prompt_lengths = sorted(set(c["prompt_length_tokens"] for c in cases))

    grouped = {mode: [] for mode in modes}
    for mode in modes:
        mode_cases = sorted(
            [c for c in cases if c["cache_mode"] == mode],
            key=lambda x: x["prompt_length_tokens"],
        )
        grouped[mode] = mode_cases

    # Latency
    plt.figure(figsize=(8, 5))
    for mode in modes:
        x = [c["prompt_length_tokens"] for c in grouped[mode]]
        y = [c["latency_mean_sec"] for c in grouped[mode]]
        yerr = [c["latency_std_sec"] for c in grouped[mode]]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=mode)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Latency (s)")
    plt.title("Cache strategy vs latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("cache_strategy_vs_latency.png")

    # Tokens/sec
    plt.figure(figsize=(8, 5))
    for mode in modes:
        x = [c["prompt_length_tokens"] for c in grouped[mode]]
        y = [c["tokens_per_sec_mean"] for c in grouped[mode]]
        yerr = [c["tokens_per_sec_std"] for c in grouped[mode]]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=mode)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Tokens/sec")
    plt.title("Cache strategy vs throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("cache_strategy_vs_throughput.png")

    # Memory
    plt.figure(figsize=(8, 5))
    for mode in modes:
        x = [c["prompt_length_tokens"] for c in grouped[mode]]
        y = [c["peak_gpu_memory_mb_mean"] for c in grouped[mode]]
        yerr = [c["peak_gpu_memory_mb_std"] for c in grouped[mode]]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=mode)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Cache strategy vs peak GPU memory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("cache_strategy_vs_memory.png")


def plot_decode_microbenchmark():
    data = load_json(resolve_decode_microbenchmark_path())
    cases = [c for c in data["cases"] if c.get("status") == "ok"]

    # sort by cache mode for stable bars
    cases = sorted(cases, key=lambda x: x["cache_mode"])
    labels = [c["cache_mode"] for c in cases]

    # Total generation time
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [c["total_generation_time_mean_sec"] for c in cases],
        yerr=[c["total_generation_time_std_sec"] for c in cases],
        capsize=6,
    )
    plt.xlabel("Cache mode")
    plt.ylabel("Total generation time (s)")
    plt.title("Decode microbenchmark: total time")
    plt.grid(True, axis="y", alpha=0.3)
    save_plot("decode_micro_total_time.png")

    # Decode throughput
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [c["decode_tokens_per_sec_mean"] for c in cases],
        yerr=[c["decode_tokens_per_sec_std"] for c in cases],
        capsize=6,
    )
    plt.xlabel("Cache mode")
    plt.ylabel("Decode tokens/sec")
    plt.title("Decode microbenchmark: decode throughput")
    plt.grid(True, axis="y", alpha=0.3)
    save_plot("decode_micro_decode_throughput.png")

    # Memory
    plt.figure(figsize=(7, 5))
    plt.bar(
        labels,
        [c["peak_gpu_memory_mb_mean"] for c in cases],
        yerr=[c["peak_gpu_memory_mb_std"] for c in cases],
        capsize=6,
    )
    plt.xlabel("Cache mode")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Decode microbenchmark: peak memory")
    plt.grid(True, axis="y", alpha=0.3)
    save_plot("decode_micro_memory.png")


def main():
    if PROMPT_SWEEP_PATH.exists():
        plot_prompt_sweep()
    else:
        print(f"Missing file: {PROMPT_SWEEP_PATH}")

    if CACHE_STRATEGY_PATH.exists():
        plot_cache_strategy_sweep()
    else:
        print(f"Missing file: {CACHE_STRATEGY_PATH}")

    decode_path = resolve_decode_microbenchmark_path()
    if decode_path.exists():
        plot_decode_microbenchmark()
    else:
        print(f"Missing file: {DECODE_MICROBENCHMARK_PATH}")


if __name__ == "__main__":
    main()
