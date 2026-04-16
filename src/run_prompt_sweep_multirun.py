import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
PROMPT_LENGTHS = [16, 32, 64, 128, 256, 512]
MAX_NEW_TOKENS = 60
NUM_TRIALS = 5
NUM_WARMUP_RUNS = 1

RAW_OUTPUT_PATH = Path("results/raw/prompt_sweep_multirun_raw.json")
AGG_OUTPUT_PATH = Path("results/raw/prompt_sweep_multirun_agg.json")


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def reset_gpu_memory_stats(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def synchronize_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_std(values):
    if values is None or len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def round_or_none(value, digits=6):
    if value is None:
        return None
    return round(value, digits)


def load_model_and_tokenizer(model_name: str, device: str, dtype):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def make_prompt_for_target_length(tokenizer, target_length: int) -> str:
    base_text = (
        "KV cache helps transformers decode faster by reusing past attention states. "
    )
    repeated = base_text * 400
    token_ids = tokenizer(
        repeated,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    clipped = token_ids[:target_length]
    prompt = tokenizer.decode(clipped, skip_special_tokens=True)
    return prompt


def run_single_generation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    use_cache: bool,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    reset_gpu_memory_stats(device)
    synchronize_if_needed(device)

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize_if_needed(device)
    end = time.perf_counter()

    total_latency = end - start
    output_length = outputs.shape[1]
    generated_tokens = output_length - input_length
    tokens_per_sec = generated_tokens / total_latency if total_latency > 0 else 0.0
    peak_memory_mb = get_peak_memory_mb(device)

    return {
        "prompt_length_tokens": int(input_length),
        "generated_tokens": int(generated_tokens),
        "total_latency_sec": float(total_latency),
        "tokens_per_sec": float(tokens_per_sec),
        "peak_gpu_memory_mb": float(peak_memory_mb) if peak_memory_mb is not None else None,
    }


def aggregate_trials(
    trials: list[dict],
    model_name: str,
    device: str,
    use_cache: bool,
    prompt_length: int,
    max_new_tokens: int,
) -> dict:
    latencies = [x["total_latency_sec"] for x in trials]
    throughputs = [x["tokens_per_sec"] for x in trials]
    memories = [x["peak_gpu_memory_mb"] for x in trials if x["peak_gpu_memory_mb"] is not None]
    generated_tokens = [x["generated_tokens"] for x in trials]

    return {
        "model_name": model_name,
        "device": device,
        "use_cache": use_cache,
        "prompt_length_tokens": prompt_length,
        "max_new_tokens_requested": max_new_tokens,
        "num_trials": len(trials),
        "generated_tokens_mean": round_or_none(safe_mean(generated_tokens), 3),
        "latency_mean_sec": round_or_none(safe_mean(latencies), 6),
        "latency_std_sec": round_or_none(safe_std(latencies), 6),
        "tokens_per_sec_mean": round_or_none(safe_mean(throughputs), 6),
        "tokens_per_sec_std": round_or_none(safe_std(throughputs), 6),
        "peak_gpu_memory_mb_mean": round_or_none(safe_mean(memories), 3),
        "peak_gpu_memory_mb_std": round_or_none(safe_std(memories), 3),
        "trials": [
            {
                "trial_index": idx + 1,
                "generated_tokens": t["generated_tokens"],
                "total_latency_sec": round_or_none(t["total_latency_sec"], 6),
                "tokens_per_sec": round_or_none(t["tokens_per_sec"], 6),
                "peak_gpu_memory_mb": round_or_none(t["peak_gpu_memory_mb"], 3),
            }
            for idx, t in enumerate(trials)
        ],
    }


def save_json(data, path: Path) -> None:
    ensure_output_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


def print_case_summary(agg_result: dict) -> None:
    print(
        f"[prompt={agg_result['prompt_length_tokens']:>4} | "
        f"use_cache={str(agg_result['use_cache']):>5}] "
        f"latency={agg_result['latency_mean_sec']:.4f}s ± {agg_result['latency_std_sec']:.4f} | "
        f"tok/s={agg_result['tokens_per_sec_mean']:.4f} ± {agg_result['tokens_per_sec_std']:.4f} | "
        f"peak_mem={agg_result['peak_gpu_memory_mb_mean']:.2f} MB ± {agg_result['peak_gpu_memory_mb_std']:.2f}"
    )


def main():
    device = get_device()
    dtype = get_dtype(device)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, device, dtype)

    raw_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "device": device,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_trials": NUM_TRIALS,
            "num_warmup_runs": NUM_WARMUP_RUNS,
            "prompt_lengths": PROMPT_LENGTHS,
        },
        "cases": [],
    }

    aggregated_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "device": device,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_trials": NUM_TRIALS,
            "num_warmup_runs": NUM_WARMUP_RUNS,
            "prompt_lengths": PROMPT_LENGTHS,
        },
        "cases": [],
    }

    for prompt_length in PROMPT_LENGTHS:
        prompt = make_prompt_for_target_length(tokenizer, prompt_length)

        for use_cache in [False, True]:
            print(f"\n=== Case: prompt_length={prompt_length}, use_cache={use_cache} ===")

            # Warmup
            for warmup_idx in range(NUM_WARMUP_RUNS):
                print(f"Warmup {warmup_idx + 1}/{NUM_WARMUP_RUNS} ...")
                _ = run_single_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=use_cache,
                )

            # Recorded trials
            trials = []
            for trial_idx in range(NUM_TRIALS):
                print(f"Trial {trial_idx + 1}/{NUM_TRIALS} ...")
                trial_result = run_single_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=use_cache,
                )
                trials.append(trial_result)

            raw_case = {
                "model_name": MODEL_NAME,
                "device": device,
                "use_cache": use_cache,
                "prompt_length_tokens": prompt_length,
                "max_new_tokens_requested": MAX_NEW_TOKENS,
                "num_trials": NUM_TRIALS,
                "num_warmup_runs": NUM_WARMUP_RUNS,
                "trials": trials,
            }
            raw_results["cases"].append(raw_case)

            agg_case = aggregate_trials(
                trials=trials,
                model_name=MODEL_NAME,
                device=device,
                use_cache=use_cache,
                prompt_length=prompt_length,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            aggregated_results["cases"].append(agg_case)
            print_case_summary(agg_case)

    save_json(raw_results, RAW_OUTPUT_PATH)
    save_json(aggregated_results, AGG_OUTPUT_PATH)


if __name__ == "__main__":
    main()