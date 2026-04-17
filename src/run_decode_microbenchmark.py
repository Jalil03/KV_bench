import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
PROMPT_LENGTH = 512

# Keep this intentionally narrow on a 6 GB laptop GPU.
# This is a decode-focused microbenchmark, not a broad sweep.
GEN_LENGTHS = [64]

# Compare only the safest two modes here.
CACHE_MODES = [
    "no_cache",
    "dynamic",
]

NUM_WARMUP_RUNS = 1
NUM_TRIALS = 3

RAW_OUTPUT_PATH = Path("results/raw/decode_microbenchmark_raw.json")
AGG_OUTPUT_PATH = Path("results/raw/decode_microbenchmark_agg.json")


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def synchronize_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_gpu_memory_stats(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_std(values):
    if len(values) < 2:
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
    repeated = base_text * 500
    token_ids = tokenizer(
        repeated,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    clipped = token_ids[:target_length]
    return tokenizer.decode(clipped, skip_special_tokens=True)


def build_generate_kwargs(cache_mode: str, tokenizer, max_new_tokens: int) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if cache_mode == "no_cache":
        kwargs["use_cache"] = False
    elif cache_mode == "dynamic":
        kwargs["use_cache"] = True
    else:
        raise ValueError(f"Unknown cache mode: {cache_mode}")

    return kwargs


def build_forward_kwargs(cache_mode: str):
    if cache_mode == "no_cache":
        return {"use_cache": False}
    return {"use_cache": True}


def measure_prefill_time(model, tokenizer, prompt: str, device: str, cache_mode: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    synchronize_if_needed(device)
    start = time.perf_counter()

    with torch.no_grad():
        _ = model(**inputs, **build_forward_kwargs(cache_mode))

    synchronize_if_needed(device)
    end = time.perf_counter()

    return end - start, int(input_length)


def run_single_generation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    cache_mode: str,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    reset_gpu_memory_stats(device)

    prefill_time_sec, input_length = measure_prefill_time(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        cache_mode=cache_mode,
    )

    generate_kwargs = build_generate_kwargs(
        cache_mode=cache_mode,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    synchronize_if_needed(device)
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )

    synchronize_if_needed(device)
    end = time.perf_counter()

    total_generation_time_sec = end - start
    output_length = outputs.shape[1]
    generated_tokens = output_length - input_length

    decode_estimate_sec = max(total_generation_time_sec - prefill_time_sec, 0.0)
    decode_tokens_per_sec = (
        generated_tokens / decode_estimate_sec if decode_estimate_sec > 0 else 0.0
    )

    total_tokens_per_sec = (
        generated_tokens / total_generation_time_sec if total_generation_time_sec > 0 else 0.0
    )

    peak_memory_mb = get_peak_memory_mb(device)

    return {
        "prompt_length_tokens": int(input_length),
        "generated_tokens": int(generated_tokens),
        "prefill_time_sec": float(prefill_time_sec),
        "total_generation_time_sec": float(total_generation_time_sec),
        "decode_estimate_sec": float(decode_estimate_sec),
        "total_tokens_per_sec": float(total_tokens_per_sec),
        "decode_tokens_per_sec": float(decode_tokens_per_sec),
        "peak_gpu_memory_mb": float(peak_memory_mb) if peak_memory_mb is not None else None,
    }


def aggregate_trials(
    trials: list[dict],
    model_name: str,
    device: str,
    cache_mode: str,
    prompt_length: int,
    max_new_tokens: int,
) -> dict:
    prefill_times = [x["prefill_time_sec"] for x in trials]
    total_times = [x["total_generation_time_sec"] for x in trials]
    decode_times = [x["decode_estimate_sec"] for x in trials]
    total_tps = [x["total_tokens_per_sec"] for x in trials]
    decode_tps = [x["decode_tokens_per_sec"] for x in trials]
    memories = [x["peak_gpu_memory_mb"] for x in trials if x["peak_gpu_memory_mb"] is not None]
    generated_tokens = [x["generated_tokens"] for x in trials]

    return {
        "model_name": model_name,
        "device": device,
        "cache_mode": cache_mode,
        "prompt_length_tokens": prompt_length,
        "max_new_tokens_requested": max_new_tokens,
        "num_trials": len(trials),
        "generated_tokens_mean": round_or_none(safe_mean(generated_tokens), 3),
        "prefill_time_mean_sec": round_or_none(safe_mean(prefill_times), 6),
        "prefill_time_std_sec": round_or_none(safe_std(prefill_times), 6),
        "total_generation_time_mean_sec": round_or_none(safe_mean(total_times), 6),
        "total_generation_time_std_sec": round_or_none(safe_std(total_times), 6),
        "decode_estimate_mean_sec": round_or_none(safe_mean(decode_times), 6),
        "decode_estimate_std_sec": round_or_none(safe_std(decode_times), 6),
        "total_tokens_per_sec_mean": round_or_none(safe_mean(total_tps), 6),
        "total_tokens_per_sec_std": round_or_none(safe_std(total_tps), 6),
        "decode_tokens_per_sec_mean": round_or_none(safe_mean(decode_tps), 6),
        "decode_tokens_per_sec_std": round_or_none(safe_std(decode_tps), 6),
        "peak_gpu_memory_mb_mean": round_or_none(safe_mean(memories), 3),
        "peak_gpu_memory_mb_std": round_or_none(safe_std(memories), 3),
        "trials": [
            {
                "trial_index": idx + 1,
                "generated_tokens": t["generated_tokens"],
                "prefill_time_sec": round_or_none(t["prefill_time_sec"], 6),
                "total_generation_time_sec": round_or_none(t["total_generation_time_sec"], 6),
                "decode_estimate_sec": round_or_none(t["decode_estimate_sec"], 6),
                "total_tokens_per_sec": round_or_none(t["total_tokens_per_sec"], 6),
                "decode_tokens_per_sec": round_or_none(t["decode_tokens_per_sec"], 6),
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


def print_case_summary(case: dict) -> None:
    print(
        f"[decode_micro | gen={case['max_new_tokens_requested']:>3} | mode={case['cache_mode']:<9}] "
        f"prefill={case['prefill_time_mean_sec']:.4f}s | "
        f"total={case['total_generation_time_mean_sec']:.4f}s | "
        f"decode_est={case['decode_estimate_mean_sec']:.4f}s | "
        f"decode_tok/s={case['decode_tokens_per_sec_mean']:.4f} | "
        f"peak_mem={case['peak_gpu_memory_mb_mean']:.2f} MB"
    )


def main():
    device = get_device()
    dtype = get_dtype(device)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, device, dtype)
    prompt = make_prompt_for_target_length(tokenizer, PROMPT_LENGTH)

    raw_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "device": device,
            "experiment_name": "decode_microbenchmark",
            "prompt_length": PROMPT_LENGTH,
            "gen_lengths": GEN_LENGTHS,
            "cache_modes": CACHE_MODES,
            "num_trials": NUM_TRIALS,
            "num_warmup_runs": NUM_WARMUP_RUNS,
        },
        "cases": [],
    }

    aggregated_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "device": device,
            "experiment_name": "decode_microbenchmark",
            "prompt_length": PROMPT_LENGTH,
            "gen_lengths": GEN_LENGTHS,
            "cache_modes": CACHE_MODES,
            "num_trials": NUM_TRIALS,
            "num_warmup_runs": NUM_WARMUP_RUNS,
        },
        "cases": [],
    }

    for gen_length in GEN_LENGTHS:
        for cache_mode in CACHE_MODES:
            print(
                f"\\n=== Decode microbenchmark: gen_length={gen_length}, "
                f"cache_mode={cache_mode} ==="
            )

            trials = []

            for warmup_idx in range(NUM_WARMUP_RUNS):
                print(f"Warmup {warmup_idx + 1}/{NUM_WARMUP_RUNS} ...")
                _ = run_single_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=gen_length,
                    cache_mode=cache_mode,
                )

            for trial_idx in range(NUM_TRIALS):
                print(f"Trial {trial_idx + 1}/{NUM_TRIALS} ...")
                trial_result = run_single_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=gen_length,
                    cache_mode=cache_mode,
                )
                trials.append(trial_result)

            raw_case = {
                "model_name": MODEL_NAME,
                "device": device,
                "cache_mode": cache_mode,
                "prompt_length_tokens": PROMPT_LENGTH,
                "max_new_tokens_requested": gen_length,
                "num_trials": NUM_TRIALS,
                "num_warmup_runs": NUM_WARMUP_RUNS,
                "status": "ok",
                "trials": trials,
            }
            raw_results["cases"].append(raw_case)

            agg_case = aggregate_trials(
                trials=trials,
                model_name=MODEL_NAME,
                device=device,
                cache_mode=cache_mode,
                prompt_length=PROMPT_LENGTH,
                max_new_tokens=gen_length,
            )
            agg_case["status"] = "ok"
            aggregated_results["cases"].append(agg_case)
            print_case_summary(agg_case)

            # Cool down a bit between cases.
            print("Cooling down for 10 seconds...")
            time.sleep(10)

    save_json(raw_results, RAW_OUTPUT_PATH)
    save_json(aggregated_results, AGG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
