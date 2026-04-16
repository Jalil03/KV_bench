import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
PROMPT = "Explain in simple words what KV cache does in a transformer:"
MAX_NEW_TOKENS = 60
OUTPUT_PATH = Path("results/raw/benchmark_results.json")


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def load_model_and_tokenizer(model_name: str, device: str, dtype):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,   # use dtype instead of deprecated torch_dtype
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def reset_gpu_memory_stats(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str) -> float | None:
    if device != "cuda":
        return None
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)


def benchmark_generation(
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

    print(f"\nRunning benchmark | use_cache={use_cache}")

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_latency = end - start
    output_length = outputs.shape[1]
    generated_tokens = output_length - input_length
    tokens_per_sec = generated_tokens / total_latency if total_latency > 0 else 0.0
    peak_memory_mb = get_peak_memory_mb(device)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "model_name": MODEL_NAME,
        "device": device,
        "use_cache": use_cache,
        "prompt": prompt,
        "prompt_length_tokens": input_length,
        "max_new_tokens_requested": max_new_tokens,
        "generated_tokens": generated_tokens,
        "total_latency_sec": round(total_latency, 6),
        "tokens_per_sec": round(tokens_per_sec, 6),
        "peak_gpu_memory_mb": round(peak_memory_mb, 3) if peak_memory_mb is not None else None,
        "generated_text": generated_text,
    }

    return result


def save_results(results: list[dict], output_path: Path) -> None:
    ensure_output_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: list[dict]) -> None:
    print("\n=== Benchmark Summary ===")
    for result in results:
        print(f"\nuse_cache = {result['use_cache']}")
        print(f"Latency: {result['total_latency_sec']} s")
        print(f"Generated tokens: {result['generated_tokens']}")
        print(f"Tokens/sec: {result['tokens_per_sec']}")
        print(f"Peak GPU memory (MB): {result['peak_gpu_memory_mb']}")


def main():
    device = get_device()
    dtype = get_dtype(device)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, device, dtype)

    results = []

    for use_cache in [False, True]:
        result = benchmark_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=use_cache,
        )
        results.append(result)

    save_results(results, OUTPUT_PATH)
    print_summary(results)


if __name__ == "__main__":
    main()