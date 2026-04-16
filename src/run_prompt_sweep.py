import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
PROMPT_LENGTHS = [16, 32, 64, 128, 256, 512]
MAX_NEW_TOKENS = 60
OUTPUT_PATH = Path("results/raw/prompt_sweep_results.json")


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
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model


def reset_gpu_memory_stats(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str):
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def make_prompt_for_target_length(tokenizer, target_length: int) -> str:
    base_text = "KV cache helps transformers decode faster by reusing past attention states. "
    repeated = base_text * 200
    tokens = tokenizer(repeated, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    clipped = tokens[:target_length]
    prompt = tokenizer.decode(clipped, skip_special_tokens=True)
    return prompt


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

    return {
        "model_name": MODEL_NAME,
        "device": device,
        "use_cache": use_cache,
        "prompt_length_tokens": input_length,
        "max_new_tokens_requested": max_new_tokens,
        "generated_tokens": generated_tokens,
        "total_latency_sec": round(total_latency, 6),
        "tokens_per_sec": round(tokens_per_sec, 6),
        "peak_gpu_memory_mb": round(peak_memory_mb, 3) if peak_memory_mb is not None else None,
    }


def save_results(results: list[dict], output_path: Path) -> None:
    ensure_output_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def main():
    device = get_device()
    dtype = get_dtype(device)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = load_model_and_tokenizer(MODEL_NAME, device, dtype)

    results = []

    for prompt_length in PROMPT_LENGTHS:
        prompt = make_prompt_for_target_length(tokenizer, prompt_length)

        for use_cache in [False, True]:
            print(f"Running | prompt_length={prompt_length} | use_cache={use_cache}")
            result = benchmark_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=use_cache,
            )
            results.append(result)
            print(result)

    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()