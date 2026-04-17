## Limitations

- Experiments were run on a laptop GPU with **6 GB VRAM**, so some heavier configurations were intentionally avoided.
- Thermal constraints affected experiment scope; one heavier benchmark design caused system instability and had to be reduced.
- `static` cache was unavailable on the current Windows/local backend due to Triton-related support issues.
- The prefill/decode separation used in the decode-focused microbenchmark is a practical timing proxy, not a kernel-level instrumentation method.
- Results are currently based on a small causal LM (`HuggingFaceTB/SmolLM2-360M`) rather than a larger production-scale model.