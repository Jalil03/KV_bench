## Key findings

### 1. KV cache becomes much more valuable at long context
Across prompt-length experiments, cache-enabled decoding stayed relatively stable while no-cache latency increased sharply at long prompts, especially around 512 tokens.

### 2. Dynamic cache is the strongest default on this setup
Among the working strategies, `dynamic` cache provided the best overall latency and throughput on the tested RTX 3060 Laptop GPU.

### 3. Offloaded cache is a useful fallback under memory pressure
`offloaded` cache stayed competitive at longer prompt lengths and can be attractive when GPU memory constraints matter more than absolute best latency.

### 4. No-cache breaks down in decode-heavy long-context inference
In the decode-focused microbenchmark at prompt length 512 and generated length 64, `dynamic` cache drastically reduced total generation time and improved decode throughput compared with `no_cache`.

### 5. Backend/toolchain support matters in practice
`static` cache could not be evaluated on the current local setup because of Triton/backend dependency issues, which is itself a useful deployment insight.