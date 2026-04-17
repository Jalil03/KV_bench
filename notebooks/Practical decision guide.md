## Practical cache selection guide

Based on the current experiments:

- **Use Dynamic cache** when GPU memory is acceptable and you want the best overall speed.
- **Use Offloaded cache** when VRAM is tighter and long-context inference still matters.
- **Avoid No cache** for long-context decoding workloads.
- **Do not assume Static cache will work locally** without confirming backend/toolchain support first.