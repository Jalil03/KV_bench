# PROJECT_CHECKLIST.md

# KVBench

A reproducible benchmark suite for studying the latency, throughput, and memory tradeoffs of KV cache strategies in transformer inference.

---

## 1. Project Objective

- [ ] Finalize the project goal statement
- [ ] Keep the scope focused on KV-cache benchmarking for causal LMs
- [ ] Keep the project reproducible and portfolio-ready

**Goal statement**
> KVBench is a reproducible benchmark suite for studying the latency, throughput, and memory tradeoffs of KV cache strategies in transformer inference.

---

## 2. Core Questions

- [ ] How much does KV cache improve decoding speed?
- [ ] How does KV cache affect memory usage?
- [ ] How do prompt length and generation length affect the tradeoff?
- [ ] What changes across different cache strategies?

---

## 3. Phase 0 — Framing

### Tasks
- [ ] Choose final repo name
- [ ] Write the project objective in the README
- [ ] Define benchmark questions
- [ ] Define initial scope for v1
- [ ] Define hardware constraints
- [ ] Create the repo structure

### Done when
- [ ] The problem is clearly defined
- [ ] The benchmark scope is clear
- [ ] Out-of-scope items for v1 are written down

---

## 4. Phase 1 — Environment Setup

### Tasks
- [ ] Create GitHub repo
- [ ] Create Python virtual environment
- [ ] Install dependencies
  - [ ] torch
  - [ ] transformers
  - [ ] accelerate
  - [ ] pandas
  - [ ] matplotlib
- [ ] Check GPU detection
- [ ] Check model loading
- [ ] Check tokenization
- [ ] Check text generation on GPU

### Done when
- [ ] `python test_model.py` runs successfully
- [ ] A small causal LM generates text on the GPU

---

## 5. Phase 2 — Benchmark Core

### Tasks
- [ ] Load model and tokenizer
- [ ] Run generation with cache enabled
- [ ] Run generation with cache disabled
- [ ] Measure total latency
- [ ] Measure generated tokens
- [ ] Measure tokens/sec
- [ ] Measure peak GPU memory
- [ ] Save results to file

### Metrics to collect
- [ ] Model name
- [ ] Prompt length
- [ ] Generation length
- [ ] Batch size
- [ ] Cache mode
- [ ] Total latency
- [ ] Tokens generated
- [ ] Tokens/sec
- [ ] Peak memory

### Done when
- [ ] One script compares cache on vs off
- [ ] Output is saved in structured form

---

## 6. Phase 3 — Experiment System

### Tasks
- [ ] Define experiment configs
- [ ] Automate repeated runs
- [ ] Save all results to CSV or JSON
- [ ] Use consistent naming for runs
- [ ] Log failed runs clearly

### Done when
- [ ] One command can launch multiple benchmark cases
- [ ] Results are saved consistently
- [ ] Results load cleanly in pandas

---

## 7. Phase 4 — Cache Strategy Comparison

### Tasks
- [ ] Benchmark no cache
- [ ] Benchmark dynamic/default cache
- [ ] Benchmark static cache
- [ ] Benchmark offloaded cache
- [ ] Benchmark quantized cache if supported

### Done when
- [ ] A comparison table exists
- [ ] Plots compare the strategies
- [ ] Speed vs memory tradeoffs are interpretable

---

## 8. Phase 5 — Analysis and Visualization

### Tasks
- [ ] Build plotting scripts
- [ ] Plot prompt length vs latency
- [ ] Plot prompt length vs peak memory
- [ ] Plot generation length vs total time
- [ ] Plot cache mode vs tokens/sec
- [ ] Plot cache mode vs peak memory
- [ ] Write short interpretation notes for each plot

### Done when
- [ ] The figures tell a clear story
- [ ] Main findings are identified
- [ ] Suspicious results are investigated

---

## 9. Phase 6 — Professional Polish

### Tasks
- [ ] Write a strong README
- [ ] Explain motivation
- [ ] Explain methodology
- [ ] Document hardware and software setup
- [ ] Add example commands
- [ ] Add results section
- [ ] Add limitations section
- [ ] Add future work section

### Done when
- [ ] A recruiter or researcher can understand the repo quickly
- [ ] The project is reproducible
- [ ] The repo looks clean and professional

---

## 10. Suggested Repo Structure

```text
kvbench/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── src/
│   ├── run_benchmark.py
│   ├── metrics.py
│   ├── utils.py
│   └── prompts.py
├── experiments/
├── results/
│   ├── raw/
│   └── figures/
├── notebooks/
└── docs/
```

---

## 11. Task Board

### Backlog
- [ ] Choose final repo name
- [ ] Define benchmark questions
- [ ] Choose initial model
- [ ] Define experiment ranges
- [ ] Decide result file format

### In Progress
- [ ] Environment setup
- [ ] Minimal model test
- [ ] Benchmark runner v1

### Next
- [ ] Add no-cache vs cache comparison
- [ ] Add memory tracking
- [ ] Add CSV logging
- [ ] Add prompt-length sweep

### Later
- [ ] Add static cache
- [ ] Add offloaded cache
- [ ] Add quantized cache
- [ ] Add repeated-prefix experiment
- [ ] Add report and README polish

### Done
- [ ] Repo initialized
- [ ] Model runs on GPU
- [ ] First benchmark saved
- [ ] First plot created
- [ ] README draft created

---

## 12. Session Workflow

### Before each session
- [ ] Write today's goal
- [ ] Write how success will be checked
- [ ] Write which file(s) will be modified

### During the session
- [ ] Focus on one unit only
- [ ] Test immediately after each change
- [ ] Save outputs and logs

### End of session
- [ ] Write what was finished
- [ ] Write what blocked progress
- [ ] Write the next step

**Example**
```text
Finished:
- Loaded model on GPU
- Ran generation with use_cache=True
- Recorded total latency

Blocked:
- Peak memory tracking needs cleanup

Next:
- Add use_cache=False baseline
```

---

## 13. Validation Checkpoints

### Checkpoint 1 — Environment
- [ ] GPU detected
- [ ] Model loads without crashing
- [ ] Generation works

### Checkpoint 2 — Benchmark Runner
- [ ] Cache on/off both work
- [ ] Latency is measured
- [ ] Memory is measured
- [ ] Output is saved to file

### Checkpoint 3 — Experiment Sweeps
- [ ] Multiple runs automated
- [ ] Filenames are consistent
- [ ] Results are readable in pandas

### Checkpoint 4 — Analysis
- [ ] First plot generated
- [ ] Plots make sense
- [ ] Suspicious results investigated

### Checkpoint 5 — Portfolio Quality
- [ ] README understandable
- [ ] Commands reproducible
- [ ] Findings clearly written
- [ ] Repo looks clean

---

## 14. Immediate Next Actions

### Today
- [ ] Create repo structure
- [ ] Set up environment
- [ ] Run one small model successfully on GPU
- [ ] Create `docs/progress_log.md`

### After that
- [ ] Build benchmark runner v1
- [ ] Compare cache on vs off
- [ ] Save first result file
