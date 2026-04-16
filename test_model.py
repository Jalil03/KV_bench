import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)

model.to(device)
model.eval()

prompt = "Explain in simple words what KV cache does in a transformer:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating...")
start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        use_cache=True,
    )
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Output ===")
print(text)
print(f"\nGeneration time: {elapsed:.3f} s")