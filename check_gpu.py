import sys
import platform
import torch

print("=== System ===")
print("Python:", sys.version)
print("Platform:", platform.platform())

print("\n=== PyTorch ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by torch:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"Total VRAM: {total_vram_gb:.2f} GB")

    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    torch.cuda.synchronize()

    print("GPU tensor test: OK")
    print("Result shape:", tuple(z.shape))
else:
    print("No CUDA GPU detected by PyTorch.")