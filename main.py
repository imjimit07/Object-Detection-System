import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpuname = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpuname}")
else:
    print("Using CPU")