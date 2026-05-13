import torch

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Number of GPUs found: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")