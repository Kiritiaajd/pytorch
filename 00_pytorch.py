import torch

# Check if PyTorch is working
print("PyTorch version:", torch.__version__)

# Check if CUDA is available (it should be False)
print("CUDA available:", torch.cuda.is_available())
