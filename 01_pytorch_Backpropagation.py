import torch
import numpy as np

# Backpropagation
x = torch.randn(3, requires_grad=True)
print("x:", x)

y = x + 2
print("y:", y)

z = y * y * 2
print("z:", z)

# Define a vector for the backward pass
vector = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
print("vector:", vector)

# Backpropagation with the specified vector
z.backward(vector)
print("x.grad:", x.grad)
