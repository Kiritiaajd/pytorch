import torch
import torch.nn as nn

# Define the input and output tensors
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

# Get the number of samples and features
n_samples, n_features = X.shape
print(n_samples, n_features)

# Define the model
input_size = n_features
output_size = n_features


# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define our layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)
# Print the prediction before training
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Define the learning rate and number of iterations
learning_rate = 0.01
n_iters = 100

# Define the loss function and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # Prediction: forward pass
    y_pred = model(X)

    # Loss
    l = loss(y_pred, Y)

    # Gradient: backward pass
    l.backward()  # dl/dw

    # Update weights
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

# Print the prediction after training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
