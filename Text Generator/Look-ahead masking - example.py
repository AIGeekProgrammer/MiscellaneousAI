import torch

# Create a tensor with random values
t = torch.randn(4,4)
print(t)

# Create diagonal mask with 0s on the upper triangle
t = torch.tril(t, diagonal=0)
print(t)

# Fill 0s with -1e9 (minus infinity)
t = torch.masked_fill(t, t == 0, -1e9)
print(t)

# Apply softmax
t = torch.softmax(t, dim=1)
print(t)

