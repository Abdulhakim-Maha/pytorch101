import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x[0, :]

print(x[:, 0].shape)

print(x[2, 0:10])
print(x[2, :10]) # 0:10 -> [0,1,2,3,...,9]

# more advances indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])

# Useful operation
print(torch.where(x > 5, x, x * 2)) # if x > 5, then assign to x, otherwise x * 2

print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())
print(x.numel()) # num  elements