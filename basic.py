import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], 
                         dtype=torch.float32, 
                         device=device, # specify device
                         requires_grad=True) # for gradient descent
# attribute of tensor
# print(my_tensor)
# print(my_tensor.device)
# print(my_tensor.dtype)
# print(my_tensor.shape) 
# print(my_tensor.requires_grad) 

# Other common initialization methods
x = torch.empty(size= (3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5) # identity metrics
x = torch.arange(start=0, end=5, step=1) # identity metrics
x = torch.linspace(start=0.1, end=1, steps=10) # identity metrics

x = torch.empty((1,5)).normal_(mean=0, std=1) # identity metrics
x = torch.empty((1,5)).uniform_(0, 1) # identity metrics
x = torch.diag(torch.ones(3)) # create 3x3 diagonal with 1
x = torch.diag(torch.ones(3)) # create 3x3 diagonal with 1

# Conversion
ts = torch.arange(4)
# print(ts.bool()) # boolean True/Fasle
# print(ts.short()) # int16
# print(ts.long())  # int64  almost always used
# print(ts.half())  # float16
# print(ts.float()) # float32
# print(ts.double()) # float64

# Array to Tensor conversion and vice-versa.
# import numpy as np
# np_array = np.zeros((5,5))
# tensor = torch.from_numpy(np_array)
# np_array_back = tensor.numpy()
# print(tensor)
# print(np_array_back)

# Tensor Math & Comparison Operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y
# Subtraction
z = x - y
# Division
z = torch.true_divide(x, y) # element-wise division

# inplace operation (followed by _)
t = torch.zeros(3) # it will replace and doesn't create a copy.
t.add_(x)
t += x # t = t + x; do inplace addition 

# Exponentiation
z = x.pow(2) # 1, 4, 9
z = x ** 2 # 1, 4, 9

# Simple Comparison
z = x > 0 # [True, True, True]

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # (2,3)
x3 = x1.mm(x2) # same above

# Matrix exponentiation
matrix_exp = torch.rand((5,5))
# print(matrix_exp)
# print(matrix_exp.matrix_power(3))

# element-wise mult.
z = x * y
# print(z) # 9, 16, 21

# dot product
z = torch.dot(x, y) 
# print(z) # 46

# Batch Matrix Multiplication
batch = 32
n = 2
m = 4
p = 6

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)

# print(tensor1)
# print(tensor2)
# print(out_bmm)

# Example of Broadcasting
x1 = torch.rand((5,5))
x1 = torch.rand((1,5))

# z = x1 - x2
# z = x1 ** x2

# Other useful operation
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)

z = torch.argmax(x, dim=0) # return index of max value
# print(z)
z = torch.argmin(x, dim=0)


mean_x = torch.mean(x.float(), dim=0)

sorted_y, indices = torch.sort(y, dim=0, descending=False)
# print(sorted_y)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
print(z)





