import torch
import math
# input: 256
# output: 4
# layer: 1

weights = torch.randn(256, 4) / math.sqrt(256)
# print(weights.shape)
# ensure weigth is trainable
weights.requires_grad_()

# add the bias weigths for the 4-dimensional output, and make these trainable too
bias = torch.zeros(4, requires_grad=True)
