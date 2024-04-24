import torch
from torch.nn import L1Loss
from torch import nn

# L1Loss
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets  = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape()添加维度，原来tensor是二维
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)

# 输出
# tensor(0.6667)