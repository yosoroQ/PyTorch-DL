import torch
from torch.nn import L1Loss
from torch import nn

# MSELoss
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets  = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape()添加维度，原来tensor是二维
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# MSELoss 均方损失函数
loss_mse = nn.MSELoss(reduction="sum")
result_mse = loss_mse(inputs, targets)
print(result_mse)

# 输出 #均方误差损失函数计算结果
# tensor(4.)