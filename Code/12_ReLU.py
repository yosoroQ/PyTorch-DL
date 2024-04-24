import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# _神经网络_非线性激活
# 向神经网络骨架中添加一个激活函数，并可视化结果

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


ocean = Ocean()
output = ocean(input)
print(output)

# # 输出
# torch.Size([1, 1, 2, 2])
# tensor([[[[1., 0.],
#           [0., 3.]]]])