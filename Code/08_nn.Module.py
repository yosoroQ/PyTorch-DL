# 神经网络 基本骨架
import torch
from torch import nn

# 两个骨头就是骨架:
# def __init__(self)
# def forward(self, input)

class Ocean(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

ocean = Ocean()
x = torch.tensor(1.0)
output = ocean(x)
print(output)