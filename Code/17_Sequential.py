import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# （实现一个简单的神经网络）搭建一个vgg神经网络
class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x
        
ocean = Ocean()
print(ocean)

input = torch.ones((64, 3, 32, 32))
output = ocean(input)
print(output.shape)  # 输出output尺寸

writer = SummaryWriter("Code/maxlogs")
writer.add_graph(ocean, input)
writer.close()