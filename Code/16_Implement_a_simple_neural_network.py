import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# （实现一个简单的神经网络）
class Ocean(nn.modules):
    def __init__(self):
        super(Ocean, self).__init__()
        # stride 默认为1 所以不写也可
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = self.flatten(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x
        
ocean = Ocean
print(ocean)

input = torch.ones((64, 3, 32, 32))
output = ocean(input)
print(output.shape)  # 输出output尺寸