# 神经网络 卷积层 向神经网络骨架中添加一个卷积层，并可视化查看卷积结果
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 向神经网络骨架中添加一个卷积层，并可视化查看卷积结果

# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

ocean = Ocean()
print(ocean)
writer = SummaryWriter('Code/logs')

step = 0
for data in dataloader:
    imgs, target = data
    output = ocean(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("before conv2d", imgs, step)
    # torch.Size([64, 6, 30, 30])
    # 因为channel是6，board不知道该怎么写入图片了，所以要reshape
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("after conv2d", output, step)
    step = step + 1

writer.close()

#输出结果
# Ocean(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )
# torch.Size([64, 3, 32, 32])
# torch.Size([64, 6, 30, 30])
# # 余下略
# torch.Size([16, 6, 30, 30])