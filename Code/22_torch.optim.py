# 使用随机梯度下降的优化器
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
ocean = Ocean()
optim = torch.optim.SGD(ocean.parameters(), lr = 0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = ocean(imgs)
        res_loss = loss(outputs, targets)
        optim.zero_grad() # 梯度清零
        res_loss.backward() # 反向传播求出每个点的梯度
        optim.step() # 对每个参数进行调优
        running_loss = running_loss + res_loss
    print(running_loss)