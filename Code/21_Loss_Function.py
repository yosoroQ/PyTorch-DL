import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# 再来看看如何在之前写的神经网络中用到Loss Function（损失函数）
dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

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
            Linear(64, 64)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
    
loss = nn.CrossEntropyLoss()
ocean = Ocean()
for data in dataloader:
    imgs, targets = data
    outputs = ocean(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print("ok")