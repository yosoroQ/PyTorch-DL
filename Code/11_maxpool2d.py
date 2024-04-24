import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 最大池化操作
# 向神经网络骨架中添加一个池化层，并可视化查看池化结果
# 这一部分代码和上边几乎一模一样，需要注意的是，池化层必须直接作用在float数据类型上，所以如果使用torch.tensor的话，就要加上dtype=float32，然后同样还要reshape为四维tensor

# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


ocean = Ocean()

writer = SummaryWriter("Code/maxlogs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = ocean(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()