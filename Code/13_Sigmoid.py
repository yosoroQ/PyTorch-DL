import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# _神经网络_非线性激活
# 向神经网络骨架中添加一个激活函数，并可视化结果

# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

# input = torch.tensor([[1, -0.5],
#                      [-1, 3]])

# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        # self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


ocean = Ocean()
writer = SummaryWriter("Code/maxlogs")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input_Sigmoid", imgs, global_step=step)
    output = ocean(imgs)
    writer.add_images("output_Sigmoid", output, global_step=step)
    step = step+1

writer.close()