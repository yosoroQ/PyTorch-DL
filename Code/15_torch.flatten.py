import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 向神经网络骨架中添加一个线性层，并可视化结果
# 特别注意这里在把图片放入线性层之前要用flatten把图片弄成一维的


# 使用CIFAR10测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("Code/dataset", train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)


class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

ocean = Ocean()
for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    # flatten
    output = torch.flatten(imgs)
    print(output.shape)

# 输出：torch.Size([64, 3, 32, 32])
# torch.Size([196608])
# torch.Size([64, 3, 32, 32])
# torch.Size([196608])
# torch.Size([64, 3, 32, 32])
# torch.Size([196608])
# 余下略
# torch.Size([196608])
# torch.Size([16, 3, 32, 32])
# torch.Size([49152])