import torchvision
from torch import nn

# 现有模型的使用及修改
# 加载vgg训练好的模型，并在里边加入一个线性层
# 暂且不用ImageNet数据集，继续用CIFAR10
train_data = torchvision.datasets.CIFAR10("Code/dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 加载现有的vgg模型
vgg16_not_pretrain = torchvision.models.vgg16(pretrained=False)
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)

# 修改方法1：加入一个线性层,编号7
vgg16_pretrained.add_module("7", nn.Linear(1000, 10))
print(vgg16_pretrained)

# 修改方法2：修改原来的第六个线性层
vgg16_not_pretrain.classifier[6] = nn.Linear(4096, 10)
print(vgg16_not_pretrain)