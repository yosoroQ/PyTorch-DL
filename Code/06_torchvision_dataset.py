#  torchvision中数据集dataset的使用
#  transform和torchvision中数据集的联合使用

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 使用compose对数据集做transform操作

dataset_trans = transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 下载数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trans, download=True)

# 输出
writer = SummaryWriter('Code/logs')
for i in range(10):
    img, target = train_set[i]
    writer.add_image('test torchvison compose', img, i)

writer.close()