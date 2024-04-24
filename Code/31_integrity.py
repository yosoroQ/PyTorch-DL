import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
import time

# 方法二：
# 定义训练的设备
device = torch.device("cuda:0")

train_data = torchvision.datasets.CIFAR10(root="Code/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="Code/data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 看一看训练集 测试集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
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

# 创建网络模型
ocean = Ocean()

# 方法二：
# 在特定位置加入.to(device)
ocean.to(device)


# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 方法二：
# 在特定位置加入.to(device)
loss_fn.to(device)

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(ocean.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 50

# 添加tensorboard
writer = SummaryWriter("Code/logs_train")
# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    ocean.train()
    for data in train_dataloader:
        imgs, targets = data

        # 方法二：
        # 在特定位置加入.to(device)
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = ocean(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # batch=64,训练集=5W，学习一边训练集就需要781.25次训练
        if total_train_step % 100 == 0:
            # 记录结束时间
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    ocean.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

           # 方法二：
    # 在特定位置加入.to(device)
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = ocean(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(ocean, "ocean_{}_pp.pth".format(i))
    print("模型已保存")

writer.close()