# PyTorch深度学习

* 本文基于AutoDL构建PyTorch环境，所以安装环节略过。
* AutoDL官网：https://www.autodl.com/market/list
* 恒源云官网：https://www.gpushare.com
  * ~~这个比较便宜，同时比较多空闲的GPU，但缺点不太稳定，不保证睡一觉过后还能不能开机，关机前建议备份数据。~~
  * ~~经常出现“无空闲显卡”~~
* 代码参考（我是土堆）：
  * https://www.bilibili.com/video/av74281036/
  * https://gitcode.net/weixin_43453218/tudui_pytorch
  * https://github.com/xiaotudui/PyTorch-Tutorial
* 相关文档：
  * 深入浅出PyTorch（https://datawhalechina.github.io/thorough-pytorch/）
  * PyTorch中文文档（https://pytorch-cn.readthedocs.io/zh/latest/）
  * PyTorch 学习笔记（https://pytorch.zhangxiann.com/8-shi-ji-ying-yong）
  * 动手学深度学习（https://zh-v2.d2l.ai/）

# 00_Vscode连接AutoDL服务器

* 在[AutoDL](https://www.autodl.com/login)租用并开机实例，获取实例的SSH登录信息（登录指令和登录密码）
* 本地安装VSCode远程开发插件（需配置Remote-SSH）
* SSH连接并登录您远端租用的实例
* 详细文档：https://www.autodl.com/docs/vscode/

# 01_Dataset类代码实战

## 新建项目test & read_data.py

### 项目目录

![image-20240404224704823](http://qny.expressisland.cn/gdou24/image-20240404224704823.png)

### read_data.py

* 类中`__init__(self)`初始化函数

```python
# 01_DataSet类讲解和代码实战
import os
from torch.utils.data import Dataset
import torchvision
from PIL import Image

class MyData(Dataset):
    
    # self就是把root dir变成一个class中全部def都可以使用的全局变量
    def __init__(self):
        self.root.dir = root_dir
        self.label.dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    # 对MyData对象使用索引操作就会自动来到这个函数下边，双下划线是python中的魔法函数
    def __getitem__(self, idx):
        img.name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        laber = self.label_dir
        return img, label

    # 再写一个获取数据集长度的魔法函数
    def __len__(self)
        return len (self.img_path)

# 获取蚂蚁数据集dataset
root_dir = "dataset/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)

# 获取蜜蜂的数据集
root_dir = "./dataset/train"
label_dir = "bees"
bees_dataset = MyData(root_dir, label_dir)

# dataset数据集拼接
train_dataset = ants_dataset + bees_dataset
```

# 02_TensorBoard的使用

## `add scalar()`

* `add scalar()`常用来绘制`train/valloss`。
* `Summarywrite`类：`SummaryWriter`类提供了一个高级API来创建一个事件文件，在给定的目录中添加摘要和事件。
* `add_image()`方法：在事件文件中添加图片（本次加载的图片为PIL类型，不符合类型要求，所以要转换，可用OpenCV或numpy直接转换）

```python
# TensorBoard的使用
from torch.utils.tensorboard import SummaryWriter
 
writer = SummaryWriter("02testTB/logs")
for i in range(100):
    writer.add_scalar("y=2x",2*i,i) #标签y=2x，y轴2*i，x轴i

writer.close()
```

### 在终端中输入

```shell
python 02testTB/testTB.py

# logdir为logs的目录路径，port端口号
tensorboard --logdir=02testTB/logs --port=6007
```

### 打开AutoDL自带的TensorBoard

* 端口号需为**“6007”**

![image-20240404233645811](http://qny.expressisland.cn/gdou24/image-20240404233645811.png)

## 写图片数据

```python
# 写图片数据
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("01test/logs")

image_path = "01test/dataset/train/ants_image/0013035.jpg"
img_pil = Image.open(image_path)
img_array = np.array(img_pil)

# writer.add_image("img test", img_array, 1)
writer.add_image("img test", img_array, 1, dataformats='HWC')
writer.close()
```

![image-20240404234912340](http://qny.expressisland.cn/gdou24/image-20240404234912340.png)

# 03_Transforms的使用

## 如何使用Transform
![image-20240406115158570](http://qny.expressisland.cn/gdou24/image-20240406115158570.png)
## 为什么需要tensor数据类型

* 因为`tensor`包含了一些属性是计算神经网络是必不可少的。
  * `grad`：梯度
  * `device`：设备
  * `is CUDA`：
  * `requires grad`：保留梯度

```python
tensor_img.grad = 0
tensor_img.requires_grad = False
```

## transforms.py

```python
# transforms的使用
# transform是一个py文件，其中tosensor compose normalize是很常用的操作
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tosensor简单使用 

# 获取pil类型图片
img_path = "01test/dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# 创建需要的transforms工具，并给工具起名字
tensor_trans = transforms.ToTensor()
# 使用工具
tensor_img = tensor_trans(img)
print(tensor_img)

# demo3:使用tensor数据类型写入board
writer = SummaryWriter("01test/logs")
writer.add_image('tensor img', tensor_img, 1)
writer.close()
```

# 04_常见的Transforms

​	***备注：2024.04.05 autodl大白天服务器没几个空闲的，换了个恒源云（https://gpushare.com/center/hire）也不错***

​	**话说恒源云的如果运行tensorboard显示`“AttributeError: module 'distutils' has no attribute 'version'”`，那么可以在终端中输入`pip install setuptools==58.0.4`。**

* 其实就是更好的使用`transform`中各种各样的类。

![image-20240405170902898](http://qny.expressisland.cn/gdou24/image-20240405170902898.png)

## `ToTensor`的使用与`Normalize`(归一化)的使用
```python
# 常见的transforms
# ToTensor的使用与Normalize(归一化)的使用
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("test/logs")
img = Image.open("test/train/ants_image/0013035.jpg")
print(img)

# ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
write.add_image("ToTensor",img_tensor)
write.close()

# Normalize(归一化)的使用
# 计算方法：output[channel] = (input[channel] - mean[channel]) / std[channel]
# 说人话：该像素上的值减去均值，再除以方差

print(img_tensor[0][0][0])
tran_norm = transforms.Normalize([1, 3, 5], [0.5, 0.5, 0.5])
img_norm = tran_norm(img_tensor)
print(img_norm[0][0][0])
write.add_image("Normalize", img_norm, 1)
write.close()
```
![image-241](http://qny.expressisland.cn/gdou24/241.png)
## 归一化公式
* 该像素上的值减去均值，再除以方差。
![image-242](http://qny.expressisland.cn/gdou24/242_%E5%BD%92%E4%B8%80%E5%8C%96%E7%9A%84%E5%85%AC%E5%BC%8F.png)

## `Resize`、`Compose`与`RanodmCrop`的使用
```python
# 常见的transforms
# Resize、Compose与RanodmCrop的使用
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("test/logs")
img = Image.open("test/train/ants_image/0013035.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
write.add_image("ToTensor",img_tensor)
write.close()

# Resize 拉伸
# Resize the input image to the given size.
# 注意如果给了一个int就是变为正方形，给（H，W）才是H W
# resize不会改变图片的数据类型
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
write.add_image("Resize", img_resize, 0)
print(img_resize)
write.close()

# Compose resize - 2 等比缩放
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
write.add_image("Resize_2", img_resize_2, 1)
write.close()

# RanodmCrop 随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    write.add_image("RandomCrop", img_crop, i)
write.close()
```

![image-243](http://qny.expressisland.cn/gdou24/243_Resize%E3%80%81Compose%E4%B8%8ERanodmCrop.png)

# 05_torchivision中的数据集的使用
* 可以下载一些数据集。
* torchivision地址：https://pytorch.org/vision/stable/index.html

## `transform`和`torchvision`中数据集的联合使用
### 下载CIFAR-100数据集
* CIFAR-100数据集：https://www.cs.toronto.edu/~kriz/cifar.html
* 或者在代码中直接拉取也可：
```python
import torchvision

train_set = torchvision.datasets.CIFAR10(root="test/dataset1", train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="test/dataset1", train=True,download=True)
```
### 完整代码
```python
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
writer = SummaryWriter('01test/logs')
for i in range(10):
    img, target = train_set[i]
    writer.add_image('test torchvison compose', img, i)

writer.close()
```
### 结果：一只32*32的猫和其他图片
* 循环了10次，就有10张图片，在TensorBoard中往右划还有其他几张图片。
* 这是第0张：
* ![image-244](http://qny.expressisland.cn/gdou24/244_transform%E5%92%8Ctorchvision.png)

# 06_`torchvision`中`DataLoader`的使用
* `torch.utils.data.dataloader`文档：https://pytorch.org/vision/stable/datasets.html?highlight=dataloader

## `DataLoader`各项参数详解
* `batch size`：loader能每次装弹4枚进入枪膛，或者理解每次抓四张牌；
* `shuffle`：每次epoch是否打乱原来的顺序，就像打完一轮牌后，洗不洗牌；
* `drop last`：最后的打他不够一个batch，还要不要了。

## 代码
```python
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# torchvision中dataloader的使用

# 测试集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# batch size：loader能每次装弹4枚进入枪膛，或者理解每次抓四张牌
# shuffle：每次epoch是否打乱原来的顺序，就像打完一轮牌后，洗不洗牌
# drop last：最后的打他不够一个batch 还要不要了

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("test/logs")

for epoch in range(2):
    step = 0
    for data in test_loader: 
      # batch_size=4的含义是以4为一组进行打包 shuffle是是否进行打乱 drop_last是最后剩余的余数是否进行舍去
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()
```

## 运行
* 数据集比较大，所以会比较慢，稍安勿躁。
* ![image-245](http://qny.expressisland.cn/gdou24/245_DataLoader.png)

# 07_神经网络的基本骨架_`nn.Module`的使用
* `TORCH.NN`文档地址：https://pytorch.org/docs/stable/nn.html#module-torch.nn

## 工作流程
* ![image-246](http://qny.expressisland.cn/gdou24/246_nn.png)

* 两个骨头就是骨架:
  * `def __init__(self)`
  * `def forward(self, input)`

## 代码
```python
# 神经网络 基本骨架
import torch
from torch import nn

# 两个骨头就是骨架:
# def __init__(self)
# def forward(self, input)

class Ocean(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

ocean = Ocean()
x = torch.tensor(1.0)
output = ocean(x)
print(output)
```

# 08_神经网络_卷积操作

* 理解什么是卷积操作，怎么算卷积结果。
* `TORCH.NN.FUNCTIONAL.CONV2D`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d

## `TORCH.NN.FUNCTIONAL.CONV2D`的一些参数Parameters

![image-20240406154739590](http://qny.expressisland.cn/gdou24/image-20240406154739590.png)

## 使用`TORCH.NN.FUNCTIONAL.CONV2D`

* `input kernel`：都是四维；

* `stride`：步长；

* `padding`：如果步长是1，又想保持输入输出高宽不变，就把`padding`设置1。
* 输入数据必须是四维的，所以用`torch.reshape`改变维度，`bias`是偏置，`weight`是卷积核，`stride`是步径，`padding`是边距。

```python
# 神经网络 卷积操作
# 理解什么是卷积操作 怎么算卷积结果
import torch
import torch.nn.functional as F

"""
使用conv2d
input kernel：都是四维
stride：步长
padding：如果步长是1，又想保持输入输出高宽不变，就把padding设置1
"""
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# conv2d需要输入的tensor是四维的（batch， c，h，w），但是现在的input kernel是二维
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)
print(output)
# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])

output2 = F.conv2d(input, kernel, stride=1, padding=1)
print(output2)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])
```

# 09_神经网络_卷积层

* 卷积后的输出计算方式是：输入图像和卷积核的对应位相乘再相加。

![image-20240406154103510](http://qny.expressisland.cn/gdou24/image-20240406154103510.png)

## `CONV2D`的一些参数Parameters

![image-20240406154625476](http://qny.expressisland.cn/gdou24/image-20240406154625476.png)

## `CONV2D`常用的前五个参数

* `CONV2D`常用的前五个参数，输入通道数，输出通道数，卷积核的大小，步径，边距。

* `CONV2D`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d

![image-20240406160141676](http://qny.expressisland.cn/gdou24/image-20240406160141676.png)

## 输入通道数为1，输出通道数为2的时候，卷积核应该有两个

![image-20240406163356660](http://qny.expressisland.cn/gdou24/image-20240406163356660.png)

## 一个重要的计算，维持图像的尺寸

* 看论文时可能会用到。

![image-20240406163451929](http://qny.expressisland.cn/gdou24/image-20240406163451929.png)

## 向神经网络骨架中添加一个卷积层，并可视化查看卷积结果

```python
# 神经网络 卷积层 向神经网络骨架中添加一个卷积层，并可视化查看卷积结果
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 向神经网络骨架中添加一个卷积层，并可视化查看卷积结果

# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False, transform=torchvision.transforms.ToTensor())
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
writer = SummaryWriter('test/logs')

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
```

### 输出结果

```python
#输出结果
Ocean(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
)
torch.Size([64, 3, 32, 32])
torch.Size([64, 6, 30, 30])
# 余下略
torch.Size([16, 6, 30, 30])
```

#### after and before conv2d

![image-20240406163108670](http://qny.expressisland.cn/gdou24/image-20240406163108670.png)

# 10_神经网络_池化层_最大池化的使用

* `MAXPOOL2D`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

* 作用：降低参数的数量，但保持输入数据的主要特征。

## `MAXPOOL2D`的一些参数Parameters

![image-20240406164145678](http://qny.expressisland.cn/gdou24/image-20240406164145678.png)

## 公式

![image-20240406164604329](http://qny.expressisland.cn/gdou24/image-20240406164604329.png)

## 最大池化操作

![image-20240406164741148](http://qny.expressisland.cn/gdou24/image-20240406164741148.png)

## 代码

* `ceil mode`：池化核走出input时还要不要里边的最大值 默认不要

```python
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
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


ocean = Ocean()

writer = SummaryWriter("test/maxlogs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = ocean(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```

## 测试 - 经过池化的图像（模糊）

![image-20240406170139079](http://qny.expressisland.cn/gdou24/image-20240406170139079.png)

# 11_神经网络_非线性激活

* 如果神经元的输出是输入的线性函数，而线性函数之间的嵌套任然会得到线性函数。
* 如果不加非线性函数处理，那么最终得到的仍然是线性函数，所以需要在神经网络中引入非线性激活函数。
* 常见的非线性激活函数主要包括**Sigmoid函数、tanh函数、ReLU函数、Leaky ReLU函数**，这几种非线性激活函数的介绍在神经网络中重要的概念（超参数、激活函数、损失函数、学习率等）中有详细说明。
* ReLU函数处理自然语言效果更佳，Sigmoid函数处理图像效果更佳

* 目的：在网络中引入更多的非线性特征。

* `RELU`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

![image-20240406170653969](http://qny.expressisland.cn/gdou24/image-20240406170653969.png)

* `inplace`含义：是否替换原数据。

## 代码（ReLU函数处理）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# _神经网络_非线性激活
# 向神经网络骨架中添加一个激活函数，并可视化结果

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


ocean = Ocean()
output = ocean(input)
print(output)

# # 输出
# torch.Size([1, 1, 2, 2])
# tensor([[[[1., 0.],
#           [0., 3.]]]])
```

## ReLU函数处理自然语言效果尤佳，但Sigmoid函数处理图像效果更佳

* `Sigmoid`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

### 代码（Sigmoid函数处理）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# _神经网络_非线性激活
# 向神经网络骨架中添加一个激活函数，并可视化结果

# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False,transform=torchvision.transforms.ToTensor())
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
writer = SummaryWriter("test/maxlogs")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input_Sigmoid", imgs, global_step=step)
    output = ocean(imgs)
    writer.add_images("output_Sigmoid", output, global_step=step)
    step = step+1

writer.close()
```

### 运行测试

![image-20240407170932090](http://qny.expressisland.cn/gdou24/image-20240407170932090.png)

# 12_神经网络_线性层及其他层介绍

![image-20240407171819635](http://qny.expressisland.cn/gdou24/image-20240407171819635.png)

## `LINEAR`

* `LINEAR`文档地址：https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

![image-20240407172011426](http://qny.expressisland.cn/gdou24/image-20240407172011426.png)

### 代码（`LINEAR`）

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 向神经网络骨架中添加一个线性层，并可视化结果
# 特别注意这里在把图片放入线性层之前要用flatten把图片弄成一维的


# 使用CIFAR10测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)


class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

# 报错是正常的，RuntimeError
ocean = Ocean()
for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = ocean(output)
    print(output.shape)
```

## 改为`TORCH.FLATTEN`

* 类似“平铺”。
* `TORCH.FLATTEN`文档地址：https://pytorch.org/docs/stable/generated/torch.flatten.html

### 代码

```python
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 向神经网络骨架中添加一个线性层，并可视化结果
# 特别注意这里在把图片放入线性层之前要用flatten把图片弄成一维的


# 使用CIFAR10测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False,transform=torchvision.transforms.ToTensor())
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
```

## Pytorch提供的网络模型

### 图像方面：`TORCHVISION.MODELS`

* `TORCHVISION.MODELS`文档地址：https://pytorch.org/vision/0.9/models.html
* ![image-20240407201747997](http://qny.expressisland.cn/gdou24/image-20240407201747997.png)

# 13_神经网络_搭建简易网络模型和Sequential的使用

## 神经网络图

![image-20240407205810428](http://qny.expressisland.cn/gdou24/image-20240407205810428.png)

* 输入图像是3通道的32×32的，先后经过卷积层（5×5的卷积核）、最大池化层（2×2的池化核）、卷积层（5×5的卷积核）、最大池化层（2×2的池化核）、卷积层（5×5的卷积核）、最大池化层（2×2的池化核）、拉直、全连接层的处理，最后输出的大小为10。
* 注：
  * 通道变化时通过调整卷积核的个数（即输出通道）来实现的，再`nn.conv2d`的参数中有`out_channel`这个参数就是对应输出通道；
  * 32个355的卷积核，然后input对其一个个卷积得到32个32 * 32（通道数变不变看用几个卷积核）；
  * 最大池化不改变`通道channel`数。

### 搭建一个简易网络模型（未使用Sequential）

```python
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
```

## 使用Sequential搭建简易网络模型

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# （实现一个简单的神经网络）搭建一个vgg神经网络
class Ocean(nn.Module):
    def __init__(self):
        super(Ocean, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x
        
ocean = Ocean()
print(ocean)

input = torch.ones((64, 3, 32, 32))
output = ocean(input)
print(output.shape)  # 输出output尺寸

writer = SummaryWriter("test/maxlogs")
writer.add_graph(ocean, input)
writer.close()
```

### 测试

![png](http://qny.expressisland.cn/gdou24/png.png)

#### 还能再大

![image-20240407215234532](http://qny.expressisland.cn/gdou24/image-20240407215234532.png)

# 14_神经网络_损失函数与反向传播

## 损失函数的作用

* `LossFunction`（损失函数）的作用：
  * 计算实际输出和目标之间的差距；
  * 为我们更新输出提供一定的依据（反向传播）。

* **`TORCH.NN.LossFunction`链接**：https://pytorch.org/docs/1.8.1/nn.html

## 损失函数的使用

### L1Loss

* **`L1LOSS`链接**：https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html

* 注：

  * `reduction = “sum”` 表示求和；

  * `reduction = "mean"` 表示求平均值，默认求平均值。

![image-20240409203133766](http://qny.expressisland.cn/gdou24/image-20240409203133766.png)

#### 代码（L1Loss）

* 代码计算了实际输出`[1, 2, 3]`和目标输出`[1, 2, 5]`之间的L1Loss。

```python
import torch
from torch.nn import L1Loss
from torch import nn

# L1Loss
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets  = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape()添加维度，原来tensor是二维
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)

# 输出
# tensor(0.6667)
```

### MSELOSS（均方损失函数）

* MSELOSS（均方损失函数）：可以设置reduction参数来决定具体的计算方法

* **`MSELOSS`链接**：https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

![image-20240409204047084](http://qny.expressisland.cn/gdou24/image-20240409204047084.png)

#### 代码（MSELOSS）

```python
import torch
from torch.nn import L1Loss
from torch import nn

# MSELoss
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets  = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape()添加维度，原来tensor是二维
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# MSELoss 均方损失函数
loss_mse = nn.MSELoss(reduction="sum")
result_mse = loss_mse(inputs, targets)
print(result_mse)

# 输出 #均方误差损失函数计算结果
# tensor(4.)
```

### CrossEntropyLoss（交叉熵损失函数）

* **`CROSSENTROPYLOSS`链接**：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

![image-20240409204648648](http://qny.expressisland.cn/gdou24/image-20240409204648648.png)

#### 代码（CrossEntropyLoss）

```python
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# CrossEntropyLoss交叉熵损失函数
# reshape
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# CrossEntropyLoss
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

# 输出
# tensor(1.1019)
```

## 再来看看如何在之前写的神经网络中用到Loss Function（损失函数）

```python
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# 再来看看如何在之前写的神经网络中用到Loss Function（损失函数）
dataset = torchvision.datasets.CIFAR10("test/dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
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
```

# 15_优化器

## TORCH.OPTIM

* **`TORCH.OPTIM`链接**：https://pytorch.org/docs/stable/optim.html#module-torch.optim
* 这个网站讲的挺详细的（深入浅出PyTorch）：https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%89%E7%AB%A0/3.9%20%E4%BC%98%E5%8C%96%E5%99%A8.html

## 使用随机梯度下降的优化器

```python
# 使用随机梯度下降的优化器
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("test/dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
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
```

# 16_现有网络模型的使用及修改

## VGG16简介

* **`vgg16`文档链接**：https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html

*  VGG16网络是14年牛津大学计算机视觉组和Google DeepMind公司研究员一起研发的深度网络模型，该网络一共有16个训练参数的网络。
* 该网络主要用于对 224 x 224 的图像进行 1000 分类。

### 该网络的具体网络结构如下所示：

![image-20240409212155622](http://qny.expressisland.cn/gdou24/image-20240409212155622.png)

## VGG16的简单使用 —— 使用**`ImageNet`**数据集？（在这里继续用`CIFAR10`数据集）

* 使用**`ImageNet`**数据集：
* **ImageNet文档链接**：https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html

### 代码（VGG16）

```python
import torchvision
from torch import nn

# 现有模型的使用及修改
# 加载vgg训练好的模型，并在里边加入一个线性层
# 暂且不用ImageNet数据集，继续用CIFAR10
train_data = torchvision.datasets.CIFAR10("test/dataset", train=True, transform=torchvision.transforms.ToTensor(),
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
```

#### 输出

```python
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
  (7): Linear(in_features=1000, out_features=10, bias=True)
)
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
```

# 17_网络模型的保存和加载

* 在根目录下已经生成了两个VGG16网络模型pth文件。

![image-20240410100427470](http://qny.expressisland.cn/gdou24/image-20240410100427470.png)

* 保存模型都是用`torch.save`，加载模型都是用`torch.load`；
* 一起保存的时候`save`整个模型，加载时直接`torch.load`加载；
* 保存时只保存参数的，需要先向`model vgg`加载结构，再用`model vgg.load state dict`加载参数，加载参数还是要`torch.load`方法。

## 方式1

* 输出后保存了网络模型及模型的参数。
* 注：没有预训练的模型不是没有参数，而是参数在初始化的状态。

```python
import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained = False)

# 保存方式1：模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")
# 加载模型
model = torch.load("vgg16_method1.pth")
print(model)
```

## 方式2

* 官方推荐使用方式2（保存的是模型参数）。

```python
import torchvision
import torch
# 保存方式2：保存的是模型参数（官方推荐）

# 先加载模型结构
vgg16 = torchvision.models.vgg16(pretrained = False)

torch.save(vgg16.state_dict,"vgg16_method2.pth")
# 输出完整的模型结构，与第一种方式输出的模型结构相同
model = torch.load("vgg16_method2.pth")
print(model)
```

## 保存方法1的陷阱

* 在使用方法1保存现有模型时，不会出错，代码更少，但是使用方法1保存自己的模型时，必须要引入这个模型的定义才可以。
* 需要先把网络结构放进来，`import`或者把`class`的定义代码粘贴过来。

### 先保存Ocean模型

```python
# 保存方法1的‘陷阱’
# 先保存Ocean模型
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


ocean = Ocean()
torch.save(ocean, "ocean_save_method1.pth")
```

### 这时直接加载ocean模型会报错

```python
ocean = torch.load("test/ocean_save_method1.pth")
```

#### 输出报错

```python
AttributeError: Can't get attribute 'Ocean' on <module '__main__' from '/root/test/save.py'>
```

#### 解决方案

* 需要先把网络结构放进来，`import`或者把`class`的定义代码粘贴过来。

##### import

```python
from xxxx import *
```
##### 或者把`class`的定义代码粘贴过来
```python
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
```

# 18_完整的模型训练套路

* 10轮训练也挺久的，建议选个好点的GPU。

## 搭建神经网络（model.py）

```python
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


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


if __name__ == '__main__':
    ocean = Ocean()
    input = torch.ones((64, 3, 32, 32))
    output = ocean(input)
    print(output.shape)
```

## 训练模块（train.py）

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="test/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="test/data", train=False, transform=torchvision.transforms.ToTensor(),
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

# 创建网络模型
ocean = Ocean()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

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
epoch = 10

# 添加tensorboard
writer = SummaryWriter("test/logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    ocean.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = ocean(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # batch=64,训练集=5W，学习一边训练集就需要781.25次训练
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    ocean.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
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

    torch.save(ocean, "ocean_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

## 最后输出结果（第 10 轮训练）

![image-20240410111740264](http://qny.expressisland.cn/gdou24/image-20240410111740264.png)

### test_accuracy

![test_accuracy](http://qny.expressisland.cn/gdou24/test_accuracy.png)

### test_loss

![test_loss](http://qny.expressisland.cn/gdou24/test_loss.png)

### train_loss

![train_loss](http://qny.expressisland.cn/gdou24/train_loss.png)

# 19_利用GPU训练

* 使用gpu训练，能提高10倍训练速度。

## 方法一：在特定位置加入`.cuda()`

* 能加的有3个地方：**模型**、**loss**、**模型输入**。

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="test/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="test/data", train=False, transform=torchvision.transforms.ToTensor(),
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

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
# 先判断再cuda
if torch.cuda.is_available():
    tudui = ocean.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
# 先判断再cuda
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

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
epoch = 10

# 添加tensorboard
writer = SummaryWriter("test/logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    ocean.train()
    for data in train_dataloader:
        imgs, targets = data

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
# 先判断再cuda
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = ocean(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # batch=64,训练集=5W，学习一边训练集就需要781.25次训练
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    ocean.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

     # 利用GPU训练
    # 方法一：在特定位置加入.cuda()
    # 先判断再cuda
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

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

    torch.save(ocean, "ocean_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

### 输出

* 虽然结果差不多，但确实快了很多，只要两分多钟。
* 实际上只用了148秒，2.46666666667 分钟。

![image-20240410160426015](http://qny.expressisland.cn/gdou24/image-20240410160426015.png)

![image-20240410155308954](http://qny.expressisland.cn/gdou24/image-20240410155308954.png)

## 方法二：在特定位置加入`.to(device)`

* 能加的有3个地方：**模型**、**loss**、**模型输入**。

### 只有cpu

```python
 device = torch.device("cpu")
```

### 只有一张显卡

```python
device = torch.device("cuda")
device = torch.device("cuda:0")
```

### 有多张显卡

```python
device = torch.device("cuda:0")
device = torch.device("cuda:1")
```
### 整体代码
```python
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

train_data = torchvision.datasets.CIFAR10(root="test/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="test/data", train=False, transform=torchvision.transforms.ToTensor(),
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
epoch = 10

# 添加tensorboard
writer = SummaryWriter("test/logs_train")
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

    torch.save(ocean, "ocean_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

### 输出测试

* 在Google Colab上运行代码。（这边快一点）

![image-20240410163026650](http://qny.expressisland.cn/gdou24/image-20240410163026650.png)

# 20_完整的模型验证套路

* **TORCHVISION.TRANSFORMS**文档：https://pytorch.org/vision/0.9/transforms.html

## 示例模型中数字代表的物品

![image-20240410170005538](http://qny.expressisland.cn/gdou24/image-20240410170005538.png)

## 测试图

![dog2](http://qny.expressisland.cn/gdou24/dog2.jpg)

## 先训练模型

* 建议直接上来训练50次来。

```python
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

train_data = torchvision.datasets.CIFAR10(root="test/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="test/data", train=False, transform=torchvision.transforms.ToTensor(),
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
writer = SummaryWriter("test/logs_train")
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
```

## 再进行模型验证

* `model = torch.load`部分加载刚才训练好的模型。

```python
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "test/images/dog2.jpg"
image = Image.open(image_path)
# 这步需要
image = image.convert("RGB")

print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)

print(image)


class Ocean(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 采用GPU训练的东西，如果只是想单纯在CPU上面跑的话，一定要从GPU上面映射到CPU上面
model = torch.load("./ocean_9_p.pth", map_location=torch.device("cpu"))
print(model)
# 这步也需要，因为这一步通常需要batchsize
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
# 这一步可以节约一些性能
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))
```

## 运行测试

* 总体上来说还算可以，但有时还是会预测为其他物体。

### 预测狗的图像

* 成功预测成功为狗（`tensor([5])`）。

![image-20240410171505459](http://qny.expressisland.cn/gdou24/image-20240410171505459.png)

### 预测猫的图像

* 成功预测成功为猫（`tensor([3])`）。

#### 测试图

![cat2](http://qny.expressisland.cn/gdou24/cat2.png)

![image-20240410172017842](http://qny.expressisland.cn/gdou24/image-20240410172017842.png)

# 21_题外话：使用Google Colab训练网络模型

* **Google Colab**地址：https://colab.research.google.com

* 如果需要使用GPU，先设置“笔记本设置”，选择GPU。

## 设置“笔记本设置”

![image-20240410160231595](http://qny.expressisland.cn/gdou24/image-20240410160231595.png)

## 整体配置

* 新建代码块输入`!nvidia-smi`

![image-20240410161358253](http://qny.expressisland.cn/gdou24/image-20240410161358253.png)

## import pytorch环境

```python
import torch
```

```python
print(torch.__version__)
# 输出：2.2.1+cu121
```

```python
print(torch.cuda.is_available())
# 输出：True
# 输出为True即可使用GPU
```

## 新建代码块

* 新建代码块，将之前利用GPU训练的代码复制进来。

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
import time

train_data = torchvision.datasets.CIFAR10(root="test/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="test/data", train=False, transform=torchvision.transforms.ToTensor(),
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

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
if torch.cuda.is_available():
    tudui = ocean.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

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
epoch = 10

# 添加tensorboard
writer = SummaryWriter("test/logs_train")
# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    ocean.train()
    for data in train_dataloader:
        imgs, targets = data

# 利用GPU训练
# 方法一：在特定位置加入.cuda()
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

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

     # 利用GPU训练
    # 方法一：在特定位置加入.cuda()
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

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

    torch.save(ocean, "ocean_{}.pth".format(i))
    print("模型已保存")

writer.close()
```

## 运行测试

* 比恒源云提供的快得多。
* 恒源云的显卡是**Tesla P4**的，Colab提供的是**Tesla T4**，高级不少。

![image-20240410161123533](http://qny.expressisland.cn/gdou24/image-20240410161123533.png)
