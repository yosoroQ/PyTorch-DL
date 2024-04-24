import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained = False)

# 保存方式1：模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")
# 加载模型
model = torch.load("vgg16_method1.pth")
print(model)