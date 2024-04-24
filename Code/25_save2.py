import torchvision
import torch
# 保存方式2：保存的是模型参数（官方推荐）

# 先加载模型结构
vgg16 = torchvision.models.vgg16(pretrained = False)

torch.save(vgg16.state_dict,"vgg16_method2.pth")
# 输出完整的模型结构，与第一种方式输出的模型结构相同
model = torch.load("vgg16_method2.pth")
print(model)