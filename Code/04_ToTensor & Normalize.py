# 常见的transforms
# ToTensor的使用与Normalize(归一化)的使用
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("Code/logs")
img = Image.open("Code/train/ants_image/0013035.jpg")
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