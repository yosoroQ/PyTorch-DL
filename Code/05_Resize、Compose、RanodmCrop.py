# 常见的transforms
# Resize、Compose与RanodmCrop的使用
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("Code/logs")
img = Image.open("Code/train/ants_image/0013035.jpg")
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