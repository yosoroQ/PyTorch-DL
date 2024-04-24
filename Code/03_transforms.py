# transforms的使用
# transform是一个py文件，其中tosensor compose normalize是很常用的操作
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# tosensor简单使用 

# 获取pil类型图片
img_path = "Code/dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
# 创建需要的transforms工具，并给工具起名字
tensor_trans = transforms.ToTensor()
# 使用工具
tensor_img = tensor_trans(img)
print(tensor_img)

# demo3:使用tensor数据类型写入board
writer = SummaryWriter("Code/logs")
writer.add_image('tensor img', tensor_img, 1)
writer.close()