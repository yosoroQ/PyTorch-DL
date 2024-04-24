# 写图片数据
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("Code/logs")

image_path = "Code/dataset/train/ants_image/0013035.jpg"
img_pil = Image.open(image_path)
img_array = np.array(img_pil)

# writer.add_image("img test", img_array, 1)
writer.add_image("img test", img_array, 1, dataformats='HWC')
writer.close()