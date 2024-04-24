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