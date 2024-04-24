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

writer = SummaryWriter("Code/logs")

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