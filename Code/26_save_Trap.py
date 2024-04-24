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

# 这时直接加载ocean模型会报错
# ocean = torch.load("Code/ocean_save_method1.pth")

# 输出报错
# AttributeError: Can't get attribute 'Ocean' on <module '__main__' from '/root/Code/save.py'>

# #### 解决方案
# 需要先把网络结构放进来，`import`或者把`class`的定义代码粘贴过来
# 或者把`class`的定义代码粘贴过来