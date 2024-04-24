import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "Code/images/dog2.jpg"
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