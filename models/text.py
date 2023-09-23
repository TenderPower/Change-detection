import torch
import torch.nn as nn
from mmflow.models.decoders.raft_decoder import CorrelationPyramid
# 定义一个包含单个卷积层的网络
class ImageConverter(nn.Module):
    def __init__(self):
        super(ImageConverter, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(8, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=6, stride=2, padding=2)

    def forward(self, x):
        # x = self.layer(x)
        # x = self.layer2(x)
        x = self.conv(x)
        return x


# 创建一个示例输入
input_image = torch.randn(1, 3, 32, 32)  # 输入图像的大小为28x28，通道数为3

# 创建图像转换器
converter = ImageConverter()

# 将输入图像传递给图像转换器
output_image = converter(input_image)

# 打印输出图像的大小
print(output_image.size())  # 输出结果应为torch.Size([1, 10, 3, 3])
img1 = torch.randn(1,256,8,8)
img2 = torch.randn(1,256,8,8)
cp = CorrelationPyramid(1)
out = cp(img1,img2)
out