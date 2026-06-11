import math
import torch
from torch import nn

class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, first_part_extend=0):
        super(ESPCN, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5 // 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2),
            nn.Tanh(),
        )
        for _ in range(first_part_extend):
            self.first_part.extend([
                nn.Conv2d(32, 32, kernel_size=3, padding=3 // 2),
            ])
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x

# # 创建一个ESPCN网络实例
# scale_factor = 2  # 举例，你可以使用其他比例
# num_channels = 3  # 输入图像通道数，这里假设为3
# num_second_layers = 1  # 第二部分的卷积层数量
# espcn = ESPCN(scale_factor, num_channels, num_second_layers)

# # 计算总层数
# total_layers = len(espcn.first_part) + len(espcn.last_part)
# print("Total number of layers in the network:", total_layers)
