import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """单个 Dense 层，包含 BN + ReLU + 3D Conv"""

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)  # 连接输入，实现特征复用


class DenseBlock(nn.Module):
    """包含多个 DenseLayer"""

    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    """过渡层，降维 + 池化"""

    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)  # 2x 下采样

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)


class DenseNet3D(nn.Module):
    """完整的 3D DenseNet 结构"""

    def __init__(self, num_classes=2, in_channels=1, growth_rate=16, block_config=(4, 4, 4)):
        super(DenseNet3D, self).__init__()

        # 初始 3D 卷积层
        num_init_features = 32
        self.conv1 = nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense Blocks
        num_features = num_init_features
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:  # 过渡层
                trans = TransitionBlock(num_features, num_features // 2)
                self.blocks.append(trans)
                num_features //= 2

        # BN + GAP + FC
        self.bn_final = nn.BatchNorm3d(num_features)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # 全局平均池化
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.bn_final(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 测试模型
if __name__ == "__main__":
    model = DenseNet3D(num_classes=1)
    x = torch.randn(2, 1, 96, 96, 60)  # batch_size=2, 单通道 MRI, 深度 32, 高 64, 宽 64
    output = model(x)
    print(output.shape)  # 预期输出: (2, 2)
