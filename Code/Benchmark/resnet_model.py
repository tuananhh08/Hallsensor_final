import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class Stem(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = Stem(16)

        # 8x8
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 16),
        )

        # 8x8 -> 4x4
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32),
        )

        # 4x4
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64),
            BasicBlock(64, 64),
        )

        # 4x4
        self.layer4 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.head_xyz = nn.Linear(256, 3)

        self.head_mvec = nn.Linear(256, 3)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.conv(x)

        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)

        xyz = self.head_xyz(x)

        m_raw = self.head_mvec(x)

        m_norm = F.normalize(m_raw, dim=1, eps=1e-6)

        return torch.cat([xyz, m_norm], dim=1)

