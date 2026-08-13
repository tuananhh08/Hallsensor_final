import torch
import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        se_input = x

        y = self.avg_pool(x).view(b, c)      # AdaptiveAvgPool
        y = self.fc(y).view(b, c, 1, 1)       # Linear -> ReLU -> Linear -> Sigmoid

        return se_input * y                   # multiply (⊗)


class IRAB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_factor: int = 6,
        se_reduction: int = 4,
    ):
        super().__init__()

        assert stride in (1, 2), "stride chỉ nên là 1 hoặc 2"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        expanded_channels = in_channels * expansion_factor

        # Chỉ dùng shortcut (nhánh cộng residual) khi stride=1 và k == k'
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)

        # 1) 1x1 conv2d, ReLU6  
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )

        # 2) 3x3 dwise s=s, ReLU6 
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                      stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )

        # 3) linear 1x1 conv2d 
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),  # "linear" -> không có activation
        )

        # 4) SEBlock 
        self.se_block = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.project_conv(out)

        if self.use_shortcut:
            out = out + identity  # nhánh "shortcut" (⊕)
        # else: nhánh "otherwise" -> giữ nguyên out (không cộng residual)

        se_input = out
        out = self.se_block(se_input)

        return out
