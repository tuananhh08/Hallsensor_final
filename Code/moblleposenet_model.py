# mobileposenet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from convnext_block import ConvNeXtBlock

class Stem(nn.Module):
    def __init__(self, out_ch: int = 6):
        super().__init__()
        assert out_ch % 2 == 0
        c = out_ch // 2
        self.Conv= nn.Sequential(
            nn.Conv2d(1, c, kernel_size=1, padding=2, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU6(inplace=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([self.Conv(x)], dim=1))
