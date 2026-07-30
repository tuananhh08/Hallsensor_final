import torch
import torch.nn as nn
import torch.nn.functional as F
from convnext_block import ConvNeXtBlock

class DualPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x).flatten(1)  
        mx  = self.max_pool(x).flatten(1)  
        return torch.cat([avg, mx], dim=1)

class Stage1(nn.Module):   
    def __init__(self, out_ch: int = 8):
        super().__init__()
        assert out_ch % 3 == 0
        c = out_ch // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, padding=2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.bn_out = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.bn_out(out) 


class Model(nn.Module):
    def __init__(self, out_dim: int = 6, drop_path_rate: float = 0.015):
        super().__init__()
        self.out_dim = out_dim

        # Stage 1: 9x8x8
        self.stage1 = Stage1(out_ch=9)

        # Stage 2: 18x8x8
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(9,  18, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(18, 18, drop_path_rate=drop_path_rate),
        )
        
        # Stage 3: 36x8x8
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(18, 36, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(36, 36, drop_path_rate=drop_path_rate),
        )
        
        # Stage 4: 64x8x8
        self.stage4 = nn.Sequential(                                    
            ConvNeXtBlock(36, 64, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(64, 64, drop_path_rate=drop_path_rate),
        )
        
        # Stage 5: 96x8x8
        self.stage5 = nn.Sequential(                                    
            ConvNeXtBlock(64, 96, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(96, 96, drop_path_rate=drop_path_rate),
        )

        # Attention + Pool
        self.pool     = DualPool()                          

        self.head_xyz = nn.Linear(192, 3)

        self.head_mvec = nn.Sequential(
            nn.Linear(192, 32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(32, 3),
            )

    def forward(self, x):
        x = self.stage1(x)            

        x = self.stage2(x)
        x = self.stage3(x)
                    
        x = self.stage4(x)    
        
        x = self.stage5(x)            

        x = self.pool(x)              
      
        xyz = self.head_xyz(x)

        m_raw  = self.head_mvec(x)
        m_norm = F.normalize(m_raw, dim=-1)

        return torch.cat([xyz, m_norm], dim=1)