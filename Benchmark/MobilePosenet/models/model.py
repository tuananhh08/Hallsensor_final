import torch
import torch.nn as nn
import torch.nn.functional as F
from IRAB_block import IRAB

class Stem(nn.Module):
    def __init__(self, out_ch: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)
    
class Model(nn.Module):
    def __init__(self, out_dim: int = 6):
        super().__init__()

        # Stem block
        self.stem = Stem(out_ch=4)
        
        # IRAB1
        self.irab1 = IRAB(in_channels=4, out_channels=8, stride=1, expansion_factor=1, se_reduction=4)
        
        # IRAB2
        self.irab2 = IRAB(in_channels=8, out_channels=16, stride=1, expansion_factor=6, se_reduction=4)
        
        # IRAB3
        self.irab3 = IRAB(in_channels=16, out_channels=32, stride=2, expansion_factor=6, se_reduction=4)
        
        # IRAB4
        self.irab4 = IRAB(in_channels=32, out_channels=64, stride=1, expansion_factor=6, se_reduction=4)
        
        # IRAB5
        self.irab5 = IRAB(in_channels=64, out_channels=128, stride=1, expansion_factor=1, se_reduction=4)
        
        # Conv Block
        
        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Flatten Layer
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer
        self.head_xyz = nn.Linear(256, 3)
        
        self.head_mvec = nn.Linear(256, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        x = self.stem(x)
        
        x = self.irab1(x)
        x = self.irab2(x)
        x = self.irab3(x)
        x = self.irab4(x)
        x = self.irab5(x)
        x = self.conv(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        
        xyz = self.head_xyz(x)
            
        m_raw  = self.head_mvec(x)
        m_norm = F.normalize(m_raw, dim=-1, eps = 1e-6)

        return torch.cat([xyz, m_norm], dim=1)
            
        
        
        
        
        