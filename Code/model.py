# import torch
# import torch.nn as nn
# from cbam import CBAM
# from convnext_block import ConvNeXtBlock
# import torch.nn.functional as F

# class Stem(nn.Module):
#     def __init__(self, out_ch: int = 16):
#         super().__init__()
#         assert out_ch % 2 == 0
#         c = out_ch // 2
#         self.b1 = nn.Sequential(
#             nn.Conv2d(1, c, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(c),
#         )
#         self.b3 = nn.Sequential(
#             nn.Conv2d(1, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#         )
#         self.act = nn.LeakyReLU(0.01, inplace=True)
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.act(torch.cat([self.b1(x), self.b3(x)], dim=1))
    
    
# class ConvNeXtBlockWithCBAM(nn.Module):
#     def __init__(self, in_ch, out_ch, drop_path_rate=0, cbam_reduction=2):
#         super().__init__()
#         self.block = ConvNeXtBlock(in_ch, out_ch, stride=1, drop_path_rate=drop_path_rate)
#         self.cbam  = CBAM(out_ch, reduction=cbam_reduction)

#     def forward(self, x):

#         return self.cbam(self.block(x))

# class DualPool(nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         avg = F.adaptive_avg_pool2d(x, 1).flatten(1)  
#         mx  = F.adaptive_max_pool2d(x, 1).flatten(1)  
#         return torch.cat([avg, mx], dim=1)             
    
# class Model(nn.Module):
#     def __init__(self, out_dim: int = 5, drop_path_rate: float = 0.035):
#         super().__init__()
 
#         # Stem
#         self.stem = Stem(out_ch=16)
 
#         # Stage1
#         self.stage2 = nn.Sequential(
#             ConvNeXtBlockWithCBAM(16, 32, drop_path_rate=drop_path_rate),
#             ConvNeXtBlockWithCBAM(32, 32, drop_path_rate=drop_path_rate),
#         )
 
#         # Stage2
#         self.stage3 = nn.Sequential(
#             ConvNeXtBlockWithCBAM(32, 64, drop_path_rate=drop_path_rate),
#             ConvNeXtBlockWithCBAM(64, 64, drop_path_rate=drop_path_rate),
#         )

#         # Stage 3
#         self.stage4 = nn.Sequential(
#             ConvNeXtBlockWithCBAM(64, 96, drop_path_rate=drop_path_rate),
#             ConvNeXtBlockWithCBAM(96, 96, drop_path_rate=drop_path_rate),
#         ) 
        
#         # GAP + Flatten: 
#         self.pool = DualPool()  
#         self.flatten = nn.Flatten(1)
 
 
#         self.shared = nn.Sequential(
#             nn.Linear(192, 64, bias=False),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.01, inplace=True),
#         )
 
#         # --- Regression heads ---
#         self.head_xyz = nn.Linear(64, 3)
 
#         self.head_ang = nn.Sequential(
#             nn.Linear(64, 16),
#             nn.LeakyReLU(0.01, inplace=True),
#             nn.Linear(16, 2),
#             nn.Tanh(),                          
#         )
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.stem(x)     
 
#         x = self.stage2(x)   
#         x = self.stage3(x)   
#         x = self.stage4(x)   

#         x = self.pool(x)      
#         x = self.flatten(x)  # 128
#         x = self.shared(x)   # 64
 
#         xyz = self.head_xyz(x)   # (B, 3)
#         ang = self.head_ang(x)   # (B, 2) 
 
#         return torch.cat([xyz, ang], dim=1)  # (B, 5)  
    
# if __name__ == "__main__":
#     model = Model()
#     total = sum(p.numel() for p in model.parameters())
#     print(f"Total params : {total:,}")
 
#     x   = torch.randn(4, 1, 8, 8)
#     out = model(x)
#     print(f"Input shape  : {x.shape}")
#     print(f"Output shape : {out.shape}")
 
#     print("\nParams per top-level module:")
#     for name, mod in model.named_children():
#         p = sum(v.numel() for v in mod.parameters())
#         print(f"  {name:<12} {p:>7,}")


import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM, ChannelAttention
from convnext_block import ConvNeXtBlock
from Code.calibnet import CalibNet

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
            nn.SiLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, padding=2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.bn_out = nn.BatchNorm2d(out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.bn_out(out) 


class LocalizationNet(nn.Module):
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
        self.cbam = CBAM(36)
        
        # Stage 4: 72x8x8
        self.stage4 = nn.Sequential(                                    
            ConvNeXtBlock(36, 72, drop_path_rate=drop_path_rate),
            ConvNeXtBlock(72, 72, drop_path_rate=drop_path_rate),
        )
        self.cbam2 = CBAM(72)
        
        # Stage 5: 144x4x4
        self.stage5 = nn.Sequential(                                    
            ConvNeXtBlock(72, 144, drop_path_rate=drop_path_rate, stride = 2),
            ConvNeXtBlock(144, 144, drop_path_rate=drop_path_rate),
        )

        # Attention + Pool
        self.ca5   = ChannelAttention(144)
        self.pool  = DualPool()                          

        self.head_xyz = nn.Linear(288, 3)

        self.head_mvec = nn.Sequential(
            nn.Linear(288, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            )

    def forward(self, x):
        x = self.stage1(x)            

        x = self.stage2(x)
        x = self.stage3(x)
                    
        x = self.cbam(x)              

        x = self.stage4(x)    
        x = self.cbam2(x)
        
        x = self.stage5(x)            
        x = self.ca5(x) 

        x = self.pool(x)              
      
        xyz = self.head_xyz(x)

        m_raw  = self.head_mvec(x)
        m_norm = F.normalize(m_raw, dim=-1, eps = 1e-6)

        return torch.cat([xyz, m_norm], dim=1)


class Model(nn.Module):
    """Full localization model"""
    def __init__(self, out_dim: int = 6, drop_path_rate: float = 0.015,
                 use_modnet: bool = True):
        super().__init__()
        self.use_calibnet = use_modnet
        self.calibnet = CalibNet()
        self.locnet = LocalizationNet(out_dim=out_dim, drop_path_rate=drop_path_rate)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        input_x = x
        if self.use_calibnet:
            corrected, residual = self.calibnet(x, return_residual=True)
        else:
            corrected, residual = x, torch.zeros_like(x)
        pred = self.locnet(corrected)
        if return_features:
            return pred, {"input": input_x, "corrected": corrected, "residual": residual}
        return pred

