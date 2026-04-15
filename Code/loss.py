import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberPoseLoss(nn.Module):

    def __init__(self,
                 ang_weight: float = 1.0,
                 delta_xyz:  float = 0.055,
                 delta_ang:  float = 0.21):

        super().__init__()
        self.ang_weight = ang_weight
        self.delta_xyz  = delta_xyz
        self.delta_ang  = delta_ang

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
 
        loss_xyz = F.huber_loss(pred[:, :3], target[:, :3],
                                delta=self.delta_xyz)
        loss_ang = F.huber_loss(pred[:, 3:], target[:, 3:],
                                delta=self.delta_ang)
        total = loss_xyz + self.ang_weight * loss_ang
        return total, loss_xyz, loss_ang