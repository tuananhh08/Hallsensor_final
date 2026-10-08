"""AMagPoseNet – losses/__init__.py"""
from .calinet_loss import CaliNetLoss
from .pose_loss import AMagPoseLoss, CalibPoseLoss

__all__ = ["CaliNetLoss", "AMagPoseLoss", "CalibPoseLoss"]
