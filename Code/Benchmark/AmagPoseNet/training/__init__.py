"""AMagPoseNet – training/__init__.py"""
from .train_calinet import train_calinet
from .pretrain_posenet import pretrain_posenet
from .finetune import finetune

__all__ = ["train_calinet", "pretrain_posenet", "finetune"]
