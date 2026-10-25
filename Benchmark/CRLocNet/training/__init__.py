"""Three-stage transfer-learning training API for CRLocNet."""
from .stages import train_modnet, pretrain_locnet, finetune

__all__ = ["train_modnet", "pretrain_locnet", "finetune"]
