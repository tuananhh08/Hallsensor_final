"""Report the parameter count and inference MACs for :mod:`model`.

Examples
--------
python Code/abc.py
python Code/abc.py --checkpoint training_ckpt/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count parameters and Conv/Linear MACs for model.py.")
    parser.add_argument(
        "--input-size", nargs=4, type=int, metavar=("B", "C", "H", "W"),
        default=(1, 1, 8, 8), help="Input tensor shape (default: 1 1 8 8).")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional .pt checkpoint to validate against the current architecture.")
    return parser.parse_args()


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Load common checkpoint formats, including torch.compile key prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")

    state_dict = {
        key.removeprefix("_orig_mod.").removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return total parameters, trainable parameters, and float32 storage bytes."""
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    bytes_used = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return total, trainable, bytes_used


def count_macs(model: nn.Module, input_size: tuple[int, int, int, int]) -> tuple[int, torch.Size]:
    """Count one forward-pass MAC for every Conv2d and Linear multiply-accumulate.

    Non-parametric operations such as activations, pooling, normalization, and
    elementwise attention multiplications are deliberately excluded. This is
    the conventional model MAC count used for convolutional neural networks.
    """
    macs = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        output_elements = output.numel()
        kernel_elements = module.kernel_size[0] * module.kernel_size[1]
        macs += output_elements * (module.in_channels // module.groups) * kernel_elements

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], _output: torch.Tensor) -> None:
        nonlocal macs
        batch_instances = inputs[0].numel() // module.in_features
        macs += batch_instances * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    try:
        with torch.inference_mode():
            output = model(torch.zeros(input_size, dtype=torch.float32))
    finally:
        for hook in hooks:
            hook.remove()
    return macs, output.shape


def main() -> None:
    args = parse_args()
    input_size = tuple(args.input_size)
    if input_size[1:] != (1, 8, 8):
        raise ValueError("Model hien tai yeu cau input co shape (B, 1, 8, 8).")
    if input_size[0] < 1:
        raise ValueError("Batch size phai >= 1.")

    model = Model(out_dim=6).eval()
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)

    total_params, trainable_params, size_bytes = count_parameters(model)
    macs, output_shape = count_macs(model, input_size)
    macs_per_sample = macs / input_size[0]

    print(f"Input shape        : {input_size}")
    print(f"Output shape       : {tuple(output_shape)}")
    print(f"Parameters (total) : {total_params:,}")
    print(f"Parameters (train) : {trainable_params:,}")
    print(f"Model size FP32    : {size_bytes / 1024:.1f} KiB")
    print(f"MACs / batch       : {macs:,} ({macs / 1e6:.3f} M)")
    print(f"MACs / sample      : {macs_per_sample:,.0f} ({macs_per_sample / 1e6:.3f} M)")
    print("MAC scope          : Conv2d + Linear (bias, activation, BN, pooling excluded)")


if __name__ == "__main__":
    main()
