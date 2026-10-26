"""Training-output helpers local to the AMagPoseNet benchmark."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

def save_loss_plot(log_path: Path, output_path: Path, title: str) -> None:
    try: history = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): history = []
    if not history: return
    keys = [key for key, value in history[0].get("train", {}).items() if isinstance(value, (int, float))]
    if not keys: return
    epochs = [entry["epoch"] for entry in history]
    figure, axis = plt.subplots(figsize=(10, 6))
    for key in keys:
        axis.plot(epochs, [entry["train"].get(key, float("nan")) for entry in history], label=f"Train {key}")
        axis.plot(epochs, [entry["val"].get(key, float("nan")) for entry in history], linestyle="--", label=f"Val {key}")
    axis.set(xlabel="Epoch", ylabel="Loss", title=title); axis.grid(alpha=0.35); axis.legend(); figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(figure)
    print(f"Saved loss plot: {output_path}")
