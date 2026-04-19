import os, sys, pickle, argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 1.  Model definitions
# ══════════════════════════════════════════════════════════════════════

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    mask = torch.empty(x.shape[0], 1, 1, 1,
                       device=x.device).bernoulli_(keep) / keep
    return x * mask

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        return drop_path(x, self.p, self.training)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        h = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(h, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.mlp(self.avg_pool(x).view(b, c))
        m = self.mlp(self.max_pool(x).view(b, c))
        return x * self.sigmoid(a + m).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_path_rate=0.0):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                              padding=1, groups=in_ch, bias=False)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.dp   = DropPath(drop_path_rate)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if stride != 1 or in_ch != out_ch else nn.Identity()
        )
    def forward(self, x):
        res = self.skip(x)
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw(out)))
        return self.dp(out) + res

# ── Model CÓ CBAM ────────────────────────────────────────────────────
class ModelWithCBAM(nn.Module):
    def __init__(self, drop_path_rate=0.055):
        super().__init__()
        dp = drop_path_rate
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.LeakyReLU(0.01, inplace=True))
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(8,  16, drop_path_rate=dp),
            ConvNeXtBlock(16, 16, drop_path_rate=dp))
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(16, 32, drop_path_rate=dp),
            ConvNeXtBlock(32, 32, drop_path_rate=dp))
        self.cbam   = CBAM(32)                      # <-- CÓ CBAM
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(32, 64, stride=2, drop_path_rate=dp),
            ConvNeXtBlock(64, 64, drop_path_rate=dp))
        self.stage5 = nn.Sequential(
            ConvNeXtBlock(64,  128, drop_path_rate=dp),
            ConvNeXtBlock(128, 128, drop_path_rate=dp))
        self.ca5     = ChannelAttention(128)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head_xyz = nn.Linear(128, 3)
        self.head_ang = nn.Sequential(
            nn.Linear(128, 16), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 2),   nn.Tanh())
    def forward(self, x):
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.cbam(x)
        x = self.stage4(x); x = self.stage5(x); x = self.ca5(x)
        x = self.flatten(self.gap(x))
        return torch.cat([self.head_xyz(x), self.head_ang(x)], dim=1)

# ── Model KHÔNG CÓ CBAM (ablation baseline) ──────────────────────────
class ModelWithoutCBAM(nn.Module):
    def __init__(self, drop_path_rate=0.055):
        super().__init__()
        dp = drop_path_rate
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.LeakyReLU(0.01, inplace=True))
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(8,  16, drop_path_rate=dp),
            ConvNeXtBlock(16, 16, drop_path_rate=dp))
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(16, 32, drop_path_rate=dp),
            ConvNeXtBlock(32, 32, drop_path_rate=dp))
        # KHÔNG có CBAM
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(32, 64, stride=2, drop_path_rate=dp),
            ConvNeXtBlock(64, 64, drop_path_rate=dp))
        self.stage5 = nn.Sequential(
            ConvNeXtBlock(64,  128, drop_path_rate=dp),
            ConvNeXtBlock(128, 128, drop_path_rate=dp))
        self.ca5     = ChannelAttention(128)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head_xyz = nn.Linear(128, 3)
        self.head_ang = nn.Sequential(
            nn.Linear(128, 16), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 2),   nn.Tanh())
    def forward(self, x):
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        # Bỏ qua CBAM
        x = self.stage4(x); x = self.stage5(x); x = self.ca5(x)
        x = self.flatten(self.gap(x))
        return torch.cat([self.head_xyz(x), self.head_ang(x)], dim=1)


# ══════════════════════════════════════════════════════════════════════
# 2.  Load data — giống _read() trong train.py của bạn
# ══════════════════════════════════════════════════════════════════════

def read_csv_auto(path):
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce") \
             .dropna().reset_index(drop=True).values.astype(np.float32)

def load_data(voltage_path, label_path, scaler_path):
    voltages_raw = read_csv_auto(voltage_path)   # (N, 64)
    labels_raw   = read_csv_auto(label_path)     # (N,  5)
    N = min(len(voltages_raw), len(labels_raw))
    voltages_raw, labels_raw = voltages_raw[:N], labels_raw[:N]

    assert voltages_raw.shape[1] == 64, \
        f"Voltage cần 64 cột, hiện có {voltages_raw.shape[1]}"
    assert labels_raw.shape[1] == 5, \
        f"Label cần 5 cột, hiện có {labels_raw.shape[1]}"

    with open(scaler_path, "rb") as f:
        sc = pickle.load(f)
    volt_scaler  = sc["volt"]
    label_scaler = sc["label"]
    voltages_scaled = volt_scaler.transform(voltages_raw)  # (N, 64)

    print(f"  Tổng mẫu          : {N}")
    print(f"  Voltage (raw)     : [{voltages_raw.min():.4f},  {voltages_raw.max():.4f}]")
    print(f"  Voltage (scaled)  : [{voltages_scaled.min():.4f}, {voltages_scaled.max():.4f}]")
    return voltages_raw, voltages_scaled, labels_raw, label_scaler


# ══════════════════════════════════════════════════════════════════════
# 3.  Load checkpoint — giống load_checkpoint() trong train.py của bạn
# ══════════════════════════════════════════════════════════════════════

def load_model(model_cls, ckpt_path, device):
    model = model_cls().to(device)
    if ckpt_path is None or not os.path.exists(str(ckpt_path)):
        print(f"  [!] Không tìm thấy '{ckpt_path}' — dùng random weights (demo)")
        return model.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw  = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    # Xử lý prefix _orig_mod. của torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    model.load_state_dict(state, strict=True)
    ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Loaded: {ckpt_path}  (epoch {ep})")
    return model.eval()


# ══════════════════════════════════════════════════════════════════════
# 4.  Grad-CAM
# ══════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._feat = None
        self._grad = None
        h1 = target_layer.register_forward_hook(
            lambda _, __, out: setattr(self, "_feat", out.detach()))
        h2 = target_layer.register_full_backward_hook(
            lambda _, __, gin: setattr(self, "_grad", gin[0].detach()))
        self._hooks = [h1, h2]

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, x, output_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad()
        # Scalar target
        ((out**2).sum() if output_idx is None else out[0, output_idx]).backward()
        # GAM weights
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(8, 8),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo) if hi > lo else cam


# ══════════════════════════════════════════════════════════════════════
# 5.  Figure
# ══════════════════════════════════════════════════════════════════════

def add_cbar(fig, ax, im, label):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7); cb.ax.tick_params(labelsize=6)

def set_sensor_ticks(ax):
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([str(i) for i in range(8)], fontsize=6)
    ax.set_yticklabels([str(i) for i in range(8)], fontsize=6)
    ax.tick_params(length=2)
    ax.set_xlabel("Column", fontsize=6); ax.set_ylabel("Row", fontsize=6)

def make_figure(volt_raw_8x8, cam_with, cam_without,
                pred_with_raw, pred_without_raw, gt_raw,
                sample_idx, out_path):

    fig = plt.figure(figsize=(7.16, 6.8))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            left=0.07, right=0.97,
                            top=0.92,  bottom=0.07,
                            wspace=0.52, hspace=0.62)

    # (a) Raw sensor heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(volt_raw_8x8, cmap="RdYlBu_r",
                   aspect="equal", interpolation="nearest")
    ax.set_title("(a) Input: sensor voltage map",
                 fontsize=8, fontweight="bold", pad=4)
    set_sensor_ticks(ax); add_cbar(fig, ax, im, "Voltage")

    # Normalise raw map for overlay background
    norm = (volt_raw_8x8 - volt_raw_8x8.min()) / \
           max(volt_raw_8x8.max() - volt_raw_8x8.min(), 1e-9)

    # (b) Grad-CAM CÓ CBAM
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(norm, cmap="gray", aspect="equal",
              interpolation="nearest", vmin=0, vmax=1)
    im = ax.imshow(cam_with, cmap="jet", aspect="equal",
                   interpolation="bilinear", alpha=0.60, vmin=0, vmax=1)
    ax.set_title("(b) Grad-CAM  WITH  CBAM",
                 fontsize=8, fontweight="bold", pad=4)
    set_sensor_ticks(ax); add_cbar(fig, ax, im, "Activation")

    # (c) Grad-CAM KHÔNG CÓ CBAM
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(norm, cmap="gray", aspect="equal",
              interpolation="nearest", vmin=0, vmax=1)
    im = ax.imshow(cam_without, cmap="jet", aspect="equal",
                   interpolation="bilinear", alpha=0.60, vmin=0, vmax=1)
    ax.set_title("(c) Grad-CAM  WITHOUT  CBAM",
                 fontsize=8, fontweight="bold", pad=4)
    set_sensor_ticks(ax); add_cbar(fig, ax, im, "Activation")

    # (d) Difference map
    ax   = fig.add_subplot(gs[1, :2])
    diff = cam_with - cam_without
    lim  = max(abs(diff.min()), abs(diff.max()), 1e-9)
    im   = ax.imshow(diff, cmap="RdBu_r", aspect="equal",
                     interpolation="bilinear", vmin=-lim, vmax=lim)
    ax.set_title("(d) Attention difference  (With CBAM − Without CBAM)",
                 fontsize=8, fontweight="bold", pad=4)
    set_sensor_ticks(ax)
    ax.text(0.99, 0.02,
            "Red = CBAM focuses MORE   |   Blue = CBAM focuses LESS",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom",
            color="#555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    add_cbar(fig, ax, im, "Δ Activation")

    # (e) Prediction summary table
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    err_w  = np.linalg.norm(pred_with_raw[:3]    - gt_raw[:3])
    err_wo = np.linalg.norm(pred_without_raw[:3] - gt_raw[:3])
    lines = [
        f"Sample : {sample_idx}",
        "",
        "Ground truth:",
        f"  X={gt_raw[0]:.1f}  Y={gt_raw[1]:.1f}  Z={gt_raw[2]:.1f} mm",
        f"  α={np.degrees(gt_raw[3]):.1f}°   β={np.degrees(gt_raw[4]):.1f}°",
        "",
        "─"*26,
        "With CBAM:",
        f"  X={pred_with_raw[0]:.1f}  Y={pred_with_raw[1]:.1f}  Z={pred_with_raw[2]:.1f}",
        f"  α={np.degrees(pred_with_raw[3]):.1f}°   β={np.degrees(pred_with_raw[4]):.1f}°",
        f"  Pos. err = {err_w:.2f} mm",
        "",
        "Without CBAM:",
        f"  X={pred_without_raw[0]:.1f}  Y={pred_without_raw[1]:.1f}  Z={pred_without_raw[2]:.1f}",
        f"  α={np.degrees(pred_without_raw[3]):.1f}°   β={np.degrees(pred_without_raw[4]):.1f}°",
        f"  Pos. err = {err_wo:.2f} mm",
    ]
    ax.text(0.04, 0.97, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=5.8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#aaa", lw=0.8))

    # (f) Line profile
    ax   = fig.add_subplot(gs[2, :])
    peak = int(np.unravel_index(cam_with.argmax(), cam_with.shape)[0])
    cols = np.arange(8)
    ax.plot(cols, cam_with[peak], color="#e63946", lw=2,
            marker="o", ms=5, label="With CBAM")
    ax.plot(cols, cam_without[peak], color="#457b9d", lw=2,
            marker="s", ms=5, linestyle="--", label="Without CBAM")
    ax.fill_between(cols, cam_with[peak], cam_without[peak],
                    alpha=0.12, color="#e63946")
    ax.set_title(
        f"(e) Activation profile — sensor row {peak}"
        f"  (row of peak activation, With CBAM)",
        fontsize=8, fontweight="bold")
    ax.set_xlabel("Sensor column index", fontsize=8)
    ax.set_ylabel("Grad-CAM activation", fontsize=8)
    ax.set_xticks(cols)
    ax.set_xticklabels([str(i) for i in cols], fontsize=7)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=7); ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "Grad-CAM: Effect of CBAM attention on network focus\n"
        f"(ConvNeXt stage-3 output — sample index {sample_idx})",
        fontsize=9, fontweight="bold", y=0.985)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure saved → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--voltage",      required=True,
                    help="CSV voltage (không header, 64 cột)")
    pa.add_argument("--label",        required=True,
                    help="CSV label   (có header, 5 cột: X Y Z alpha beta)")
    pa.add_argument("--scaler",       required=True,
                    help="scalers.pkl được tạo bởi train.py  (ckpt2/scalers.pkl)")
    pa.add_argument("--ckpt_with",    default=None,
                    help="Checkpoint CÓ CBAM  (ckpt2/best.pt)")
    pa.add_argument("--ckpt_without", default=None,
                    help="Checkpoint KHÔNG CBAM  (ckpt_no_cbam/best.pt)")
    pa.add_argument("--sample_idx",   type=int, default=0,
                    help="Index mẫu cần visualize (mặc định 0)")
    pa.add_argument("--output_idx",   type=int, default=None,
                    help="Chiều output cho Grad-CAM: 0=X 1=Y 2=Z 3=α 4=β "
                         "(None = L2 norm toàn bộ 5 output, khuyến nghị)")
    pa.add_argument("--out",          default="gradcam_compare.pdf")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Grad-CAM Attention Comparison")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Voltage     : {args.voltage}")
    print(f"  Label       : {args.label}")
    print(f"  Scaler      : {args.scaler}")
    print(f"  Ckpt WITH   : {args.ckpt_with}")
    print(f"  Ckpt W/O    : {args.ckpt_without}")
    print(f"  Sample idx  : {args.sample_idx}")
    print(f"  Output idx  : {args.output_idx}  (None = L2 norm, khuyến nghị)")
    print(f"{'='*55}\n")

    # ── Load data ───────────────────────────────────────────────────────
    print("Đọc data...")
    volt_raw, volt_scaled, labels_raw, label_scaler = load_data(
        args.voltage, args.label, args.scaler)

    idx = args.sample_idx
    assert 0 <= idx < len(volt_raw), \
        f"sample_idx={idx} vượt quá số mẫu ({len(volt_raw)})"

    volt_raw_8x8    = volt_raw[idx].reshape(8, 8)
    volt_scaled_flat = volt_scaled[idx]               # (64,)  đã scale
    gt_raw           = labels_raw[idx]                # (5,)   đơn vị thực

    x = torch.tensor(volt_scaled_flat, dtype=torch.float32) \
              .reshape(1, 1, 8, 8).to(device)

    # ── Load models ─────────────────────────────────────────────────────
    print("\nLoad models...")
    model_w  = load_model(ModelWithCBAM,    args.ckpt_with,    device)
    model_wo = load_model(ModelWithoutCBAM, args.ckpt_without, device)

    # ── Grad-CAM ────────────────────────────────────────────────────────
    # Target: pointwise conv của ConvNeXtBlock cuối trong stage3
    # → spatial size = 8×8, khớp 1-1 với sensor grid
    cam_w  = GradCAM(model_w,  model_w.stage3[-1].pw)
    cam_wo = GradCAM(model_wo, model_wo.stage3[-1].pw)

    print("\nTính Grad-CAM...")
    heatmap_w  = cam_w(x,  args.output_idx)
    heatmap_wo = cam_wo(x, args.output_idx)

    # ── Predictions → inverse transform về đơn vị thực ─────────────────
    with torch.no_grad():
        pred_w  = model_w(x).cpu().numpy()
        pred_wo = model_wo(x).cpu().numpy()

    pred_w_raw  = label_scaler.inverse_transform(pred_w)[0]
    pred_wo_raw = label_scaler.inverse_transform(pred_wo)[0]

    err_w  = np.linalg.norm(pred_w_raw[:3]  - gt_raw[:3])
    err_wo = np.linalg.norm(pred_wo_raw[:3] - gt_raw[:3])

    print(f"\n  GT           : X={gt_raw[0]:.2f}  Y={gt_raw[1]:.2f}"
          f"  Z={gt_raw[2]:.2f}  α={np.degrees(gt_raw[3]):.1f}°"
          f"  β={np.degrees(gt_raw[4]):.1f}°")
    print(f"  Pred WITH    : X={pred_w_raw[0]:.2f}  Y={pred_w_raw[1]:.2f}"
          f"  Z={pred_w_raw[2]:.2f}  "
          f"err={err_w:.3f} mm")
    print(f"  Pred W/O     : X={pred_wo_raw[0]:.2f}  Y={pred_wo_raw[1]:.2f}"
          f"  Z={pred_wo_raw[2]:.2f}  "
          f"err={err_wo:.3f} mm")

    # ── Vẽ ──────────────────────────────────────────────────────────────
    make_figure(volt_raw_8x8, heatmap_w, heatmap_wo,
                pred_w_raw, pred_wo_raw, gt_raw,
                idx, args.out)

    cam_w.remove(); cam_wo.remove()


if __name__ == "__main__":
    main()