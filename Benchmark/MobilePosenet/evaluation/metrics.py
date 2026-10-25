from __future__ import annotations
import numpy as np

def compute_metrics(pred_xyz, gt_xyz, pred_mvec, gt_mvec):
    pos = np.linalg.norm(pred_xyz - gt_xyz, axis=1) * 1000.
    pred_mvec = pred_mvec / (np.linalg.norm(pred_mvec, axis=1, keepdims=True) + 1e-14); gt_mvec = gt_mvec / (np.linalg.norm(gt_mvec, axis=1, keepdims=True) + 1e-14)
    ori = np.degrees(np.arccos(np.clip((pred_mvec * gt_mvec).sum(1), -1., 1.)))
    return {"pos_errors_mm": pos, "ori_errors_deg": ori, "mean_pos_mm": float(pos.mean()), "std_pos_mm": float(pos.std()), "rmse_pos_mm": float(np.sqrt(np.mean(pos**2))), "max_pos_mm": float(pos.max()), "mae_xyz_mm": np.abs(pred_xyz - gt_xyz).mean(0) * 1000., "mean_ori_deg": float(ori.mean()), "std_ori_deg": float(ori.std()), "rmse_ori_deg": float(np.sqrt(np.mean(ori**2))), "max_ori_deg": float(ori.max())}
