"""AMagPoseNet – evaluation/metrics.py

Evaluation metrics, matching test_mvec.py exactly for fair comparison.
"""
from __future__ import annotations

import numpy as np


def pos_error_mm(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Euclidean position error per sample, in millimetres.

    Args:
        pred_xyz: [N, 3] – predicted positions in metres (after inverse_transform)
        gt_xyz:   [N, 3] – ground-truth positions in metres

    Returns:
        [N] array of position errors in mm.
    """
    return np.linalg.norm(pred_xyz - gt_xyz, axis=1) * 1000.0


def ori_error_deg(pred_mvec: np.ndarray, gt_mvec: np.ndarray) -> np.ndarray:
    """Angular orientation error per sample in degrees.

    Computes the angle between predicted and ground-truth unit direction vectors:
        θ = arccos(clamp(m_pred · m_gt, -1, 1))

    Both inputs are L2-normalised internally (safe for raw predictions).

    Args:
        pred_mvec: [N, 3] – predicted orientation (need not be unit-length)
        gt_mvec:   [N, 3] – ground-truth orientation (need not be unit-length)

    Returns:
        [N] array of orientation errors in degrees.
    """
    p_n = pred_mvec / (np.linalg.norm(pred_mvec, axis=1, keepdims=True) + 1e-14)
    g_n = gt_mvec   / (np.linalg.norm(gt_mvec,   axis=1, keepdims=True) + 1e-14)
    dot = np.clip(np.sum(p_n * g_n, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def compute_metrics(
    pred_xyz:  np.ndarray,
    gt_xyz:    np.ndarray,
    pred_mvec: np.ndarray,
    gt_mvec:   np.ndarray,
) -> dict:
    """Compute the full metric suite matching test_mvec.py output format.

    Returns:
        dict with keys:
            pos_errors_mm, ori_errors_deg,
            mean_pos_mm, rmse_pos_mm, max_pos_mm, mae_xyz_mm,
            mean_ori_deg, rmse_ori_deg, max_ori_deg
    """
    p_err = pos_error_mm(pred_xyz, gt_xyz)
    o_err = ori_error_deg(pred_mvec, gt_mvec)
    mae_xyz = np.abs(pred_xyz - gt_xyz).mean(axis=0) * 1000.0

    return {
        "pos_errors_mm":  p_err,
        "ori_errors_deg": o_err,
        "mean_pos_mm":    float(p_err.mean()),
        "rmse_pos_mm":    float(np.sqrt(np.mean(p_err ** 2))),
        "max_pos_mm":     float(p_err.max()),
        "mae_xyz_mm":     mae_xyz,          # [3] – per-axis MAE
        "mean_ori_deg":   float(o_err.mean()),
        "rmse_ori_deg":   float(np.sqrt(np.mean(o_err ** 2))),
        "max_ori_deg":    float(o_err.max()),
    }
