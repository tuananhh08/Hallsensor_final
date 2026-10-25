"""Metrics shared with AMagPoseNet, retained locally for standalone use."""
from __future__ import annotations
import numpy as np

def compute_metrics(pred_xyz, gt_xyz, pred_mvec, gt_mvec):
    pos = np.linalg.norm(pred_xyz-gt_xyz, axis=1)*1000.
    p = pred_mvec/(np.linalg.norm(pred_mvec,axis=1,keepdims=True)+1e-14)
    g = gt_mvec/(np.linalg.norm(gt_mvec,axis=1,keepdims=True)+1e-14)
    ori = np.degrees(np.arccos(np.clip((p*g).sum(1),-1,1)))
    return dict(pos_errors_mm=pos,ori_errors_deg=ori,mean_pos_mm=float(pos.mean()),rmse_pos_mm=float(np.sqrt((pos**2).mean())),max_pos_mm=float(pos.max()),mae_xyz_mm=np.abs(pred_xyz-gt_xyz).mean(0)*1000.,mean_ori_deg=float(ori.mean()),rmse_ori_deg=float(np.sqrt((ori**2).mean())),max_ori_deg=float(ori.max()))
