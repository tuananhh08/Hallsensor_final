"""Colab-friendly three-stage CRLocNet benchmark launcher.

Example (run at Hallsensor_final root)::

  python -m Benchmark.CRLocNet.run --stage modnet --ckpt-dir Benchmark/CRLocNet/ckpt --cache-dir Benchmark/CRLocNet/data_cache
  python -m Benchmark.CRLocNet.run --stage locnet --ckpt-dir Benchmark/CRLocNet/ckpt --cache-dir Benchmark/CRLocNet/data_cache
  python -m Benchmark.CRLocNet.run --stage finetune --modnet-checkpoint Benchmark/CRLocNet/ckpt/modnet_best.pt --locnet-checkpoint Benchmark/CRLocNet/ckpt/locnet_best.pt --ckpt-dir Benchmark/CRLocNet/ckpt --cache-dir Benchmark/CRLocNet/data_cache

The default input is the same 64-column data used for AMagPoseNet.  To use
the paper's multi-source condition, pass row-aligned ``--raw-epm`` and
``--synthetic-epm`` theoretical EPM CSVs.  Their absence means zero EPM,
which is appropriate only for a single-source baseline.
"""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"Network_Architecture"))
from train_mvec import MultiMemmapDataset, prepare_cache
from training_loop import default_num_workers, make_loaders
from Benchmark.CRLocNet.training import train_modnet, pretrain_locnet, finetune
from Benchmark.CRLocNet.losses import DipoleForward, ScaledDipoleForward

def path(x): return Path(x).expanduser().resolve()
def epm_csv(p, scaler, n):
    if p is None: return np.zeros((n,1,8,8),np.float32)
    frame=pd.read_csv(p,header=None)
    try: frame.iloc[0].astype(float)
    except (ValueError,TypeError): frame=pd.read_csv(p)
    a=frame.apply(pd.to_numeric,errors="coerce").to_numpy(np.float32)
    if a.shape != (n,64) or not np.isfinite(a).all(): raise ValueError(f"{p}: expected exactly ({n}, 64) finite EPM values")
    return scaler.transform(a).reshape(-1,1,8,8).astype(np.float32)
class AddEPM(Dataset):
    def __init__(self, base, epm, mode): self.base,self.epm,self.mode=base,epm,mode
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        item=self.base[i]; e=torch.from_numpy(self.epm[self.base.indices[i]].copy())
        if self.mode=="locnet": return item[0],e,item[1]
        if self.mode=="finetune": return item[0],item[1],e,item[2]
        return item
    def __getattr__(self,name): return getattr(self.base,name)
def args():
 d=ROOT/"Dataset"; p=argparse.ArgumentParser(description=__doc__); p.add_argument("--stage",required=True,choices=("modnet","locnet","finetune"));
 for k,v in (("raw-voltage",d/"Grid_add_Random_data.csv"),("clean-voltage",d/"Grid_add_Random_data_computed.csv"),("raw-label",d/"Grid_add_Random_points_coordinates.csv"),("synthetic-voltage",d/"synthetic_grid_data.csv"),("synthetic-label",d/"synthetic_grid_coordinates.csv")): p.add_argument("--"+k,type=path,default=v)
 p.add_argument("--raw-epm",type=path); p.add_argument("--synthetic-epm",type=path); p.add_argument("--ckpt-dir",type=path,required=True); p.add_argument("--cache-dir",type=path,required=True); p.add_argument("--modnet-checkpoint",type=path); p.add_argument("--locnet-checkpoint",type=path); p.add_argument("--epochs",type=int,default=200); p.add_argument("--batch-size",type=int,default=None); p.add_argument("--batch-real",type=int,default=32); p.add_argument("--batch-synthetic",type=int,default=128); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--lr-modnet",type=float,default=1e-4); p.add_argument("--lr-locnet",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=4.56e-3); p.add_argument("--warmup-epochs",type=int,default=5); p.add_argument("--eta-min",type=float,default=1e-6); p.add_argument("--patience",type=int,default=45); p.add_argument("--save-every",type=int,default=5); p.add_argument("--grad-clip",type=float,default=1.); p.add_argument("--noise-std",type=float,default=0.); p.add_argument("--lambda-direction",type=float,default=.2); p.add_argument("--lambda-physics",type=float,default=1e-4); p.add_argument("--lambda-alignment",type=float,default=.1); p.add_argument("--physics-informed",action=argparse.BooleanOptionalAction,default=False,help="Enable L_ph only after setting the real sensor geometry/moment below."); p.add_argument("--sensor-pitch-m",type=float,default=.01); p.add_argument("--moment-am2",type=float,default=1.); p.add_argument("--measurement-axis",type=int,default=2); p.add_argument("--val-ratio",type=float,default=.2); p.add_argument("--seed",type=int,default=42); p.add_argument("--num-workers",type=int,default=None); p.add_argument("--compile",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--rebuild-cache",action="store_true"); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); return p.parse_args()
def main():
 a=args()
 if a.stage=="finetune" and not (a.modnet_checkpoint and a.locnet_checkpoint): raise ValueError("finetune requires --modnet-checkpoint and --locnet-checkpoint")
 for f in (a.raw_voltage,a.clean_voltage,a.raw_label,a.synthetic_voltage,a.synthetic_label):
  if not f.is_file(): raise FileNotFoundError(f)
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device)); a.ckpt_dir.mkdir(parents=True,exist_ok=True)
 cfg=SimpleNamespace(raw_voltage=str(a.raw_voltage),clean_voltage=str(a.clean_voltage),raw_label=str(a.raw_label),synthetic_voltage=str(a.synthetic_voltage),synthetic_label=str(a.synthetic_label),val_ratio=a.val_ratio,seed=a.seed,rebuild_cache=a.rebuild_cache)
 arrays,split,scalers=prepare_cache(cfg,a.cache_dir,None); raw_epm=epm_csv(a.raw_epm,scalers["volt"],len(np.load(arrays[0],mmap_mode="r"))); synth_epm=epm_csv(a.synthetic_epm,scalers["volt"],len(np.load(arrays[3],mmap_mode="r")))
 if a.stage=="modnet": train,val=MultiMemmapDataset(arrays[:2],split["train_a"]),MultiMemmapDataset(arrays[:2],split["val_a"]); bs=a.batch_size or a.batch_synthetic
 elif a.stage=="locnet": train,val=AddEPM(MultiMemmapDataset([arrays[3],arrays[4]],split["train_b"]),synth_epm,"locnet"),AddEPM(MultiMemmapDataset([arrays[3],arrays[4]],split["val_b"]),synth_epm,"locnet"); bs=a.batch_size or a.batch_synthetic
 else: train,val=AddEPM(MultiMemmapDataset(arrays[:3],split["train_a"]),raw_epm,"finetune"),AddEPM(MultiMemmapDataset(arrays[:3],split["val_a"]),raw_epm,"finetune"); bs=a.batch_size or a.batch_real
 loaders=make_loaders(train,val,bs,default_num_workers(dev) if a.num_workers is None else a.num_workers,dev,2,0)
 with open(a.ckpt_dir/"scalers.pkl","wb") as f: pickle.dump(scalers,f)
 (a.ckpt_dir/"run_config.json").write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
 common=dict(ckpt_dir=a.ckpt_dir,epochs=a.epochs,weight_decay=a.weight_decay,warmup=a.warmup_epochs,eta_min=a.eta_min,patience=a.patience,save_every=a.save_every,device=dev,grad_clip=a.grad_clip,compile_model=a.compile)
 if a.stage=="modnet": best=train_modnet(*loaders,lr=a.lr,noise_std=a.noise_std,**common)
 elif a.stage=="locnet": best=pretrain_locnet(*loaders,lr=a.lr,lambda_direction=a.lambda_direction,**common)
 else:
  physics=None
  if a.physics_informed:
   dipole=DipoleForward.planar_8x8(pitch_m=a.sensor_pitch_m,moment_am2=a.moment_am2,measurement_axis=a.measurement_axis)
   physics=ScaledDipoleForward(dipole,scalers["label"].xyz_scaler.mean_,scalers["label"].xyz_scaler.scale_,scalers["volt"].data_min_,scalers["volt"].data_max_).to(dev)
  else: print("[physics] disabled. Enable only with calibrated sensor coordinates, moment, and tesla-valued field CSVs.")
  best=finetune(*loaders,modnet_checkpoint=a.modnet_checkpoint,locnet_checkpoint=a.locnet_checkpoint,lr_modnet=a.lr_modnet,lr_locnet=a.lr_locnet,lambda_direction=a.lambda_direction,lambda_physics=a.lambda_physics,lambda_alignment=a.lambda_alignment,physics_forward=physics,**common)
 print(f"Best checkpoint: {best}")
if __name__=="__main__": main()
