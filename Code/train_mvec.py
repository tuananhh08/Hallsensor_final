"""Three-phase training entry point for ModNet + magnetic localization."""
from __future__ import annotations
import argparse, os, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from data_mvec import StreamingMinMaxScaler, PosLabelScaler
from loss_mvec import HuberPoseLossMVec
from model import Model

def parser():
    p=argparse.ArgumentParser(description=__doc__); root=Path(__file__).resolve().parent.parent/"Dataset"/"Dataset"
    p.add_argument("--phase",choices=("modnet","locnet","finetune"),required=True)
    p.add_argument("--noisy-voltage",default=str(root/"Grid_data.csv")); p.add_argument("--clean-voltage",default=str(root/"Grid_data_computed.csv")); p.add_argument("--noisy-label",default=str(root/"Grid_points_coordinates.csv"))
    p.add_argument("--synthetic-voltage",default=str(root/"synthetic_grid_data.csv")); p.add_argument("--synthetic-label",default=str(root/"synthetic_grid_coordinates.csv")); p.add_argument("--ckpt-dir",default="./ckpt_mvec"); p.add_argument("--scaler-file")
    p.add_argument("--modnet-checkpoint"); p.add_argument("--locnet-checkpoint"); p.add_argument("--resume")
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--epochs",type=int,default=200); p.add_argument("--lr-modnet",type=float,default=2e-4); p.add_argument("--lr-locnet",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=4.56e-3); p.add_argument("--lambda-mod",type=float,default=.1); p.add_argument("--mod-delta",type=float,default=.05)
    p.add_argument("--lambda-ori",type=float,default=1.); p.add_argument("--delta-xyz",type=float,default=.061); p.add_argument("--lambda-pos",type=float,default=1.); p.add_argument("--lambda-physics",type=float,default=1e-4); p.add_argument("--physics-delta",type=float,default=.002)
    p.add_argument("--calib-physical-csv",default=str(root/"Calibration_Physical_new.csv")); p.add_argument("--calib-alpha-csv",default=str(root/"Calibration_Alpha_new.csv")); p.add_argument("--no-physics",action="store_true"); p.add_argument("--val-ratio",type=float,default=.2); p.add_argument("--seed",type=int,default=42); p.add_argument("--num-workers",type=int,default=0)
    return p

def read(path,width,what):
    a=pd.read_csv(path).apply(pd.to_numeric,errors="coerce").to_numpy(np.float32)
    if a.shape[1]!=width or not np.isfinite(a).all(): raise ValueError(f"{what} must be finite and have {width} columns: {path}")
    return a
def poses(a):
    a=a.copy(); n=np.linalg.norm(a[:,3:],axis=1)
    if (n<1e-12).any(): raise ValueError("Zero magnetic-moment label")
    a[:,3:]/=n[:,None]; return a
def split(n,r,seed):
    q=np.random.default_rng(seed).permutation(n); nv=max(1,int(n*r)); return q[nv:],q[:nv]
class DS(Dataset):
    def __init__(self,*a): self.a=[torch.as_tensor(x,dtype=torch.float32) for x in a]
    def __len__(self): return len(self.a[0])
    def __getitem__(self,i): return tuple(x[i] for x in self.a)
def strip(s): return {k.replace("_orig_mod.",""):v for k,v in s.items()}
def save_scalers(path,s):
    with open(path,"wb") as f: pickle.dump(s,f)
def load_part(module,path,name,device):
    c=torch.load(path,map_location=device,weights_only=False); s=strip(c.get(name+"_state_dict",c.get("model_state_dict",c.get("model",c))))
    if name=="locnet" and any(k.startswith("locnet.") for k in s): s={k[7:]:v for k,v in s.items() if k.startswith("locnet.")}
    missing,unexpected=module.load_state_dict(s,strict=False)
    if missing or unexpected: print(f"Warning loading {name}: missing={list(missing)}, unexpected={list(unexpected)}")
    return c
def save(path,phase,epoch,best,model,opt,sch,scalers,metrics):
    d={"phase":phase,"epoch":epoch,"best_val":best,"metrics":metrics,"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":sch.state_dict(),"preprocessing":{"voltage_space":"minmax_[0,1]","sensor_order":"sensor_1..sensor_64 row-major 8x8","scaler_file":"scalers.pkl"},"scaler_metadata":{"volt_data_min":np.asarray(scalers["volt"].data_min_).tolist(),"volt_data_max":np.asarray(scalers["volt"].data_max_).tolist(),"xyz_mean":np.asarray(scalers["label"].xyz_scaler.mean_).tolist(),"xyz_scale":np.asarray(scalers["label"].xyz_scaler.scale_).tolist()}}
    if phase=="modnet": d["modnet_state_dict"]=model.modnet.state_dict()
    elif phase=="locnet": d["locnet_state_dict"]=model.locnet.state_dict()
    else: d.update({"model_state_dict":model.state_dict(),"modnet_state_dict":model.modnet.state_dict(),"locnet_state_dict":model.locnet.state_dict()})
    torch.save(d,path); save_scalers(Path(path).parent/"scalers.pkl",scalers)

def main():
    c=parser().parse_args(); random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    if not 0<c.val_ratio<1: raise ValueError("--val-ratio must be in (0,1)")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); out=Path(c.ckpt_dir); out.mkdir(parents=True,exist_ok=True)
    noisy,clean=read(c.noisy_voltage,64,"Noisy voltage"),read(c.clean_voltage,64,"Clean voltage"); synth,sy=read(c.synthetic_voltage,64,"Synthetic voltage"),poses(read(c.synthetic_label,6,"Synthetic labels")); y=poses(read(c.noisy_label,6,"Noisy pose labels"))
    if len(noisy)!=len(clean): raise ValueError("Noisy/clean voltage row mismatch")
    if len(y)!=len(noisy) and c.phase=="finetune": raise ValueError("Finetune requires paired noisy pose labels")
    if len(synth)!=len(sy): raise ValueError("Synthetic voltage/label row mismatch")
    ta,va=split(len(noisy),c.val_ratio,c.seed); tb,vb=split(len(synth),c.val_ratio,c.seed+1); sp=c.scaler_file or str(out/"scalers.pkl")
    if os.path.exists(sp):
        with open(sp,"rb") as f: scalers=pickle.load(f)
    else:
        vs=StreamingMinMaxScaler((0,1)); [vs.partial_fit(x) for x in (noisy[ta],clean[ta],synth[tb])]; ls=PosLabelScaler(); ls.partial_fit(np.concatenate((y[ta],sy[tb]))); scalers={"volt":vs,"label":ls,"label_format":"mvec","voltage_space":"minmax_[0,1]"}; save_scalers(out/"scalers.pkl",scalers)
    vs,ls=scalers["volt"],scalers["label"]; noisy,clean,synth=[vs.transform(x).reshape(-1,1,8,8).astype(np.float32) for x in (noisy,clean,synth)]; y,sy=ls.transform(y),ls.transform(sy)
    if c.phase=="modnet": tr,va_ds=DS(noisy[ta],clean[ta]),DS(noisy[va],clean[va])
    elif c.phase=="locnet": tr,va_ds=DS(synth[tb],sy[tb]),DS(synth[vb],sy[vb])
    else: tr,va_ds=DS(noisy[ta],clean[ta],y[ta]),DS(noisy[va],clean[va],y[va])
    tl=DataLoader(tr,batch_size=c.batch_size,shuffle=True,num_workers=c.num_workers); vl=DataLoader(va_ds,batch_size=c.batch_size,num_workers=c.num_workers)
    m=Model(use_modnet=c.phase!="locnet").to(dev)
    if c.phase=="finetune":
        if not(c.modnet_checkpoint and c.locnet_checkpoint): raise ValueError("finetune needs --modnet-checkpoint and --locnet-checkpoint")
        load_part(m.modnet,c.modnet_checkpoint,"modnet",dev); load_part(m.locnet,c.locnet_checkpoint,"locnet",dev)
    groups=[{"params":m.modnet.parameters(),"lr":c.lr_modnet}] if c.phase=="modnet" else ([{"params":m.locnet.parameters(),"lr":c.lr_locnet}] if c.phase=="locnet" else [{"params":m.modnet.parameters(),"lr":c.lr_modnet},{"params":m.locnet.parameters(),"lr":c.lr_locnet}])
    opt=torch.optim.AdamW(groups,weight_decay=c.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,c.epochs)); pw=0 if c.no_physics else c.lambda_physics
    crit=HuberPoseLossMVec(c.lambda_ori,c.delta_xyz,c.lambda_pos,pw,c.physics_delta,c.calib_physical_csv if pw else None,c.calib_alpha_csv if pw else None,vs if pw else None,ls if pw else None).to(dev)
    start,best=1,float("inf")
    if c.resume:
        z=torch.load(c.resume,map_location=dev,weights_only=False); (load_part(m.modnet,c.resume,"modnet",dev) if c.phase=="modnet" else load_part(m.locnet,c.resume,"locnet",dev) if c.phase=="locnet" else m.load_state_dict(strip(z["model_state_dict"]),strict=False)); opt.load_state_dict(z["optimizer_state_dict"]); sch.load_state_dict(z["scheduler_state_dict"]); start,best=z["epoch"]+1,z["best_val"]
    names={"modnet":"modnet_pretrained.pt","locnet":"locnet_pretrained.pt","finetune":"full_model_best.pt"}; last={"modnet":"modnet_last.pt","locnet":"locnet_last.pt","finetune":"full_model_last.pt"}
    def epoch(loader,training):
        m.train(training); r={k:0. for k in ("total","loc","mod","mae")}; count=0
        for b in loader:
            b=[q.to(dev) for q in b]
            if training: opt.zero_grad(set_to_none=True)
            if c.phase=="modnet": corrected=m.modnet(b[0]); mod=F.huber_loss(corrected,b[1],delta=c.mod_delta); loc=mod*0; loss=mod
            elif c.phase=="locnet": corrected=b[0]; pred=m.locnet(b[0]); loc,*_=crit(pred,b[1],X_b=b[0]); mod=loc*0; loss=loc
            else: pred,aux=m(b[0],return_features=True); corrected=aux["corrected"]; loc,*_=crit(pred,b[2],X_b=corrected); mod=F.huber_loss(corrected,b[1],delta=c.mod_delta); loss=loc+c.lambda_mod*mod
            if training: loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
            n=len(b[0]); r["total"]+=loss.item()*n; r["loc"]+=loc.item()*n; r["mod"]+=mod.item()*n; r["mae"]+=(corrected-b[1]).abs().mean().item()*n; count+=n
        return {k:v/count for k,v in r.items()}
    print(f"Training {c.phase} on {dev}; ModNet operates in saved min-max normalized voltage space.")
    for e in range(start,c.epochs+1):
        a=epoch(tl,True)
        with torch.no_grad(): b=epoch(vl,False)
        sch.step(); metrics={"train":a,"val":b,"lr":[g["lr"] for g in opt.param_groups]}; print(f"{e:03d} train={a['total']:.6f} val={b['total']:.6f} loc={b['loc']:.6f} mod={b['mod']:.6f} mae={b['mae']:.6f}")
        save(out/last[c.phase],c.phase,e,best,m,opt,sch,scalers,metrics)
        if b["total"]<best: best=b["total"]; save(out/names[c.phase],c.phase,e,best,m,opt,sch,scalers,metrics)
    print(f"Best checkpoint: {out/names[c.phase]} (val={best:.6f})")
if __name__=="__main__": main()
