from __future__ import annotations
import argparse, csv, pickle, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from Benchmark.CRLocNet.models import CRLocNet
from Benchmark.CRLocNet.evaluation.metrics import compute_metrics

def read(path):
    f=pd.read_csv(path,header=None)
    try: f.iloc[0].astype(float)
    except (ValueError,TypeError): f=pd.read_csv(path)
    return f.apply(pd.to_numeric,errors="coerce").dropna().to_numpy(np.float32)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--test-voltage",required=True); p.add_argument("--test-label",required=True); p.add_argument("--test-epm"); p.add_argument("--ckpt-dir",required=True); p.add_argument("--checkpoint",default="full_model_best.pt"); p.add_argument("--scalers"); p.add_argument("--out",required=True); p.add_argument("--out-csv"); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--device",default="auto",choices=("auto","cpu","cuda")); p.add_argument("--timing-repeats",type=int,default=100); a=p.parse_args()
 raw,labels=read(a.test_voltage),read(a.test_label); n=min(len(raw),len(labels)); raw,labels=raw[:n],labels[:n]
 if raw.shape[1]!=64 or labels.shape[1]!=6: raise ValueError("This benchmark-compatible launcher expects 64 field columns and 6 labels [x,y,z,mx,my,mz].")
 epm=np.zeros_like(raw) if a.test_epm is None else read(a.test_epm)[:n]
 if epm.shape!=raw.shape: raise ValueError("--test-epm must have one matching 64-value row per test sample")
 with open(a.scalers or str(Path(a.ckpt_dir)/"scalers.pkl"),"rb") as f: scalers=pickle.load(f)
 x=torch.tensor(scalers["volt"].transform(raw).reshape(-1,1,8,8)); e=torch.tensor(scalers["volt"].transform(epm).reshape(-1,1,8,8))
 dev=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device)); ck=torch.load(Path(a.ckpt_dir)/a.checkpoint,map_location=dev,weights_only=False)
 model=CRLocNet().to(dev); state=ck.get("model_state_dict",ck); model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in state.items()},strict=True); model.eval()
 count=lambda m:sum(q.numel() for q in m.parameters()); print(f"CRLocNet Evaluation | samples={n} | params: ModNet={count(model.modnet):,}, LocNet={count(model.locnet):,}, total={count(model):,}")
 with torch.no_grad(): model(x[:1].to(dev),e[:1].to(dev))
 if dev.type=="cuda": torch.cuda.synchronize()
 ts=[]
 with torch.no_grad():
  for _ in range(a.timing_repeats):
   if dev.type=="cuda": torch.cuda.synchronize()
   t=time.perf_counter(); model(x[:1].to(dev),e[:1].to(dev))
   if dev.type=="cuda": torch.cuda.synchronize()
   ts.append((time.perf_counter()-t)*1000)
  t=time.perf_counter(); pred=np.concatenate([model(x[i:i+a.batch_size].to(dev),e[i:i+a.batch_size].to(dev)).cpu().numpy() for i in range(0,n,a.batch_size)])
  if dev.type=="cuda": torch.cuda.synchronize()
  batch_ms=(time.perf_counter()-t)*1000
 full=scalers["label"].inverse_transform(pred); gt_dir=labels[:,3:]/(np.linalg.norm(labels[:,3:],axis=1,keepdims=True)+1e-14); m=compute_metrics(full[:,:3],labels[:,:3],pred[:,3:],gt_dir)
 print(f"Mean position error: {m['mean_pos_mm']:.2f} mm | RMSE: {m['rmse_pos_mm']:.2f} mm | Mean orientation: {m['mean_ori_deg']:.3f}° | 1-sample: {np.mean(ts):.3f} ± {np.std(ts):.3f} ms | throughput: {1000/np.mean(ts):.1f} FPS | full: {batch_ms:.1f} ms")
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
 for i in range(n): rows.append([i,*full[i,:3],*labels[i,:3],m['pos_errors_mm'][i],*pred[i,3:],*gt_dir[i],m['ori_errors_deg'][i]])
 with open(a.out_csv or str(Path(a.ckpt_dir)/"testresult_crlocnet.csv"),"w",newline="") as f: csv.writer(f).writerows([["Point","Pred_X","Pred_Y","Pred_Z","GT_X","GT_Y","GT_Z","PosErr_mm","Pred_mx","Pred_my","Pred_mz","GT_mx","GT_my","GT_mz","OriErr_deg"],*rows])
 fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection="3d"); ax.scatter(*labels[:,:3].T,label="Ground truth"); ax.scatter(*full[:,:3].T,marker="x",label="Prediction"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)"); ax.set_title(f"CRLocNet: mean position {m['mean_pos_mm']:.2f} mm, orientation {m['mean_ori_deg']:.2f}°"); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=150)
 for name,err,unit in (("position_error",m['pos_errors_mm'],"mm"),("orientation_error",m['ori_errors_deg'],"°")):
  fig,ax=plt.subplots(figsize=(11,5)); idx=np.arange(1,n+1); ax.plot(idx,err,alpha=.55,label="per sample"); ax.plot(idx,np.cumsum(err)/idx,lw=2,label="running mean"); ax.set(xlabel="Sample index",ylabel=f"Error ({unit})",title=f"CRLocNet {name.replace('_',' ')}"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out.parent/f"crlocnet_{name}.png",dpi=150)
if __name__=="__main__": main()
