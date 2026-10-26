from __future__ import annotations
import json, time
from pathlib import Path
import torch
from torch import nn
from ..models import ModNet, LocNet, CRLocNet
from ..losses import ModNetLoss, CRLocLoss
from .artifacts import save_loss_plot

def _write(path: Path, item: dict):
    history = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(history + [item], indent=2), encoding="utf-8")

def _scheduler(opt, epochs, warmup, eta):
    return torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=.1, total_iters=max(1, warmup)),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs-warmup), eta_min=eta),
    ], milestones=[warmup])

def _save(path, phase, epoch, model, opt, sched, best, metrics):
    raw = getattr(model, "_orig_mod", model)
    item = dict(phase=phase, epoch=epoch, best_val=best, metrics=metrics,
                optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict())
    if phase == "modnet": item["modnet_state_dict"] = raw.state_dict()
    elif phase == "locnet": item["locnet_state_dict"] = raw.state_dict()
    else:
        item.update(model_state_dict=raw.state_dict(), modnet_state_dict=raw.modnet.state_dict(), locnet_state_dict=raw.locnet.state_dict())
    torch.save(item, path)

def _load(module, path, key, device):
    state = torch.load(path, map_location=device, weights_only=False).get(key)
    if state is None: raise KeyError(f"{path} does not contain {key}")
    module.load_state_dict({k.replace("_orig_mod.", ""): v for k,v in state.items()}, strict=True)

def _fit(*, phase, model, loaders, criterion, step_fn, ckpt_dir, epochs, lr_groups, weight_decay, warmup, eta_min, patience, save_every, device, grad_clip, compile_model):
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(lr_groups, weight_decay=weight_decay); sched = _scheduler(opt, epochs, warmup, eta_min)
    use_amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    base = model.to(device); compiled = torch.compile(base) if compile_model and use_amp else base

    best_names = {"modnet": "modnet_pretrained.pt", "locnet": "locnet_pretrained.pt", "crlocnet": "full_model_best.pt"}
    last_names = {"modnet": "modnet_last.pt", "locnet": "locnet_last.pt", "crlocnet": "full_model_last.pt"}
    best, stale = float("inf"), 0; log = ckpt_dir / f"train_log_{phase}.json"; best_path = ckpt_dir / best_names[phase]
    for epoch in range(1, epochs+1):
        stamp=time.time(); values=[]
        for train, loader in ((True,loaders[0]),(False,loaders[1])):
            compiled.train(train); sums={}; seen=0
            with torch.set_grad_enabled(train):
                for batch in loader:
                    batch=[x.to(device, non_blocking=True) for x in batch]
                    if train: opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp): out=step_fn(compiled, batch, criterion)
                    loss=out["total"]
                    if not torch.isfinite(loss): continue
                    if train:
                        scaler.scale(loss).backward(); scaler.unscale_(opt)
                        if grad_clip>0: nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
                        scaler.step(opt); scaler.update()
                    n=len(batch[0]); seen += n
                    for k,v in out.items(): sums[k]=sums.get(k,0.)+float(v.detach())*n
            values.append({k:v/max(1,seen) for k,v in sums.items()})
        sched.step(); entry={"epoch":epoch,"train":values[0],"val":values[1],"lr":opt.param_groups[0]["lr"],"seconds":time.time()-stamp}; _write(log,entry)
        print(f"[{phase}] {epoch:03d} train={values[0]['total']:.6f} val={values[1]['total']:.6f} ({entry['seconds']:.1f}s)")
        _save(ckpt_dir/last_names[phase],phase,epoch,base,opt,sched,best,entry)
        if values[1]["total"] < best:
            best=values[1]["total"]; stale=0; _save(best_path,phase,epoch,base,opt,sched,best,entry); print(f"  -> best: {best:.6f}")
        else: stale += 1
        if epoch % save_every == 0: _save(ckpt_dir/f"{phase}_epoch_{epoch:04d}.pt",phase,epoch,base,opt,sched,best,entry)
        if stale >= patience: print(f"[{phase}] early stopping"); break
    save_loss_plot(log, ckpt_dir / f"loss_plot_{phase}.png", f"CRLocNet {phase} training loss")
    return best_path

def train_modnet(train_loader, val_loader, *, channels=1, width=64, ckpt_dir, epochs=200, lr=1e-4, noise_std=0., **kw):
    model=ModNet(channels,width); loss=ModNetLoss()
    def step(net,batch,criterion):

        raw,target=batch[:2]
        if net.training and noise_std: raw=raw+torch.randn_like(raw)*noise_std
        value=criterion(net(raw),target); return {"total":value,"alignment":value}
        
    return _fit(
    phase="modnet",
    model=model,
    loaders=(train_loader, val_loader),
    criterion=loss,
    step_fn=step,
    ckpt_dir=ckpt_dir,
    epochs=epochs,
    lr_groups=[{"params": model.parameters(), "lr": lr}],
    **kw
)

def pretrain_locnet(train_loader,val_loader,*,channels=1,width=64,ckpt_dir,epochs=200,lr=1e-4,lambda_direction=.2,**kw):
    model=LocNet(channels,width); loss=CRLocLoss(lambda_direction,0.)
    def step(net,batch,criterion):

        field,epm,pose=batch; pred=net(field,epm); total,p,d,ph=criterion(pred,pose); return {"total":total,"position":p,"direction":d,"physics":ph}
        
    return _fit(
    phase="locnet",
    model=model,
    loaders=(train_loader, val_loader),
    criterion=loss,
    step_fn=step,
    ckpt_dir=ckpt_dir,
    epochs=epochs,
    lr_groups=[{"params": model.parameters(), "lr": lr}],
    **kw
)

def finetune(train_loader,val_loader,*,modnet_checkpoint,locnet_checkpoint,channels=1,width=64,ckpt_dir,epochs=200,lr_modnet=1e-4,lr_locnet=1e-4,lambda_direction=.2,lambda_physics=1e-4,lambda_alignment=.1,physics_forward=None,**kw):
    model=CRLocNet(channels,width); _load(model.modnet,modnet_checkpoint,"modnet_state_dict",kw["device"]); _load(model.locnet,locnet_checkpoint,"locnet_state_dict",kw["device"]); loss=CRLocLoss(lambda_direction,lambda_physics)
    def step(net,batch,criterion):
        # Stage 3: jointly update both ModNet and LocNet from real
        # (raw, clean/model-domain, pose) triples.
        raw,model_field,epm,pose=batch; pred,aux=net(raw,epm,True)
        # ``physics_forward`` maps predicted pose to the same scaled field domain.
        pred_field = physics_forward(pred) if physics_forward is not None else None
        total,p,d,ph=criterion(pred,pose,pred_field,aux["corrected"] - epm)
        align=nn.functional.l1_loss(aux["corrected"],model_field)
        return {"total":total+lambda_alignment*align,"position":p,"direction":d,"physics":ph,"alignment":align}
    groups=[{"params":model.modnet.parameters(),"lr":lr_modnet},{"params":model.locnet.parameters(),"lr":lr_locnet}]
    
    return _fit(
    phase="crlocnet",
    model=model,
    loaders=(train_loader, val_loader),
    criterion=loss,
    step_fn=step,
    ckpt_dir=ckpt_dir,
    epochs=epochs,
    lr_groups=groups,
    **kw
)
