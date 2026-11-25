"""Fit regularized score corrections jointly on paired complete and single-modality views."""
import json
from statistics import mean
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from common import (JOB, SOURCE, KEYS, HEAD, setup, read, parameters, views, project,
                    logits, grid, scenario_columns, write_csv)

STATE_WEIGHTS = (.4,.3,.3)
RIDGES = (.03,.3,3.,30.)
STEPS = (.125,.25,.5,1.)


def design(data,matrices,stage):
    parts=[]
    for v,t in views(data):
        if stage=="post":
            parts.append(project((v,t),matrices))
        else:
            n=len(data["y"])
            if v is None: v=data["v0"].new_zeros(n,768)
            if t is None: t=data["t0"].new_zeros(n,512)
            parts.append(torch.cat((v,t),1))
    return torch.cat(parts).float()


class LogisticFit:
    def __init__(self,x,y,base):
        n=len(y)
        self.mass=x.new_tensor(STATE_WEIGHTS).repeat_interleave(n).div_(n)
        self.scale=(self.mass @ x.square()).sqrt().clamp_min_(1e-6)
        self.x=x.div_(self.scale)
        self.gram=self.x.T @ (self.mass[:,None]*self.x)
        rate=y.float().mean(0)
        self.positive=((1-rate)/rate.clamp_min(1e-6)).sqrt().clamp_(1,8)
        self.y=y.repeat(3,1).float()
        self.weight=1+(self.positive-1)*self.y
        self.base=base.flatten(0,1)
        assert self.base.shape==self.y.shape

    def solve(self,ridge):
        # Precondition the optimizer only. The penalty acts on RMS-scaled original coordinates.
        d=self.x.shape[1]
        factor=torch.linalg.cholesky(.25*self.gram+ridge*torch.eye(d,device=self.x.device))
        whitener=torch.linalg.solve_triangular(factor,torch.eye(d,device=self.x.device),upper=False).T.contiguous()
        xw=(self.x @ whitener).contiguous()
        shape=(d,self.y.shape[1])
        def evaluate(flat):
            u=torch.from_numpy(flat.reshape(shape)).cuda().float()
            delta=whitener @ u
            z=self.base+xw@u
            loss=F.binary_cross_entropy_with_logits(z,self.y,pos_weight=self.positive,reduction="none")
            value=(self.mass[:,None]*loss).sum(dtype=torch.float64)+.5*ridge*delta.double().square().sum()
            dz=(z.sigmoid()-self.y)*self.weight*self.mass[:,None]
            gradient=xw.T @ dz+ridge*whitener.T @ delta
            return value.item(),gradient.double().cpu().numpy().ravel()
        opt=minimize(evaluate,np.zeros(np.prod(shape)),method="L-BFGS-B",jac=True,
                     options=dict(maxiter=180,maxls=35,gtol=5e-7,ftol=1e-12))
        delta=whitener @ torch.from_numpy(opt.x.reshape(shape)).cuda().float()
        z=self.base+self.x@delta
        dz=(z.sigmoid()-self.y)*self.weight*self.mass[:,None]
        g=(self.x.T@dz+ridge*delta).abs().max().item()
        assert np.isfinite(opt.fun) and g<1e-4,(ridge,g,opt.message)
        return delta/self.scale[:,None],dict(iterations=int(opt.nit),grad_max=g,objective=opt.fun)


def folded(matrices,weight,coef,stage):
    split=768 if stage=="pre" else 512
    changes=[]
    for a,p,w in zip(coef.split((split,512)),matrices,weight.split(512,dim=1)):
        # Original class logits use h P W^T. Solve delta_P W^T = A in minimum norm.
        w=w.double()
        target=a.double() if stage=="pre" else p.double()@a.double()
        change=target @ torch.linalg.pinv(w.T)
        torch.testing.assert_close(change @ w.T,target,rtol=1e-8,atol=1e-9)
        changes.append(change)
    return tuple(changes)


def matrix_at(matrices,change,step):
    return tuple((p.double()+step*d).to(p.dtype) for p,d in zip(matrices,change))


def record(ridge,step,values,base):
    diffs=np.subtract(values,base)
    return dict(ridge=ridge,step=step,eligible=bool(min(diffs)>=-.25 and mean(values)>=mean(base)),
                text90=values[-1],mean=mean(values),worst_drop=float(min(diffs)),**scenario_columns(values))


def main(stage):
    setup()
    ck=torch.load(SOURCE,map_location="cpu")
    sd=ck["state_dict"]
    matrices,head=parameters(sd)
    train,dev=read("train"),read("dev")
    train_base=logits(train,matrices,head)
    z0=logits(dev,matrices,head)
    keep=dev["cal"]
    cal_base=grid(z0,dev,keep)
    validation_base=grid(z0,dev,~keep)
    archived=json.loads((SOURCE.parent/"selection.json").read_text())["selected"]
    for key,value in scenario_columns(validation_base).items():
        mode,rate=key.split("_")
        assert abs(value-archived[f"{mode}_{int(rate)/100}"])<.025,(key,value)
    fitter=LogisticFit(design(train,matrices,stage),train["y"],train_base)
    records=[record(0,0,cal_base,cal_base)]
    best=records[0]
    best_nonzero=None
    best_params=matrices
    nonzero_params=None
    convergence=[]
    for ridge in RIDGES:
        coef,details=fitter.solve(ridge)
        convergence.append(dict(ridge=ridge,**details))
        change=folded(matrices,sd[HEAD+".weight"].cuda(),coef,stage)
        for step in STEPS:
            candidate=matrix_at(matrices,change,step)
            z=logits(dev,candidate,head)
            row=record(ridge,step,grid(z,dev,keep),cal_base)
            records.append(row)
            key=(row["eligible"],row["text90"],row["mean"])
            if best_nonzero is None or key>(best_nonzero["eligible"],best_nonzero["text90"],best_nonzero["mean"]):
                best_nonzero,nonzero_params=row,candidate
            if key>(best["eligible"],best["text90"],best["mean"]):
                best,best_params=row,candidate
    save_mats=lambda ps:[p.detach().cpu() for p in ps]
    torch.save(dict(selected=save_mats(best_params),trained=save_mats(nonzero_params)),JOB/f"{stage}_matrices.pt")
    if best["step"]:
        for name,p in zip(KEYS,best_params):
            sd[name]=p.cpu()
        ck["joint_score_residual"]=dict(stage=stage,selected=best,source=str(SOURCE))
        torch.save(ck,JOB/f"{stage}.ckpt")
    else:
        (JOB/f"{stage}.ckpt").symlink_to(SOURCE)
    scores={}
    for name,ps in (("base",matrices),("selected",best_params),("trained",nonzero_params)):
        z=logits(dev,ps,head).cpu()
        scores[name]=z[:,~keep.cpu()]
    torch.save(dict(scores=scores,rows=dev["rows"][~keep].cpu(),y=dev["y"][~keep].cpu()),JOB/f"{stage}_validation.pt")
    v={name:grid(logits(dev,ps,head),dev,~keep) for name,ps in (("selected",best_params),("trained",nonzero_params))}
    summary=dict(stage=stage,selected=best,trained=best_nonzero,validation=v,baseline=validation_base,
                 num_coefficients=fitter.x.shape[1]*23,fit_plots=len(train["y"]),cal_movies=int(dev["rows"][keep].unique().numel()),
                 val_movies=int(dev["rows"][~keep].unique().numel()),convergence=convergence)
    (JOB/f"{stage}_selection.json").write_text(json.dumps(summary,indent=2))
    write_csv(f"{stage}_calibration.csv",records)
    print(stage,best["ridge"],best["step"],"held-out text90",v["selected"][-1],flush=True)


if __name__=="__main__":
    import sys
    main(sys.argv[1])
