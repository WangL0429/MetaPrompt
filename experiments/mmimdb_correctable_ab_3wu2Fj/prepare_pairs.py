"""Fit five movie-disjoint probes per modality and construct a matched training-only pair bank."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
CACHE=EXP/"mmimdb_proj_swap_80Obmt"
MATCH=EXP/"mmimdb_matched_probe_obZdDo"
SOURCE=EXP/"mmimdb_joint_head_UR2YBk/selected.ckpt"
sys.path.insert(0,str(MATCH))
from probe_fit import fit

SEED=517
MIN_GAP=.25
MAX_DIFF=.02


def crossfit(raw,labels,weight,fold):
    output=np.full(labels.shape,np.nan,dtype=np.float32)
    folds=[]
    for f in range(5):
        held=(fold==f)
        train=~held
        assert (labels[train].sum(0)>0).all() and held.any()
        data=dict(x=raw[train],labels=labels[train],weight=weight[train])
        w,b,g=fit(data,torch.zeros(labels.shape[1],raw.shape[1]),torch.zeros(labels.shape[1]),.3)
        z=F.linear(raw[held].float(),w,b)
        scale=z.std(0,unbiased=False).clamp_min(1e-6)
        output[held]=(z/scale).numpy()
        folds.append(dict(held_rows=int(held.sum()),fitted_rows=int(train.sum()),max_gradient=g))
    assert np.isfinite(output).all()
    return output,folds


def reference(d,sd):
    pv,pt=[sd[k].cuda().half() for k in ("model.image_encoder.proj","model.text_encoder.text_projection")]
    w,b=[sd["mmimdb_classifier."+k].cuda().half() for k in ("weight","bias")]
    rows=d["rows"]
    pieces=[[],[],[]]
    for i in range(0,len(rows),64):
        sl=slice(i,i+64)
        ix=rows[sl]
        v=d["v0"][ix].cuda() @ pv
        t=d["t0"][sl].cuda() @ pt
        for s,(a,bx) in enumerate(((v,t),(d["v1"][ix].cuda()@pv,torch.zeros_like(t)),
                                    (torch.zeros_like(v),d["t2"][sl].cuda()@pt))):
            pieces[s].append(F.linear(torch.cat([a,bx],1),w,b).float().cpu())
    return torch.stack([torch.cat(x) for x in pieces])


def make_bank(ref,oof,fold,labels,rows,state):
    rng=np.random.default_rng(SEED+state)
    indices=np.argsort(rows,kind="stable")
    start=np.r_[0,np.bincount(rows).cumsum()]
    def sentences(movies):
        return indices[rng.integers(start[movies],start[movies+1])]
    records=[]
    stats=[]
    for c in range(labels.shape[1]):
        parts=[]
        for f in range(5):
            pos=np.flatnonzero((fold==f)&labels[:,c])
            neg=np.flatnonzero((fold==f)&~labels[:,c])
            if not len(pos) or not len(neg):continue
            p=sentences(rng.choice(pos,16384))
            n=sentences(rng.choice(neg,16384))
            difficulty=ref[n,c]-ref[p,c]
            teacher=oof[p,c]-oof[n,c]
            valid=np.flatnonzero(difficulty>0)
            correct=valid[teacher[valid]>=MIN_GAP]
            if not len(correct):continue
            correct=correct[np.argsort(difficulty[correct])]
            depths=difficulty[correct]
            left=np.searchsorted(depths,difficulty[valid]-MAX_DIFF)
            right=np.searchsorted(depths,difficulty[valid]+MAX_DIFF,side="right")
            in_range=right>left
            a=valid[in_range]
            if not len(a):continue
            b=correct[rng.integers(left[in_range],right[in_range])]
            take=rng.choice(len(a),size=min(640,len(a)),replace=False)
            a,b=a[take],b[take]
            parts.append(np.stack((p[a],n[a],p[b],n[b]),axis=1))
        if not parts:raise ValueError(("no matched pairs",state,c))
        bank=np.concatenate(parts).astype(np.int32)
        pa,na,pb,nb=bank.T
        da,db=ref[na,c]-ref[pa,c],ref[nb,c]-ref[pb,c]
        ta,tb=oof[pa,c]-oof[na,c],oof[pb,c]-oof[nb,c]
        assert np.all(da>0) and np.all(db>0) and np.max(abs(da-db))<=MAX_DIFF+1e-5
        assert np.all(tb>=MIN_GAP) and np.all(fold[rows[pa]]==fold[rows[nb]])
        assert np.all(labels[rows[pa],c]&labels[rows[pb],c]&~labels[rows[na],c]&~labels[rows[nb],c])
        records.append(bank)
        stats.append(dict(state=state,label=c,slots=len(bank),max_hardness_diff=float(abs(da-db).max()),
                          mean_a=float(da.mean()),mean_b=float(db.mean()),teacher_fraction_A=float((ta>=MIN_GAP).mean()),
                          teacher_fraction_B=float((tb>=MIN_GAP).mean()),same_pair=float(((pa==pb)&(na==nb)).mean())))
    return records,stats


def main():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(CACHE/"lowrank_train.pt",map_location="cpu")
    with (MATCH/"metadata.pkl").open("rb") as stream:meta=pickle.load(stream)
    train=meta["train"]
    ids=train["ids"]
    assert set(ids).isdisjoint(meta["dev"]["ids"]) and set(ids).isdisjoint(meta["test"]["ids"])
    rows=d["rows"].numpy()
    labels=train["labels"].astype(bool)
    assert np.array_equal(rows,train["rows"]) and np.array_equal(d["y"],labels[rows])
    fold=np.array([int.from_bytes(hashlib.sha256(("ab/"+m).encode()).digest()[:8],"big")%5 for m in ids])
    o1,fs1=crossfit(d["v1"],torch.from_numpy(labels),torch.as_tensor(np.bincount(rows)),fold)
    o2,fs2=crossfit(d["t2"],d["y"],torch.ones(len(rows)),fold[rows])
    ck=torch.load(SOURCE,map_location="cpu")["state_dict"]
    ref=reference(d,ck)
    with np.load(MATCH/"shared.npz") as stored:
        np.testing.assert_allclose(ref[1].numpy(),stored["train_z"][rows],atol=.008,rtol=.002)
    banks,stats={},[]
    for state,oof in ((1,o1[rows]),(2,o2)):
        bank,counts=make_bank(ref[state].numpy(),oof,fold,labels,rows,state)
        banks[state]=bank
        stats.extend(counts)
    torch.save(dict(reference=ref,banks=banks,rows=torch.tensor(rows),fold=fold,
                    oof={1:o1[rows],2:o2}),JOB/"training_pairs.pt")
    audit=dict(fold_movies=np.bincount(fold).tolist(),probes_image=fs1,probes_text=fs2,
               movie_count=len(ids),train_rows=len(rows),min_teacher_margin_sd=MIN_GAP,
               max_baseline_margin_tolerance=MAX_DIFF,by_class=stats)
    (JOB/"pair_audit.json").write_text(json.dumps(audit,indent=2))
    print("PAIR_BANK_READY",len(stats),sum(r["slots"] for r in stats),flush=True)


if __name__=="__main__":
    main()
