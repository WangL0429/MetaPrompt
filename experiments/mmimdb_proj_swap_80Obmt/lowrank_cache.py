"""Capture both streams immediately before their frozen terminal projections."""
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from extract import JOB, BASE, MATCH, SHARED

sys.path.insert(0,str(BASE))
from model_io import load, plots, clip
from cache import Posters


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    net, ck = load(SHARED)
    cfg = ck["hyper_parameters"]["config"]
    net.cuda().eval().requires_grad_(False)
    with (MATCH/"metadata.pkl").open("rb") as f:
        meta = pickle.load(f)
    train = torch.load(BASE/"train_cache.pt",map_location="cpu")
    ln_img = net.model.image_encoder.ln_post
    ln_txt = net.model.text_encoder.ln_final
    hidden = []
    def capture(m, args, output):
        hidden.append(output)
    with np.load(JOB/"shared_features.npz") as diagnostic, torch.autocast("cuda",dtype=torch.float16):
        for split in ("train","dev"):
            m = meta[split]
            ds = plots(cfg,split)
            assert [str(v).zfill(7) for v in ds.table["image_id"].to_pylist()] == m["ids"]
            rows = m["rows"]
            n = len(rows)
            if split=="train":
                assert np.array_equal(train["rows"].numpy(),rows)
                tok = train["tokens"]
                pics = np.load(BASE/"posters.npy",mmap_mode="r")
                def images():
                    for i in range(0,len(pics),64):
                        yield torch.from_numpy(pics[i:i+64].copy())
            else:
                sents = [ds.get_text(i)["text"] for i in range(n)]
                tok = torch.cat([clip.tokenize(sents[i:i+64],context_length=77,truncate=True)
                                 for i in range(0,n,64)])
                first = np.unique(rows,return_index=True)[1].tolist()
                def images():
                    return DataLoader(Posters(ds,first),batch_size=64,num_workers=4)
            values={"v1":torch.from_numpy(diagnostic[split+"_raw"]),"rows":torch.as_tensor(rows),
                    "y":torch.as_tensor(m["labels"])[rows]}
            enc=net.model.image_encoder
            pi,_=net.model.prompt_learner([0])
            h=ln_img.register_forward_hook(capture)
            parts=[]
            for batch in images():
                k=len(batch)
                out=enc(batch.cuda().to(net.model.dtype),[p.expand(k,-1,-1) for p in pi],[0]*k)
                (pre,)=hidden
                hidden.clear()
                assert torch.equal(pre@enc.proj,out)
                parts.append(pre.half().cpu())
            h.remove()
            values["v0"]=torch.cat(parts)
            for state in (0,2):
                enc=net.model.text_encoder
                _,pt=net.model.prompt_learner([state])
                h=ln_txt.register_forward_hook(capture)
                parts=[]
                for batch in tok.split(64):
                    batch=batch.cuda()
                    k=len(batch)
                    out=enc(batch,[p.expand(k,-1,-1) for p in pt],[state]*k)
                    (seq,)=hidden
                    hidden.clear()
                    pre=seq[torch.arange(k,device=batch.device),batch.argmax(1)].to(net.model.dtype)
                    assert torch.equal(pre@enc.text_projection,out)
                    parts.append(pre.half().cpu())
                h.remove()
                values["t"+str(state)]=torch.cat(parts)
            assert len(values["v0"])==len(values["v1"])==len(m["ids"])
            assert len(values["t0"])==len(values["t2"])==n
            if split=="dev": values["cal"]=torch.from_numpy(m["cal"][rows])
            torch.save(values,JOB/("lowrank_"+split+".pt"))


if __name__=="__main__":
    main()

