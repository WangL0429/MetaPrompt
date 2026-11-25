"""Cache label-free native embeddings and local tokens, preserving the original movie splits."""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from semantic_io import JOB,SOURCE,setup,metadata,load,plots,Capture,summarize_tokens,store


@torch.inference_mode()
def main(splits):
    setup()
    net,ck=load(SOURCE)
    cfg=ck["hyper_parameters"]["config"]
    del ck
    net.cuda().eval().requires_grad_(False)
    info=metadata()
    for split in splits:
        out=JOB/f"features_{split}.pt"
        if out.exists():continue
        ds=plots(cfg,split)
        m=info[split]
        ids=[str(s).zfill(7) for s in ds.table["image_id"].to_pylist()]
        rows=np.fromiter((ds.index_mapper[i][0] for i in range(len(ds))),dtype=np.int64,count=len(ds))
        assert ids==m["ids"] and np.array_equal(rows,m["rows"])
        assert np.array_equal(ds.table["label"].to_pylist(),m["labels"])
        d=dict(rows=torch.from_numpy(rows),y=torch.as_tensor(m["labels"])[rows])
        if split=="dev":d["cal"]=torch.from_numpy(m["cal"][rows])
        n=len(ds)
        for mode in (1,2):
            d[f"h{mode}"]=torch.empty(n,512,dtype=torch.float16)
            d[f"x{mode}"]=torch.empty(n,9,512,dtype=torch.float16)
            d[f"mask{mode}"]=torch.empty(n,9,dtype=torch.bool)
        d["complete"]=torch.empty(n,1024,dtype=torch.float16)
        d["base"]=torch.empty(n,3,23,dtype=torch.float32)
        i=0
        with torch.autocast("cuda",dtype=torch.float16):
            for image,text in DataLoader(ds,batch_size=64,num_workers=5,pin_memory=True):
                image=image.cuda(non_blocking=True)
                sl=slice(i,i+len(image))
                for state in (0,1,2):
                    batch=dict(image=[torch.zeros_like(image) if state==2 else image],
                               text=([""]*len(image) if state==1 else text,),missing_type=[state]*len(image))
                    with Capture(net,{state}) as cap:
                        f=net.infer(batch)["cls_feats"]
                        d["base"][sl,state]=net.mmimdb_classifier(f).float().cpu()
                        if state:
                            x,mask=summarize_tokens(net,cap,state)
                            d[f"x{state}"][sl]=x.cpu()
                            d[f"mask{state}"][sl]=mask.cpu()
                            h=f.chunk(2,1)[state-1]
                            d[f"h{state}"][sl]=h.half().cpu()
                        else:d["complete"][sl]=f.half().cpu()
                i+=len(image)
        assert i==n and d["base"].isfinite().all()
        store(out,d)
        print("cached",split,len(ids),n,flush=True)


if __name__=="__main__":main(sys.argv[1:])
