"""Evaluate folded projections on identical encoded test items and archived movie masks."""
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import JOB, EXP, SOURCE, KEYS, setup, parameters, decode, load_test_masks, MODES, RATES
from features import DATA, PlotDataset, CLIPransformerSS


def load_models():
    ck=torch.load(SOURCE,map_location="cpu")
    cfg=ck["hyper_parameters"]["config"]
    net=CLIPransformerSS(dict(cfg,test_only=True,load_path="")).eval()
    net.load_state_dict(ck["state_dict"],strict=True)
    base,head=parameters(ck["state_dict"])
    candidates={"base":base}
    aliases={}
    for stage in ("post","pre"):
        ps=torch.load(JOB/f"{stage}_matrices.pt",map_location="cpu")
        selected=torch.load(JOB/f"{stage}.ckpt",map_location="cpu")["state_dict"]
        assert selected.keys()==ck["state_dict"].keys()
        for key,tensor in selected.items():
            assert torch.equal(tensor,ps["selected"][KEYS.index(key)] if key in KEYS else ck["state_dict"][key]),key
        del selected
        for variant in ("selected","trained"):
            name=stage+"_"+variant
            matrices=tuple(p.cuda() for p in ps[variant])
            same=next((k for k,old in candidates.items() if all(torch.equal(a,b) for a,b in zip(old,matrices))),None)
            if same is None:
                candidates[name]=matrices
            else:
                aliases[name]=same
    return net.cuda().requires_grad_(False),cfg,head,candidates,aliases


@torch.inference_mode()
def main():
    setup()
    net,cfg,head,models,aliases=load_models()
    ds=PlotDataset(data_dir=str(DATA),transform_keys=cfg["val_transform_keys"],image_size=224,
                   names=["mmimdb_test"],text_column_name="plots",remove_duplicate=False)
    ids=[str(x).zfill(7) for x in ds.table["image_id"].to_pylist()]
    rows=np.array([ds.index_mapper[i][0] for i in range(len(ds))])
    movie_y=np.array(ds.table["label"].to_pylist(),dtype=np.uint8)
    with (EXP/"mmimdb_matched_probe_obZdDo/metadata.pkl").open("rb") as f:
        archived=pickle.load(f)["test"]
    assert ids==archived["ids"] and np.array_equal(rows,archived["rows"])
    assert np.array_equal(movie_y,archived["labels"])
    masks=load_test_masks(ids,MODES,RATES)
    stream=DataLoader(ds,batch_size=64,num_workers=4,pin_memory=True)
    visual,textual=net.model.image_encoder,net.model.text_encoder
    hidden={}
    def image_hook(m,args,out):
        hidden["v"]=out
    def text_input(m,args):
        hidden["eot"]=args[0].argmax(1)
    def text_hook(m,args,out):
        hidden["t"]=out[torch.arange(len(out),device=out.device),hidden.pop("eot")].to(net.model.dtype)
    handles=[visual.ln_post.register_forward_hook(image_hook),
             textual.register_forward_pre_hook(text_input),textual.ln_final.register_forward_hook(text_hook)]
    output={k:[[],[],[]] for k in models}
    for number,(img,text) in enumerate(stream):
        img=img.cuda(non_blocking=True)
        for state in (0,1,2):
            batch={"image":[torch.zeros_like(img) if state==2 else img],
                   "text":([""]*len(img) if state==1 else text,), "missing_type":[state]*len(img)}
            with torch.autocast("cuda",dtype=torch.float16):
                f=net.infer(batch)["cls_feats"]
                native=net.mmimdb_classifier(f).float()
            view=(hidden.pop("v") if state!=2 else None,hidden.pop("t") if state!=1 else None)
            hidden.clear()
            for name,ps in models.items():
                z=decode(view,ps,head)
                if name=="base":
                    torch.testing.assert_close(z,native,atol=0,rtol=0)
                elif number==0:
                    visual.proj.copy_(ps[0])
                    textual.text_projection.copy_(ps[1])
                    with torch.autocast("cuda",dtype=torch.float16):
                        actual=net.mmimdb_classifier(net.infer(batch)["cls_feats"]).float()
                    torch.testing.assert_close(z,actual,atol=0,rtol=0)
                    visual.proj.copy_(models["base"][0])
                    textual.text_projection.copy_(models["base"][1])
                    hidden.clear()
                output[name][state].append(z.cpu())
    for handle in handles:handle.remove()
    outputs={k:torch.stack([torch.cat(chunks) for chunks in states]) for k,states in output.items()}
    assert all(v.shape==(3,len(rows),23) for v in outputs.values())
    torch.save(dict(outputs=outputs,aliases=aliases,rows=torch.from_numpy(rows),
                    y=torch.from_numpy(movie_y[rows]),masks=masks),JOB/"test_predictions.pt")


if __name__=="__main__":
    main()
