"""Evaluate the selected self-contained weights from raw images/text and verify the cached training path."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from predictor import load_predictor_checkpoint,infer
from semantic_io import JOB,setup,plots,metadata,store,json_out,csv_out,val_grid
from fit_semantics import read,get_weights,predictions
from train_last_prompt import score_split
from last_prompt import LastViews,materialize
from cost_semantics import count


@torch.inference_mode()
def run():
    setup()
    out=JOB/"joint"
    selection=json.loads((out/"selection.json").read_text())
    start=JOB/"previous/text90.ckpt"
    original,_=load_predictor_checkpoint(start)
    cfg=original.hparams.config
    ds=plots(cfg,"test")
    info=metadata()["test"]
    assert [str(i).zfill(7) for i in ds.table["image_id"].to_pylist()]==info["ids"]
    assert np.array_equal([ds.index_mapper[i][0] for i in range(len(ds))],info["rows"])
    test,dev=read("test"),read("dev")
    cal=dev["cal"].cpu()
    w=get_weights()
    diags={}
    costs=[]
    for epoch in sorted(set(selection["picks"].values())):
        net,gen=load_predictor_checkpoint(out/f"deploy_epoch{epoch}.ckpt")
        delta=torch.load(out/f"epoch_{epoch:02}.pt",map_location="cpu")["delta"].cuda()
        z=score_split(original,gen,delta,"dev",dev,w)
        grid=val_grid(z[cal].transpose(0,1),dev["y"].cpu()[cal])
        assert np.allclose(grid,selection["history"][epoch-1]["cal_grid"],atol=1e-5,rtol=0)
        all_scores=[]
        errors={}
        for n,(img,texts) in enumerate(DataLoader(ds,batch_size=64,num_workers=5,pin_memory=True)):
            img=img.cuda(non_blocking=True)
            states=[]
            if n==0:
                batch=materialize(test,LastViews(original,"test"),np.arange(len(img)),delta)
                cached=predictions(gen,batch,w)[0]
            sequence=(0,1,2,3) if n==0 else (0,1,2)
            for state in sequence:
                mode=[state]*len(img) if state<3 else [i%3 for i in range(len(img))]
                im_mask=torch.tensor(mode,device="cuda")==2
                batch=dict(image=[img.masked_fill(im_mask[:,None,None,None],0)],
                           text=(["" if s==1 else t for s,t in zip(mode,texts)],),missing_type=mode)
                actual=infer(net,gen,batch).cpu()
                if n==0:
                    ref=cached[torch.arange(len(img)),mode]
                    errors[str(state)]=(ref-actual).abs().max().item()
                    torch.testing.assert_close(actual,ref,atol=.012,rtol=.003)
                    if not costs or (len(costs)<3 and state<3):
                        if state<3:
                            before=count(lambda:infer(original,gen,batch))
                            after=count(lambda:infer(net,gen,batch))
                            costs.append(dict(state=state,batch_size=len(img),extra_FLOPs_per_sample=(after-before)/len(img),
                                              joint_counted_GFLOPs=after/len(img)/1e9))
                if state<3:states.append(actual)
            all_scores.append(torch.stack(states,1))
        scores=torch.cat(all_scores)
        assert torch.equal(scores[:,0],test["base"][:,0].cpu()) and scores.isfinite().all()
        store(out/f"test_epoch_{epoch}.pt",scores)
        diags[str(epoch)]=dict(raw_cached_max_error=errors)
    json_out(out/"test_diagnostics.json",diags)
    csv_out(JOB/"compute.csv",costs)


if __name__=="__main__":run()
