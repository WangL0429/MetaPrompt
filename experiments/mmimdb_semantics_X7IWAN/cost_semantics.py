"""Count supported arithmetic operations on the same single-modality raw input, per sample."""
import json
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from semantic_io import JOB, plots, setup, csv_out
from predictor import load_predictor_checkpoint, infer

@torch.inference_mode()
def count(fn):
    fn()
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
        fn()
    return sum(e.flops or 0 for e in p.key_averages())

@torch.inference_mode()
def main():
    setup()
    results = []
    for k in (1,4):
        model, generator = load_predictor_checkpoint(JOB/f"k{k}/mean9.ckpt")
        img, texts = next(iter(DataLoader(plots(model.hparams.config,"dev"),batch_size=16)))
        img = img.cuda()
        for state in (0,1,2):
            batch = dict(image=[torch.zeros_like(img) if state==2 else img],
                         text=([""]*len(img) if state==1 else texts,),missing_type=[state]*len(img))
            def native():
                with torch.autocast("cuda",dtype=torch.float16):
                    return model.mmimdb_classifier(model.infer(batch)["cls_feats"]).float()
            baseline = count(native)
            total = count(lambda: infer(model,generator,batch))
            results.append(dict(k=k,state=state,batch_size=len(img),base_counted_GFLOPs=baseline/len(img)/1e9,
                model_counted_GFLOPs=total/len(img)/1e9,added_counted_MFLOPs=(total-baseline)/len(img)/1e6))
        del model,generator
    csv_out(JOB/"compute.csv",results)

if __name__=="__main__":
    main()
