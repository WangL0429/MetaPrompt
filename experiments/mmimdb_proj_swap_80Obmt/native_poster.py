"""Score the deployed classifier on the same held-out poster features used in the probes."""
import csv
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F
from extract import JOB, MATCH, SHARED
sys.path.insert(0,str(MATCH))
from compare import ap, counts, fscore


def main():
    with (MATCH/"metadata.pkl").open("rb") as f:
        meta=pickle.load(f)
    results=[]
    with np.load(JOB/"shared_features.npz") as arrays, np.load(MATCH/"shared.npz") as archive:
        for name,path in (("baseline",SHARED),("trained_rank2",JOB/"lowrank_trained.ckpt")):
            sd=torch.load(path,map_location="cpu")["state_dict"]
            pv=sd["model.image_encoder.proj"].cuda().half()
            w=sd["mmimdb_classifier.weight"].cuda().half()
            b=sd["mmimdb_classifier.bias"].cuda().half()
            del sd
            for split in ("dev","test"):
                ds=meta[split]
                keep=~ds["cal"] if split=="dev" else slice(None)
                y=ds["labels"][keep]
                weight=np.bincount(ds["rows"])[keep]
                x=torch.from_numpy(arrays[split+"_raw"])
                parts=[]
                with torch.inference_mode():
                    for h in x.split(64):
                        v=h.cuda()@pv
                        parts.append(F.linear(torch.cat((v,torch.zeros_like(v)),1),w,b).float().cpu())
                z=torch.cat(parts).numpy()[keep]
                if name=="baseline":
                    np.testing.assert_allclose(z,archive[split+"_z"][keep],rtol=.001,atol=.003)
                results.append(dict(model=name,split=split,poster_AP=ap(y,z,weight),
                                    poster_F1=float(fscore(counts(y,z>0,weight).sum(0)))))
    with (JOB/"native_poster.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()

