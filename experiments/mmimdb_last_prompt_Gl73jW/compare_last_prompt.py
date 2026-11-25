"""Paired F1 and visual average-precision changes on held-out movies."""
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from semantic_io import JOB,setup,csv_out
from compare_semantics import compare,read_csv
from fit_semantics import read,get_weights,predictions
from predictor import load_predictor_checkpoint
from train_last_prompt import score_split


def bootstrap_ap(scores,labels,rows):
    n=int(rows.max())+1
    rng=np.random.default_rng(1722)
    draws=rng.multinomial(n,np.full(n,1/n),size=500).astype(np.int16)
    estimates={}
    for name,z in scores.items():
        ap=np.zeros(len(draws))
        for c in range(z.shape[1]):
            order=np.argsort(-z[:,c],kind="stable")
            sorted_scores=z[order,c]
            ends=np.r_[np.flatnonzero(np.diff(sorted_scores)),len(rows)-1]
            y=labels[order,c]
            for i in range(0,len(draws),20):
                weights=draws[i:i+20,rows[order]]
                positives=(weights*y).cumsum(1)[:,ends]
                predicted=weights.cumsum(1)[:,ends]
                steps=np.diff(positives,axis=1,prepend=0)
                ap[i:i+20]+=(steps*positives/np.maximum(predicted,1)).sum(1)/np.maximum(positives[:,-1],1)
        estimates[name]=100*ap/z.shape[1]
    points={n:100*average_precision_score(labels,z,average="macro") for n,z in scores.items()}
    output=[]
    for rule in ("mean9","text90","epoch20"):
        arm="joint_"+rule
        for ref in ("start","control_"+rule):
            lo,hi=np.percentile(estimates[arm]-estimates[ref],[2.5,97.5])
            output.append(dict(metric="image_only_macro_AP",candidate=arm,reference=ref,
                               delta_pp=points[arm]-points[ref],lower95=lo,upper95=hi,draws=len(draws)))
    return output


def main():
    setup()
    arms={n:JOB/n for n in ("control","joint")}
    anchor=torch.load(JOB/"previous/test_epoch_11.pt",map_location="cpu").numpy()
    pairs=[(f"joint_{s}",ref) for s in ("mean9","text90") for ref in ("start",f"control_{s}")]
    # Epoch 20 is prespecified and compared even if neither selector picks it.
    net,gen=load_predictor_checkpoint(JOB/"previous/text90.ckpt")
    test,w=read("test"),get_weights()
    gen.load_state_dict(torch.load(JOB/"control/epoch_20.pt",map_location="cpu"))
    fixed={"start":anchor,"control_epoch20":predictions(gen,test,w)[0].numpy()}
    state=torch.load(JOB/"joint/epoch_20.pt",map_location="cpu")
    gen.load_state_dict(state["weights"])
    fixed["joint_epoch20"]=score_split(net,gen,state["delta"].cuda(),"test",test,w).numpy()
    del net,gen,test
    pairs.append(("joint_epoch20","control_epoch20"))
    compare(arms,pairs,fixed)
    z={k:v[:,1] for k,v in fixed.items()}
    for name,path in arms.items():
        selection=json.loads((path/"selection.json").read_text())
        for rule,epoch in selection["picks"].items():
            z[name+"_"+rule]=torch.load(path/f"test_epoch_{epoch}.pt",map_location="cpu").numpy()[:,1]
    d=torch.load(JOB/"features_test.pt",map_location="cpu")
    csv_out(JOB/"ap_ci.csv",bootstrap_ap(z,d["y"].numpy(),d["rows"].numpy()))
    prior={(r["mode"],r["rate"]):r for r in read_csv((JOB/"control").resolve().parent/"test_grid.csv")}
    for row in read_csv(JOB/"test_grid.csv"):
        ref=prior[row["mode"],row["rate"]]
        assert all(abs(float(row[k])-float(ref[k]))<1e-7 for k in ("control_mean9","control_text90","control_epoch20","start","DPL"))


if __name__=="__main__":main()
