"""Refit the same validation-only visual linear probe used in the matched experiment."""
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/'mmimdb_joint_full_9aNxqw'
MATCH=EXP/'mmimdb_matched_probe_obZdDo'
sys.path[:0]=[str(BASE),str(MATCH)]
from model_io import load,plots,outputs
from cache import Posters
import compare

def extract(meta):
    net,ck=load(JOB/'selected.ckpt')
    net.cuda().eval().requires_grad_(False)
    cfg=ck['hyper_parameters']['config']
    original=torch.load(EXP/'mmimdb_visual_prompt_NOSgpO'/'initial.ckpt',map_location='cpu')['state_dict']
    sd=ck['state_dict']
    assert set(original)==set(sd)
    assert all(torch.equal(original[k],sd[k]) for k in sd if not k.startswith('model.prompt_learner.'))
    del ck,original,sd
    data={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for split in ('train','dev'):
            ds=plots(cfg,split)
            ids=[str(i).zfill(7) for i in ds.table['image_id'].to_pylist()]
            assert ids==meta[split]['ids']
            if split=='train':
                arr=np.load(BASE/'posters.npy',mmap_mode='r')
                batches=(torch.from_numpy(arr[i:i+64].copy()) for i in range(0,len(arr),64))
            else:
                first=np.unique(meta[split]['rows'],return_index=True)[1]
                batches=DataLoader(Posters(ds,first.tolist()),batch_size=64,num_workers=4)
            xx=[]; zz=[]
            for batch in batches:
                img=batch.cuda()
                if split=='dev':
                    img=img.half().float()
                # Follow the matched-probe extraction's native prediction path.
                f=net.infer(dict(image=[img],text=(['']*len(img),),missing_type=[1]*len(img)))['cls_feats']
                assert not f[:,512:].any()
                xx.append(f[:,:512].half().cpu().numpy())
                if split=='dev':
                    zz.append(net.mmimdb_classifier(f).float().cpu().numpy())
            data[split+'_x']=np.concatenate(xx)
            if zz: data['dev_z']=np.concatenate(zz)
    np.savez(JOB/'rank.npz',**data)

def main():
    torch.set_num_threads(2)
    with (MATCH/'metadata.pkl').open('rb') as f: meta=pickle.load(f)
    extract(meta)
    compare.JOB=JOB
    header,records=compare.calibration(meta,'rank')
    compare.write('probe_calibration.csv',records)
    torch.save(header,JOB/'probe.pt')
    dev=meta['dev']
    keep=~dev['cal']
    y=dev['labels'][keep]
    weight=np.bincount(dev['rows'])[keep]
    with np.load(JOB/'rank.npz') as arr:
        z=compare.logits(arr['dev_x'][keep],header['w'],header['b'])
        native=arr['dev_z'][keep]
    ap=compare.ap(y,z,weight)
    pred=z.astype(float)>header['cuts']
    f1=float(compare.fscore(compare.counts(y,pred,weight).sum(0)))
    import csv
    with (MATCH/'validation.csv').open() as f: prior=list(csv.DictReader(f))
    out=[dict(model=r['arm'],ridge='',probe_AP=float(r['poster_macro_AP']),probe_F1=float(r['poster_macro_F1']))
         for r in prior if r['readout']=='probe_calibrated']
    out.append(dict(model='rank',ridge=header['ridge'],probe_AP=ap,probe_F1=f1))
    compare.write('probe_validation.csv',out)
    print(out,flush=True)

if __name__=='__main__': main()
