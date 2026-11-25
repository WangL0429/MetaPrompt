from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

JOB=Path(__file__).resolve().parent
REF=JOB.parent/'mmimdb_joint_full_9aNxqw'
sys.path.insert(0,str(REF))
from model_io import load, plots, outputs
from common import split_movies

@torch.inference_mode()
def extract():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    net,ck=load(REF/'selected.ckpt')
    assert ck['cross_second_meta']['epoch']==6
    cfg=ck['hyper_parameters']['config']
    net.cuda().eval().requires_grad_(False)
    w=net.mmimdb_classifier.weight.detach().float().cpu()
    b=net.mmimdb_classifier.bias.detach().float().cpu()
    assert w.shape==(23,1024)
    train=torch.load(REF/'train_cache.pt',map_location='cpu')
    ids=[str(i).zfill(7) for i in train['ids']]
    posters=np.load(REF/'posters.npy',mmap_mode='r')
    rows=train['rows']
    counts=torch.bincount(rows,minlength=len(ids))
    assert min(counts)>0 and len(set(ids))==len(ids)==len(posters)==15552
    assert counts.sum()==32278
    xs=[]
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(ids),64):
            img=torch.from_numpy(posters[start:start+64].copy()).cuda()
            tokens=torch.zeros((len(img),77),dtype=torch.long,device='cuda')
            _,feature,_=outputs(net,img,tokens,1)
            xs.append(feature.cpu())
    data=dict(x=torch.cat(xs),labels=train['labels'],weight=counts,ids=ids)
    assert data['x'].shape==(15552,512) and data['x'].isfinite().all()
    ds=plots(cfg,'dev')
    dev_ids=[str(i).zfill(7) for i in ds.table['image_id'].to_pylist()]
    assert set(ids).isdisjoint(dev_ids)
    movie=torch.tensor([ds.index_mapper[i][0] for i in range(len(ds))])
    labels=torch.tensor(ds.table['label'].to_pylist()).float()
    cal=split_movies(dev_ids)[movie]
    parts={}
    for key,mask in [('cal',cal),('val',~cal)]:
        indices=mask.nonzero().flatten()
        seq=DataLoader(Subset(ds,indices.tolist()),batch_size=64,num_workers=4,pin_memory=True)
        features=[[],[],[]]
        logits=[[],[],[]]
        with torch.autocast('cuda',dtype=torch.float16):
            for img,text in seq:
                img=img.cuda()
                for state in range(3):
                    inp=dict(image=[torch.zeros_like(img) if state==2 else img],
                             text=(['']*len(img) if state==1 else text,),missing_type=[state]*len(img))
                    f=net.infer(inp)['cls_feats']
                    logits[state].append(net.mmimdb_classifier(f).float().cpu())
                    features[state].append(f.half().cpu())
        allx=torch.stack([torch.cat(a) for a in features])
        z=torch.stack([torch.cat(a) for a in logits])
        assert allx.isfinite().all() and z.isfinite().all()
        assert not allx[1,:,512:].any() and not allx[2,:,:512].any()
        parts[key]=dict(x=allx,z=z,movie=movie[indices],y=labels[movie[indices]])
    assert set(parts['cal']['movie'].tolist()).isdisjoint(parts['val']['movie'].tolist())
    print('CACHE_READY',len(ids),[(k,len(p['movie'].unique()),len(p['y'])) for k,p in parts.items()],flush=True)
    out=JOB/'features.pt'
    torch.save(dict(train=data,dev=parts,w=w,b=b),out.with_suffix('.tmp'))
    out.with_suffix('.tmp').replace(out)

if __name__=='__main__':
    extract()
