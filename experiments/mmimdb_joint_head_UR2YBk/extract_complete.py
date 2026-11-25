from pathlib import Path
import sys
import numpy as np
import torch

JOB=Path(__file__).resolve().parent
REF=JOB.parent/'mmimdb_joint_full_9aNxqw'
PROBE=JOB.parent/'mmimdb_poster_probe_S5KTyc'
sys.path.insert(0,str(REF))
from model_io import load, outputs, plots

@torch.inference_mode()
def extract():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    net,ck=load(REF/'selected.ckpt')
    assert ck['cross_second_meta']['epoch']==6
    net.cuda().eval().requires_grad_(False)
    cfg=ck['hyper_parameters']['config']
    train=torch.load(REF/'train_cache.pt',map_location='cpu')
    probe=torch.load(PROBE/'features.pt',map_location='cpu')
    assert [str(i).zfill(7) for i in train['ids']]==probe['train']['ids']
    assert torch.equal(train['labels'],probe['train']['labels'])
    assert torch.equal(probe['w'],net.mmimdb_classifier.weight.cpu()) and torch.equal(probe['b'],net.mmimdb_classifier.bias.cpu())
    posters=np.load(REF/'posters.npy',mmap_mode='r')
    ds=plots(cfg,'train')
    assert [str(i).zfill(7) for i in ds.table['image_id'].to_pylist()]==probe['train']['ids']
    rows=train['rows']; tokens=train['tokens']
    assert len(rows)==32278 and len(ds)==len(tokens)==len(rows)
    assert torch.equal(rows,torch.tensor([ds.index_mapper[i][0] for i in range(len(ds))]))
    assert torch.equal(probe['train']['weight'],torch.bincount(rows))
    chunks=[]
    errors=[]
    with torch.autocast('cuda',dtype=torch.float16):
        for start in range(0,len(rows),64):
            sub=rows[start:start+64]
            img=torch.from_numpy(posters[sub.numpy()]).cuda()
            logits,visual,txt=outputs(net,img,tokens[start:start+64].cuda(),0)
            both=torch.cat((visual,txt),1)
            if start==0:
                sentences=[ds.get_text(i)['text'] for i in range(len(sub))]
                batch=dict(image=[img],text=(sentences,),missing_type=[0]*len(sub))
                native=net.mmimdb_classifier(net.infer(batch)['cls_feats']).float()
                torch.testing.assert_close(native,logits,atol=.01,rtol=.001)
                errors.append(float((native-logits).abs().max()))
            chunks.append(both.half().cpu())
    x=torch.cat(chunks)
    assert x.shape==(32278,1024) and x.isfinite().all()
    path=JOB/'complete.pt'
    torch.save(dict(x=x,rows=rows,source=str(REF/'selected.ckpt')),path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    print('COMPLETE_FEATURES',x.shape,'native error',errors,flush=True)

if __name__=='__main__': extract()
