import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from model_io import JOB, INITIAL, load, plots, outputs, clip


class Posters(Dataset):
    def __init__(self,ds,first):
        self.ds,self.first = ds,first
    def __len__(self):
        return len(self.first)
    def __getitem__(self,i):
        return self.ds.get_image(self.first[i])['image'][0]


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    net,ck=load(INITIAL)
    ds=plots(ck['hyper_parameters']['config'],'train')
    del ck
    net.cuda().eval()
    ids=[str(v) for v in ds.table['image_id'].to_pylist()]
    rows=torch.tensor([ds.index_mapper[i][0] for i in range(len(ds))])
    first=[None]*len(ids)
    for i,row in enumerate(rows.tolist()):
        if first[row] is None:
            first[row]=i
    assert None not in first and len(set(ids))==len(ids)
    sentences=[ds.get_text(i)['text'] for i in range(len(ds))]
    tokens=torch.cat([clip.tokenize(sentences[i:i+128],context_length=77,truncate=True)
                      for i in range(0,len(ds),128)]).long()
    y=torch.tensor(ds.table['label'].to_pylist()).float()
    posters=Posters(ds,first)
    path=JOB/'posters.npy'
    data=np.load(path,mmap_mode='r')
    old=torch.load(path.resolve().parent/'train_cache.pt',map_location='cpu')
    assert ids==old['ids'] and torch.equal(rows,old['rows'])
    assert torch.equal(tokens,old['tokens']) and torch.equal(y,old['labels'])
    assert data.shape==(len(ids),3,224,224)
    assert np.array_equal(data[0],posters[0].numpy())
    del old
    teacher=torch.empty(3,len(ds),y.size(1))
    with torch.autocast('cuda',dtype=torch.float16):
        for state in range(3):
            for i in range(0,len(ds),64):
                idx=rows[i:i+64].numpy()
                img=torch.from_numpy(data[idx]).cuda() if state!=2 else None
                t=tokens[i:i+64].cuda()
                logits,_,_=outputs(net,img,t,state)
                if i==0:
                    batch=dict(image=[img if state!=2 else torch.zeros_like(torch.from_numpy(data[idx]).cuda())],
                               text=(['']*len(t) if state==1 else sentences[:len(t)],),
                               missing_type=[state]*len(t))
                    native=net.mmimdb_classifier(net.infer(batch)['cls_feats']).float()
                    torch.testing.assert_close(logits,native,atol=.02,rtol=.01)
                teacher[state,i:i+64]=logits.cpu()
    assert teacher.isfinite().all()
    out=JOB/'train_cache.pt'
    torch.save(dict(ids=ids,rows=rows,tokens=tokens,labels=y,ref=teacher),out.with_suffix('.tmp'))
    out.with_suffix('.tmp').replace(out)
    print(f'Cached {len(ids)} training movies and {len(ds)} plots; verified all three native views.',flush=True)

if __name__=='__main__':
    main()
