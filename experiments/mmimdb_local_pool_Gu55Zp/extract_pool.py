import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common_pool import BASE, SOURCE, load, metadata, plots, setup
from pool import Capture, LocalPool, classify, projectors


class Store:
    def __init__(self,tmp,split):
        self.tmp,self.split,self.arrays=tmp,split,{}

    def add(self,name,total,start,value):
        value=value.detach().cpu().numpy()
        if name not in self.arrays:
            self.arrays[name]=np.lib.format.open_memmap(self.tmp/f'{self.split}_{name}.npy','w+',
                                                        dtype=value.dtype,shape=(total,*value.shape[1:]))
        self.arrays[name][start:start+len(value)]=value


def add_state(store,net,cap,h,state,n,i,check):
    x,valid=cap.local(state)
    assert h.isfinite().all() and x.isfinite().all()
    if check:
        original=classify(net.mmimdb_classifier,h,state)
        for kind in ('uniform','weighted'):
            pool=LocalPool(kind).cuda()
            actual=classify(net.mmimdb_classifier,pool(h,x,valid,projectors(net)[state-1]),state)
            torch.testing.assert_close(actual,original,atol=0,rtol=0)
    store.add(f'h{state}',n,i,h.half())
    store.add(f'x{state}',n,i,x)
    if state==2:store.add('valid2',n,i,valid)


@torch.inference_mode()
def main(tmp):
    setup()
    net,ck=load(SOURCE)
    cfg=ck['hyper_parameters']['config']
    net.cuda().eval().requires_grad_(False)
    del ck
    meta=metadata()
    train=torch.load(BASE/'train_cache.pt',map_location='cpu')
    assert [str(v).zfill(7) for v in train['ids']]==meta['train']['ids']
    assert np.array_equal(train['rows'].numpy(),meta['train']['rows'])
    assert np.array_equal(train['labels'].numpy(),meta['train']['labels'])
    posters=np.load(BASE/'posters.npy',mmap_mode='r')
    store=Store(tmp,'train')
    with torch.autocast('cuda',dtype=torch.float16):
        for state in (1,2):
            pi,pt=net.model.prompt_learner([state])
            n=len(posters) if state==1 else len(train['tokens'])
            for i in range(0,n,64):
                end=min(i+64,n)
                modes=[state]*(end-i)
                with Capture(net,{state}) as cap:
                    if state==1:
                        img=torch.from_numpy(posters[i:end].copy()).cuda()
                        h=net.model.image_encoder(img.type(net.model.dtype),[p.expand(end-i,-1,-1) for p in pi],modes)
                    else:
                        tok=train['tokens'][i:end].cuda()
                        h=net.model.text_encoder(tok,[p.expand(end-i,-1,-1) for p in pt],modes)
                    add_state(store,net,cap,h,state,n,i,i==0)
                # Cross-check optimized frozen train features with archived native features.
                if state==1 and i==0:
                    native=net.infer(dict(image=[img],text=(['']*len(img),),missing_type=modes))['cls_feats'][:,:512]
                    torch.testing.assert_close(h,native,atol=.01,rtol=.002)
                if state==2 and i==0:
                    ds=plots(cfg,'train')
                    texts=[ds.get_text(j)['text'] for j in range(end)]
                    img=torch.from_numpy(posters[meta['train']['rows'][:end]]).cuda()
                    native=net.infer(dict(image=[torch.zeros_like(img)],text=(texts,),missing_type=modes))['cls_feats'][:,512:]
                    torch.testing.assert_close(h,native,atol=.01,rtol=.002)
                    del ds
    del store
    store=Store(tmp,'dev')
    dev=plots(cfg,'dev')
    assert [str(v).zfill(7) for v in dev.table['image_id'].to_pylist()]==meta['dev']['ids']
    assert np.array_equal([dev.index_mapper[j][0] for j in range(len(dev))],meta['dev']['rows'])
    n=len(dev)
    i=0
    with torch.autocast('cuda',dtype=torch.float16):
        for img,text in DataLoader(dev,batch_size=64,num_workers=4,pin_memory=True):
            img=img.cuda()
            for state in (0,1,2):
                batch=dict(image=[torch.zeros_like(img) if state==2 else img],
                           text=(['']*len(text) if state==1 else text,),missing_type=[state]*len(img))
                with Capture(net,{state}) as cap:
                    f=net.infer(batch)['cls_feats']
                    z=net.mmimdb_classifier(f).float()
                    store.add(f'z{state}',n,i,z)
                    if state:
                        h=f.chunk(2,dim=1)[state-1]
                        torch.testing.assert_close(classify(net.mmimdb_classifier,h,state),z,atol=0,rtol=0)
                        add_state(store,net,cap,h,state,n,i,i==0)
            i+=len(img)
    assert i==n
    print('Frozen token cache verified',len(posters),len(train['rows']),n,flush=True)

if __name__=='__main__': main(Path(sys.argv[1]))
