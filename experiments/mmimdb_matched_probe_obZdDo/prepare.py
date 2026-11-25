from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from hashlib import sha256

JOB = Path(__file__).resolve().parent
EXP = JOB.parent
BASE = EXP/'mmimdb_joint_full_9aNxqw'
sys.path.insert(0,str(BASE))
from cache import Posters
from model_io import plots
from evaluate_mmimdb import load_test_masks

def main(tmp):
    torch.set_num_threads(2)
    train = torch.load(BASE/'train_cache.pt', map_location='cpu')
    cfg = dict(val_transform_keys=['CLIP_transform'], image_size=224)
    meta = {}
    for split,n in [('train',15552),('dev',2608),('test',7799)]:
        ds = plots(cfg,split)
        ids = [str(x).zfill(7) for x in ds.table['image_id'].to_pylist()]
        assert len(set(ids)) == len(ids) == n
        rows = np.asarray([ds.index_mapper[i][0] for i in range(len(ds))])
        y = np.asarray(ds.table['label'].to_pylist(),dtype=np.uint8)
        assert y.shape == (n,23)
        assert np.array_equal(np.unique(rows),np.arange(n))
        first = np.unique(rows,return_index=True)[1].tolist()
        images = Posters(ds,first)
        if split == 'train':
            assert ids == [str(x).zfill(7) for x in train['ids']]
            assert np.array_equal(rows,train['rows'].numpy())
            assert np.array_equal(y,train['labels'].numpy())
            old = np.load(BASE/'posters.npy',mmap_mode='r')
            for i in (0,579,n-1):
                assert np.array_equal(images[i].numpy(),old[i])
        else:
            arr = np.lib.format.open_memmap(tmp/(split+'.npy'),mode='w+',dtype=np.float16,shape=(n,3,224,224))
            i=0
            for batch in DataLoader(images,batch_size=48,num_workers=4):
                arr[i:i+len(batch)] = batch.numpy()
                i+=len(batch)
            assert i==n
            arr.flush()
            del arr
        info=dict(ids=ids,rows=rows,labels=y)
        if split=='dev':
            cal=np.asarray([sha256(x.encode()).digest()[0]<144 for x in ids])
            assert cal.sum()==1444
            info['cal']=cal
        if split=='test':
            masks=load_test_masks(ids,['text'],[90])
            mask=np.stack([masks['text',90,seed].numpy() for seed in (42,2026,3407)])
            assert set(np.unique(mask))=={0,1}
            info['masks']=mask
            info['complete_indices']=np.flatnonzero((mask==0).any(axis=0)[rows])
            info['complete_texts']=[ds.get_text(int(i))['text'] for i in info['complete_indices']]
        meta[split]=info
    assert all(set(meta[a]['ids']).isdisjoint(meta[b]['ids']) for a,b in [('train','dev'),('train','test'),('dev','test')])
    import pickle
    with (JOB/'metadata.pkl').open('wb') as f: pickle.dump(meta,f,protocol=4)
    print('READY',[(k,len(v['ids']),len(v['rows'])) for k,v in meta.items()],flush=True)

if __name__=='__main__': main(Path(sys.argv[1]))

