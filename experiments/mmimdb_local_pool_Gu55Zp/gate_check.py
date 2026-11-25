"""Exploratory sign reversal on held-out validation movies; no retuning or test inference."""
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from common_pool import JOB,metadata,setup,write
from fit_pool import Features,array,summarize
from pool import load_pooled_checkpoint


@torch.inference_mode()
def main():
    setup()
    tmp=Path((JOB/'cache.path').read_text().strip())
    dev=Features(tmp,'dev')
    meta=metadata()['dev']
    valid=~meta['cal'][meta['rows']]
    y=torch.from_numpy(meta['labels'][meta['rows']][valid]).long()
    indices=torch.as_tensor(np.flatnonzero(valid),device='cuda')
    complete=array(tmp,'dev','z0')[indices].cpu()
    records=[]
    for kind in ('uniform','weighted'):
        net,pool=load_pooled_checkpoint(JOB/f'{kind}.ckpt')
        saved=pool.gate.clone()
        for label,multiplier in [('learned',1),('zero',0),('opposite',-1)]:
            pool.gate.copy_(saved*multiplier)
            z=torch.stack([complete,*[torch.cat([dev.predict(net,pool,state,sub).cpu()
                       for sub in indices.split(64)]) for state in (1,2)]])
            grid=summarize(z,y)
            metrics={}
            for s,name in ((1,'poster'),(2,'text')):
                pred=z[s]>0
                metrics[name+'_F1']=float((200*(pred*y).sum(0)/(pred.sum(0)+y.sum(0)).clamp_min(1)).mean())
                metrics[name+'_AP']=float(100*average_precision_score(y.numpy(),z[s].numpy(),average='macro'))
            records.append(dict(method=kind,setting=label,gate=float(pool.gate.tanh()),
                                text90=grid['text_90'],image90=grid['image_90'],both90=grid['both_90'],**metrics))
    write('validation_gate_sign.csv',records)
    print(records,flush=True)

if __name__=='__main__':main()
