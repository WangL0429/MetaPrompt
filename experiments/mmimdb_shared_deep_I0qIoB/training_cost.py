import json
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from train import JOB, INITIAL, EPOCHS, LOCAL_BATCH, TrainData, joint, make_schedule
from common import load as original
from shared_init import load as shared

def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=TrainData()
    size=len(data.rows)
    batches=(size+LOCAL_BATCH*2-1)//(LOCAL_BATCH*2)
    schedule=make_schedule(batches*EPOCHS,0,'curriculum')
    step=min(range(len(schedule)),key=lambda i:abs(schedule[i][0]-.6))
    epoch=step//batches+1
    full=np.random.default_rng(1000+epoch).permutation(size)
    idx=np.array_split(np.array_split(full,batches)[step%batches],2)[0]
    rate,_,seed=schedule[step]
    result=dict(local_batch=len(idx),step_index=step,missing_rate=rate)
    for name,loader in [('original',original),('shared',shared)]:
        net,ck=loader(INITIAL)
        del ck
        net.cuda().eval().requires_grad_(False)
        net.model.prompt_learner.requires_grad_(True)
        names,params=zip(*net.model.prompt_learner.named_parameters())
        with FlopCounterMode(display=False) as flops:
            g,_,_=joint(net,names,params,data,idx,rate,np.random.default_rng([seed,0]))
        assert g.isfinite().all()
        result[name]=flops.get_total_flops()
        del net,g,params
    assert result['shared']>result['original']>1e12
    (JOB/'training_cost.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':
    main()
