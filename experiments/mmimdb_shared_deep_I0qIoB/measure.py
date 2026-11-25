import json
import sys
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from model_io import VIS, JOB, INITIAL, load, plots, outputs, clip
from common import load as load_original

def main():
    torch.set_num_threads(2)
    d=torch.load(JOB/'train_cache.pt',map_location='cpu')
    costs={}
    records={}
    import numpy as np
    arr=np.load(JOB/'posters.npy',mmap_mode='r')
    ix=np.unique(d['rows'].numpy(),return_index=True)[1][:3]
    image=torch.from_numpy(arr[d['rows'][ix].numpy()]).cuda()
    tokens=d['tokens'][ix].cuda()
    labels=d['labels'][d['rows'][ix]].cuda()
    for key,loader in [('original',load_original),('shared',load)]:
        net,ck=loader(INITIAL)
        ds=plots(ck['hyper_parameters']['config'],'train')
        text=[ds.get_text(int(i))['text'] for i in ix]
        del ck,ds
        net.cuda().eval().requires_grad_(False)
        learner=net.model.prompt_learner
        learner.requires_grad_(True)
        result={'trainable':sum(p.numel() for p in net.parameters() if p.requires_grad),
                'total_parameters':sum(p.numel() for p in net.parameters()),'matmul_conv_flops':{}}
        records[key]=[]
        for state in range(3):
            batch=dict(image=[torch.zeros_like(image) if state==2 else image],
                       text=(['']*len(text) if state==1 else text,),missing_type=[state]*len(text))
            with torch.no_grad():
                native=net.mmimdb_classifier(net.infer(batch)['cls_feats'])
            z=outputs(net,image,tokens,state)[0]
            torch.testing.assert_close(native,z,atol=3e-4,rtol=1e-4)
            records[key].append(native.cpu())
            if key=='shared':
                g,=torch.autograd.grad(torch.nn.functional.binary_cross_entropy_with_logits(z,labels),
                                       learner.shared_deep_residual)
                assert g.isfinite().all() and g.norm()>0
            single=dict(image=[batch['image'][0][:1]],text=(batch['text'][0][:1],),
                        missing_type=[state])
            for path in ('native','skip'):
                with torch.no_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,
                        enable_mem_efficient=False,enable_math=True),FlopCounterMode(display=False) as count:
                    if path=='native':
                        net.mmimdb_classifier(net.infer(single)['cls_feats'])
                    else:
                        outputs(net,image[:1],tokens[:1],state)
                total=count.get_total_flops()
                assert total>1e9
                result['matmul_conv_flops'][f'{path}_{state}']=total
        costs[key]=result
        del net
    for a,b in zip(records['original'],records['shared']):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    added=costs['shared']['trainable']-costs['original']['trainable']
    assert added==9216
    for path,b in costs['original']['matmul_conv_flops'].items():
        inc=costs['shared']['matmul_conv_flops'][path]-b
        assert inc==2*18*(512*32*2+32*(512+768)),(path,inc)
    (JOB/'inference_cost.json').write_text(json.dumps(costs,indent=2))
    print(json.dumps(costs),flush=True)

if __name__=='__main__':
    main()
