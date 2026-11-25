import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from model_io import (JOB, INITIAL, MATRIX, load, original_load, export_weights,
                      validate, outputs, plots, folded_queries)

def case(mode):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    net,ck=load(INITIAL)
    cfg=ck['hyper_parameters']['config']
    net.cuda().eval().requires_grad_(False)
    learn=net.model.prompt_learner
    learn.requires_grad_(True)
    a=getattr(learn,MATRIX)
    n=sum(p.numel() for p in learn.parameters())
    assert n==103044 and a.numel()==324
    ds=plots(cfg,'train')
    cache=torch.load(JOB/'train_cache.pt',map_location='cpu')
    ix=np.unique(cache['rows'].numpy(),return_index=True)[1][:4]
    rows=cache['rows'][ix]
    tok=cache['tokens'][ix].cuda()
    images=torch.from_numpy(np.load(JOB/'posters.npy',mmap_mode='r')[rows.numpy()]).cuda()
    text=[ds.get_text(int(i))['text'] for i in ix]
    target=cache['labels'][rows].cuda()
    del ds
    state=torch.load(JOB/'resume.pt',map_location='cpu') if mode=='trained' else None
    exported=[]
    metrics={'trainable_params':n,'baseline_trainable_params':n-a.numel()}
    for label in (('last','selected') if state else ('initial','trial')):
        if state:
            vals=state['current' if label=='last' else 'best']
            path=JOB/f'{label}.ckpt'
        else:
            if label=='trial':
                with torch.no_grad():
                    a.normal_(std=.02)
            vals={k:p.detach().cpu().clone() for k,p in learn.named_parameters()}
            path=INITIAL
        with torch.no_grad():
            for k,p in learn.named_parameters():
                p.copy_(vals[k])
        deploy,baseck=original_load(path)
        if label=='trial':
            export_weights(net,ck,vals)
            deploy.load_state_dict(ck['state_dict'],strict=True)
        assert set(baseck['state_dict'])==set(deploy.state_dict())
        del baseck
        deploy.cuda().eval().requires_grad_(False)
        metrics[label]={'matrix_norm':a.norm().item(),'max_error':{},'flops':{}}
        if state and (label=='last' or state['best_epoch']>0):
            assert a.norm()>0
        for s in range(3):
            batch=dict(image=[torch.zeros_like(images) if s==2 else images],
                       text=(['']*len(text) if s==1 else text,),missing_type=[s]*len(text))
            for amp in (False,True):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    direct=outputs(net,images,tok,s)[0]
                    native=deploy.mmimdb_classifier(deploy.infer(batch)['cls_feats']).float()
                torch.testing.assert_close(direct,native,atol=.012 if amp else 3e-4,rtol=.002 if amp else .0001)
                metrics[label]['max_error'][f'{s}_amp{amp}']=(direct-native).abs().max().item()
            if not state:
                # Fast candidate parameter must override the live matrix and preserve meta derivatives.
                fast=a.detach().clone().requires_grad_()
                with torch.backends.cuda.sdp_kernel(enable_math=True,enable_flash=False,enable_mem_efficient=False):
                    z=outputs(net,images,tok,s,params={MATRIX:fast})[0]
                    g,=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,target),fast,create_graph=True)
                    h,=torch.autograd.grad(g.square().sum(),fast)
                assert g.norm()>0 and h.isfinite().all() and h.norm()>0
                one=dict(image=[batch['image'][0][:1]],text=(batch['text'][0][:1],),missing_type=[s])
                with torch.no_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True),FlopCounterMode(display=False) as flops:
                    deploy.mmimdb_classifier(deploy.infer(one)['cls_feats'])
                metrics[label]['flops'][str(s)]=flops.get_total_flops()
        del deploy
    if not state:
        assert metrics['initial']['flops']==metrics['trial']['flops']
    (JOB/f'{mode}_export.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('startup','trained'))
    case(parser.parse_args().mode)
