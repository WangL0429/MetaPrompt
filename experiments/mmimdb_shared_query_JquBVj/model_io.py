from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from torch.func import functional_call
from mixing import mix_query

JOB=Path(__file__).resolve().parent
VIS=JOB.parent/'mmimdb_visual_prompt_NOSgpO'
CROSS=JOB.parent/'mmimdb_cross_meta_k9dMAw'
sys.path[:0]=[str(CROSS),str(VIS)]
from common import INITIAL,load as original_load,plots
from train_meta import validate as original_validate, validation_data,views
from meta_weights import choose_weights
from clip.modules import clip

NAMES=('complete','image','text')
MATRIX='shared_query_mix'


def load(path):
    net,ck=original_load(path)
    learner=net.model.prompt_learner
    tokens=learner.common_prompt_complete.shape[0]
    learner.register_parameter(MATRIX,torch.nn.Parameter(learner.common_prompt_complete.new_zeros(tokens,tokens)))
    return net,ck


def query_path(state):
    return 'transformer.resblocks.0.prompts_dynamic_'+NAMES[state]


def projected(net,state,params=None):
    learner=net.model.prompt_learner
    if params is None:
        return learner([state])
    return functional_call(learner,params,([state],))


def outputs(net,images,tokens,state,params=None):
    learner=net.model.prompt_learner
    matrix=getattr(learner,MATRIX,None) if params is None else params.get(MATRIX,getattr(learner,MATRIX,None))
    pi,pt=projected(net,state,params)
    if matrix is not None and torch.is_autocast_enabled():
        with torch.autocast('cuda',enabled=False):
            qi,qt=projected(net,state,params)
    else:
        qi,qt=pi,pt
    n=len(tokens)
    def encode(encoder,input_data,static,for_query):
        args=(input_data,[p.expand(n,-1,-1) for p in static],[state]*n)
        if matrix is None:
            return encoder(*args)
        bank=getattr(encoder.transformer.resblocks[0],'prompts_dynamic_'+NAMES[state])
        query=mix_query(bank,for_query[0][0],matrix)
        return functional_call(encoder,{query_path(state):query},args)

    im=tx=None
    if state!=2:
        im=encode(net.model.image_encoder,images.type(net.model.dtype),pi,qi)
    if state!=1:
        tx=encode(net.model.text_encoder,tokens,pt,qt)
    if im is None:
        im=torch.zeros_like(tx)
    if tx is None:
        tx=torch.zeros_like(im)
    return net.mmimdb_classifier(torch.cat((im,tx),1)).float(),im.float(),tx.float()


@contextmanager
def folded_queries(net):
    learner=net.model.prompt_learner
    matrix=getattr(learner,MATRIX)
    originals=[]
    try:
        with torch.no_grad(),torch.autocast(matrix.device.type,enabled=False):
            for s,name in enumerate(NAMES):
                pi,pt=projected(net,s)
                for enc,p in [(net.model.image_encoder,pi),(net.model.text_encoder,pt)]:
                    bank=getattr(enc.transformer.resblocks[0],'prompts_dynamic_'+name)
                    originals.append((bank,bank.clone()))
                    bank.copy_(mix_query(bank,p[0][0],matrix))
        yield
    finally:
        with torch.no_grad():
            for bank,original in originals:
                bank.copy_(original)


def validate(net,dev,labels):
    with folded_queries(net):
        return original_validate(net,dev,labels)


@torch.no_grad()
def export_weights(net,ck,values):
    learner=net.model.prompt_learner
    for name,p in learner.named_parameters():
        p.copy_(values[name])
    with folded_queries(net):
        for name,tensor in net.state_dict().items():
            if name=='model.prompt_learner.'+MATRIX:
                continue
            if name.startswith('model.prompt_learner.') or '.prompts_dynamic_' in name:
                ck['state_dict'][name]=tensor.detach().cpu().clone()
    assert 'model.prompt_learner.'+MATRIX not in ck['state_dict']
