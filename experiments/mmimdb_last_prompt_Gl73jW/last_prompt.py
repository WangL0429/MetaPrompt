"""A shared residual on the existing prompt tokens at the input to the final attention block."""
import numpy as np
import torch
from torch.nn import functional as F

from semantic_io import JOB, projectors, summarize_tokens


def projected(delta,projection):
    with torch.autocast("cuda",enabled=False):
        return delta.float() @ projection.detach().float().T


def inject(x,change,active=None):
    n=len(change)
    update=change[:,None,:].to(x.dtype)
    if active is not None:
        update=update*active[None,:,None]
    return torch.cat((x[:n]+update,x[n:]),dim=0)


def install(net,delta):
    """Install equivalent precomputed projected residuals for native raw-input inference."""
    for state,enc,proj in zip((1,2),(net.model.image_encoder,net.model.text_encoder),projectors(net)):
        change=projected(delta.to(proj.device),proj).detach()
        assert change.shape==(enc.prompt_length//2,len(proj))
        def prehook(block,args,change=change,state=state):
            x,prompts,counter,states=args[0]
            active=torch.as_tensor(states,device=x.device)==state
            if active.any():
                return ([inject(x,change,active),prompts,counter,states],)
        enc.transformer.resblocks[-1].register_forward_pre_hook(prehook)


class LocalTokens:
    def __init__(self,x,mask):
        self.x=x
        self.mask=mask

    def local(self,state,idx=slice(None)):
        return self.x[idx],self.mask[idx]


class LastViews:
    def __init__(self,net,split):
        self.net=net
        folder=JOB/"activations"/split
        self.hidden={s:np.load(folder/f"state{s}.npy",mmap_mode="r") for s in (1,2)}
        self.end=np.load(folder/"end.npy",mmap_mode="r")

    def run(self,ids,state,delta):
        net=self.net
        enc=(net.model.image_encoder,net.model.text_encoder)[state-1]
        proj=projectors(net)[state-1]
        changes=projected(delta,proj)
        block=enc.transformer.resblocks[-1]
        lens=enc.prompt_length
        parts=[[],[],[],[]]
        with torch.autocast("cuda",dtype=torch.float16):
            for first in range(0,len(ids),32):
                rows=ids[first:first+32]
                x=torch.from_numpy(self.hidden[state][rows]).cuda().transpose(0,1)
                x=inject(x,changes)
                y=block([x,[],0,[state]*len(rows)])[0].transpose(0,1)
                tokens=y[:,lens:]
                if state==1:
                    global_feature=enc.ln_post(tokens[:,0]) @ proj
                    local=enc.ln_post(tokens[:,1:]).half()
                    mask=torch.ones(local.shape[:2],dtype=torch.bool,device=y.device)
                else:
                    tokens=enc.ln_final(tokens).to(enc.dtype)
                    end=torch.from_numpy(self.end[rows]).cuda()
                    global_feature=tokens[torch.arange(len(rows),device=y.device),end] @ proj
                    local=tokens[:,1:-1].half()
                    mask=torch.arange(1,tokens.size(1)-1,device=y.device)[None,:]<end[:,None]
                xlocal,valid=summarize_tokens(net,LocalTokens(local,mask),state)
                zeros=torch.zeros_like(global_feature)
                pair=(global_feature,zeros) if state==1 else (zeros,global_feature)
                logits=net.mmimdb_classifier(torch.cat(pair,1)).float()
                for dest,t in zip(parts,(global_feature.half(),xlocal,valid,logits)):
                    dest.append(t)
        return tuple(torch.cat(part) for part in parts)


def materialize(d,views,ids,delta,mode=None):
    ids=np.asarray(ids,dtype=np.int64)
    index=torch.as_tensor(ids,device="cuda")
    batch={k:v[index] for k,v in d.items() if k not in ("complete","cal","rows")}
    live=list(batch["base"].unbind(1))
    for state in (1,2):
        use=np.arange(len(ids)) if mode is None else np.flatnonzero(mode==state)
        if not len(use):
            continue
        at=torch.as_tensor(use,device="cuda")
        h,x,valid,z=views.run(ids[use],state,delta)
        for key,t in zip((f"h{state}",f"x{state}",f"mask{state}"),(h,x,valid)):
            batch[key]=batch[key].index_copy(0,at,t)
        live[state]=live[state].index_copy(0,at,z)
    batch["live_base"]=torch.stack(live,1)
    return batch
