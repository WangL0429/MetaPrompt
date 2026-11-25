"""Shared residual token pooling and the native mixed-state inference path."""
from contextlib import ExitStack

import torch
from torch import nn
from torch.nn import functional as F


class LocalPool(nn.Module):
    def __init__(self, kind, width=512):
        super().__init__()
        if kind not in ('uniform','weighted'):
            raise ValueError(kind)
        self.kind = kind
        self.gate = nn.Parameter(torch.zeros(()))
        self.query = nn.Parameter(torch.zeros(width)) if kind == 'weighted' else None

    def forward(self, h, tokens, valid, projection):
        # Compute small readout in FP32 independently of the encoder's autocast mode.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, tokens, projection = h.float(), tokens.float(), projection.float()
            if self.query is None:
                weights = valid.float() / valid.sum(1,keepdim=True).clamp_min(1)
            else:
                query = (F.normalize(h,dim=-1) + self.query) @ projection.T
                key = F.normalize(query,dim=-1)
                scores = torch.bmm(tokens,key.unsqueeze(-1)).squeeze(-1) / tokens.norm(dim=-1).clamp_min(1e-8)
                weights = (10*scores).masked_fill(~valid,-1e4).softmax(1) * valid
            local = torch.bmm(weights.unsqueeze(1),tokens).squeeze(1) @ projection
            local = F.normalize(local,dim=-1) * h.norm(dim=-1,keepdim=True)
            delta = torch.where(valid.any(1,keepdim=True),local-h,0.)
            return h + self.gate.tanh()*delta


class Capture(ExitStack):
    """Capture existing frozen activations; remove every hook when leaving the context."""
    def __init__(self, backbone, states):
        super().__init__()
        self.backbone = backbone
        self.states = states

    def __enter__(self):
        super().__enter__()
        model = self.backbone.model
        if 1 in self.states:
            def patches(module, inputs, output):
                self.vision = output[0].transpose(0,1)
            self.callback(model.image_encoder.transformer.register_forward_hook(patches).remove)
        if 2 in self.states:
            def ids(module, inputs):
                self.ids = inputs[0]
            def words(module, inputs, output):
                self.text = output
            self.callback(model.text_encoder.token_embedding.register_forward_pre_hook(ids).remove)
            self.callback(model.text_encoder.ln_final.register_forward_hook(words).remove)
        return self

    def local(self, state, idx=slice(None)):
        if state == 1:
            enc = self.backbone.model.image_encoder
            patches = self.vision[idx, enc.prompt_length+1:]
            assert patches.shape[1] == enc.positional_embedding.shape[0]-1
            x = enc.ln_post(patches).half()
            valid = torch.ones(x.shape[:2],device=x.device,dtype=torch.bool)
            return x,valid
        enc = self.backbone.model.text_encoder
        ids = self.ids[idx]
        assert self.text.shape[1]==ids.shape[1]
        pos = torch.arange(1,ids.shape[1]-1,device=ids.device)
        return self.text[idx,1:-1].half(), pos[None,:]<ids.argmax(-1)[:,None]


def projectors(net):
    return (net.model.image_encoder.proj, net.model.text_encoder.text_projection)


def classify(head, h, state):
    zeros = torch.zeros_like(h)
    x = torch.cat((h,zeros) if state==1 else (zeros,h),dim=1)
    with torch.autocast(device_type=h.device.type,enabled=h.is_cuda,dtype=torch.float16):
        return head(x).float()


def pooled_logits(net, pool, batch, head_name='mmimdb_classifier'):
    """Use with the native inference precision. Supports mixed and complete-only batches."""
    states = torch.as_tensor(batch['missing_type'],device=batch['image'][0].device)
    active = set(states.tolist())
    head = getattr(net,head_name)
    with Capture(net,active) as capture:
        full = net.infer(batch)['cls_feats']
        result = head(full).float()
        for state in sorted(active & {1,2}):
            idx = torch.where(states==state)[0]
            x,valid = capture.local(state,idx)
            h = full[idx].chunk(2,dim=1)[state-1]
            h = pool(h,x,valid,projectors(net)[state-1])
            result[idx] = classify(head,h,state)
    return result


def load_pooled_checkpoint(path, device='cuda'):
    from clip.modules import CLIPransformerSS
    ck = torch.load(path,map_location='cpu')
    net = CLIPransformerSS(dict(ck['hyper_parameters']['config'],test_only=True,load_path=''))
    pool = LocalPool(ck['local_token_pool']['kind'])
    net.add_module('local_token_pool',pool)
    net.load_state_dict(ck['state_dict'],strict=True)
    return net.to(device).eval().requires_grad_(False), pool
