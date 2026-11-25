"""One shared conditional mixture over opposite-modality embeddings; both directions use the native head."""
import torch
from torch import nn
from torch.nn import functional as F

class SemanticPredictor(nn.Module):
    def __init__(self,k,width=96):
        super().__init__()
        self.k=k
        self.input=nn.Linear(512,width)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.query=nn.Linear(width,width)
        self.context=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.vector=nn.Linear(width,512)
        self.magnitude=nn.Linear(width,1)
        self.mix_score=nn.Linear(width,1)
        self.gate=nn.Linear(width,1)
        self.tag=nn.Parameter(torch.randn(2,width)*.05)
        self.slots=nn.Parameter((torch.randn(4,width)*.15)[:k].clone())
        nn.init.zeros_(self.magnitude.weight)
        nn.init.zeros_(self.magnitude.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,-2.)
        if k==1:
            self.mix_score.requires_grad_(False)

    def forward(self,h,x,mask,state,base,missing_weight):
        h=h.float()
        tokens=torch.cat((h[:,None],x.float()),1)
        valid=F.pad(mask,(1,0),value=True)
        memory=F.gelu(self.input(F.layer_norm(tokens,(512,)))+self.tag[state-1])
        global_token=memory[:,0]
        q=self.query(global_token)[:,None]+self.slots
        attn=(q @ self.key(memory).transpose(1,2))*q.shape[-1]**-.5
        weights=attn.masked_fill(~valid[:,None],-1e4).softmax(-1)
        context=q+weights@self.value(memory)
        context=context+self.context(context)
        direction=F.normalize(self.vector(context),dim=-1)
        norm=h.norm(dim=-1).clamp_min(1e-5)[:,None]
        magnitude=norm*(1.5*self.magnitude(context).squeeze(-1).tanh()).exp()
        vectors=direction*magnitude[:,:,None]
        logmix=self.mix_score(context).squeeze(-1).log_softmax(-1)
        gate=self.gate(global_token)
        logweights=torch.cat((F.logsigmoid(-gate),F.logsigmoid(gate)+logmix),dim=1)
        completed=base[:,None]+vectors@missing_weight.T
        scores=torch.cat((base[:,None],completed),dim=1)
        # Marginalize Bernoulli probabilities per label; use stable log odds for BCE.
        yes=torch.logsumexp(logweights[:,:,None]+F.logsigmoid(scores),dim=1)
        no=torch.logsumexp(logweights[:,:,None]+F.logsigmoid(-scores),dim=1)
        return yes-no,(direction,magnitude,logmix,gate)


def paired_nll(aux,target):
    direction,magnitude,logmix,_=aux
    target=target.float()
    cosine=(direction*F.normalize(target,dim=-1)[:,None]).sum(-1)
    component=logmix+32*(cosine-1)
    angular=-torch.logsumexp(component,dim=1)/32
    lognorm=target.norm(dim=-1).clamp_min(1e-5).log()
    radial=F.smooth_l1_loss(magnitude.log(),lognorm[:,None].expand_as(magnitude),reduction="none")
    return (angular+.2*(component.softmax(1).detach()*radial).sum(1)).mean()


def load_predictor_checkpoint(path,device="cuda"):
    """Load the entire frozen backbone, head and generator from one checkpoint."""
    from semantic_io import load
    ck=torch.load(path,map_location="cpu")
    spec=ck["semantic_predictor"]
    # Existing native loader reads the original strict state_dict; extension is kept separately.
    net,_=load(path)
    generator=SemanticPredictor(spec["k"])
    generator.load_state_dict(spec["weights"],strict=True)
    net=net.to(device).eval().requires_grad_(False)
    if "last_prompt" in spec:
        from last_prompt import install
        install(net,spec["last_prompt"])
    return net,generator.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def infer(net,generator,batch):
    from semantic_io import Capture,summarize_tokens
    states=torch.as_tensor(batch["missing_type"],device=batch["image"][0].device)
    active=set(states.tolist())
    with torch.autocast("cuda",dtype=torch.float16),Capture(net,active) as capture:
        f=net.infer(batch)["cls_feats"]
        base=net.mmimdb_classifier(f).float()
        result=base.clone()
        for state in sorted(active & {1,2}):
            ix=torch.where(states==state)[0]
            x,mask=summarize_tokens(net,capture,state,ix)
            h=f[ix].chunk(2,1)[state-1]
            w=net.mmimdb_classifier.weight.chunk(2,1)[2-state].float()
            with torch.autocast("cuda",enabled=False):
                result[ix]=generator(h,x,mask,state,base[ix],w)[0]
    return result
