import torch
from common import CLIPransformerSS

def load(path):
    torch.manual_seed(0)
    checkpoint=torch.load(path,map_location='cpu')
    config=checkpoint['hyper_parameters']['config']
    config['shared_deep_prompt']=True
    net=CLIPransformerSS(dict(config,test_only=True,load_path='')).eval()
    net.load_state_dict(checkpoint['state_dict'],strict=True)
    checkpoint['state_dict']=net.state_dict()
    return net,checkpoint
