import sys, json, pickle
from pathlib import Path
import numpy as np
import torch

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/'mmimdb_joint_full_9aNxqw'

def run(arm,tmp):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if arm=='shared':
        sys.path.insert(0,str(BASE))
        from model_io import load
        ck=EXP/'mmimdb_joint_head_UR2YBk'/'selected.ckpt'
        model,obj=load(ck)
        cfg=obj['hyper_parameters']['config']
        del obj
        size=64
    else:
        import pytorch_lightning
        import torchmetrics
        import yaml
        manifest=json.loads((EXP/'mmimdb_curriculum_20261119'/'dpl_specialists.json').read_text())['datasets']['mmimdb']
        sys.path.insert(0,manifest['root'])
        from clip.modules import CLIPransformerSS
        case=manifest['case_overrides']['text_90']
        ck=Path(case['checkpoint'])
        cfg=yaml.safe_load(Path(case['historical_config']).read_text())['config']
        assert cfg['missing_ratio']['train']==.9 and cfg['missing_type']['train']=='text'
        model=CLIPransformerSS(dict(cfg,load_path=str(ck),test_only=True))
        size=48
    assert cfg['image_size']==224 and cfg['val_transform_keys']==['CLIP_transform']
    model.cuda().eval().requires_grad_(False)
    with (JOB/'metadata.pkl').open('rb') as f: meta=pickle.load(f)

    def infer(images,texts,state):
        img=torch.from_numpy(images).cuda().float()
        types=[state]*len(img)
        z=model.infer(dict(image=[img],text=(texts,),missing_type=types))['cls_feats']
        if state==1:
            assert not z[:,512:].any()
        assert z.shape==(len(img),1024)
        if arm=='dpl':
            out=model.mmimdb_pl_head(model.mmimdb_proj(z),torch.tensor(types,device='cuda'))
            pred=out.sigmoid()>=.5
        else:
            out=model.mmimdb_classifier(z)
            pred=out.float().sigmoid()>.5
        assert torch.isfinite(out).all() and torch.isfinite(z).all()
        return z[:,:512].half().cpu().numpy(),out.float().cpu().numpy(),pred.cpu().numpy()

    arrays={}
    with torch.no_grad(), torch.cuda.amp.autocast():
        for split,info in meta.items():
            pics=np.load(BASE/'posters.npy' if split=='train' else tmp/(split+'.npy'),mmap_mode='r')
            n=len(info['ids'])
            assert len(pics)==n
            fields=[[],[],[]]
            for i in range(0,n,size):
                images=pics[i:i+size].copy()
                vals=infer(images,['']*len(images),1)
                for field,value in zip(fields,vals): field.append(value)
            for k,field in zip(('x','z','pred'),fields):
                arrays[split+'_'+k]=np.concatenate(field)
            if split=='test':
                ix=info['complete_indices']
                zs=[]; ps=[]
                for i in range(0,len(ix),size):
                    rows=info['rows'][ix[i:i+size]]
                    _,z,p=infer(pics[rows],info['complete_texts'][i:i+size],0)
                    zs.append(z); ps.append(p)
                arrays['complete_z']=np.concatenate(zs)
                arrays['complete_pred']=np.concatenate(ps)
    np.savez(JOB/(arm+'.npz'),**arrays)
    print(arm,ck,'finished',flush=True)

if __name__=='__main__': run(sys.argv[1],Path(sys.argv[2]))

