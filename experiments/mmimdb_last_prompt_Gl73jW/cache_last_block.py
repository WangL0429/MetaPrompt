"""Cache lossless frozen inputs to each encoder's final block, indexed like the existing features."""
import json
import sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from semantic_io import JOB, SOURCE, load, metadata, plots, setup, json_out


def split_dir(split):
    return JOB/"activations"/split


@torch.inference_mode()
def cache(splits):
    setup()
    net,ck=load(SOURCE)
    cfg=ck["hyper_parameters"]["config"]
    del ck
    net.cuda().eval().requires_grad_(False)
    for split in splits:
        folder=split_dir(split)
        folder.mkdir(exist_ok=True)
        if (folder/"manifest.json").exists():
            continue
        ds=plots(cfg,split)
        info=metadata()[split]
        assert [str(i).zfill(7) for i in ds.table["image_id"].to_pylist()]==info["ids"]
        assert np.array_equal([ds.index_mapper[i][0] for i in range(len(ds))],info["rows"])
        arrays={}
        buffer={}
        def save_block(block,inputs):
            buffer["hidden"]=inputs[0][0].transpose(0,1)
        def save_ids(block,inputs):
            buffer["end"]=inputs[0].argmax(1)
        def write(key,tensor,start):
            array=tensor.cpu().numpy()
            if key not in arrays:
                arrays[key]=np.lib.format.open_memmap(folder/(key+".npy"),mode="w+",dtype=array.dtype,
                                                      shape=(len(ds),*array.shape[1:]))
            arrays[key][start:start+len(array)]=array
        offset=0
        stream=DataLoader(ds,batch_size=64,num_workers=4,pin_memory=True)
        with torch.autocast("cuda",dtype=torch.float16):
            for images,text in stream:
                images=images.cuda(non_blocking=True)
                for state,encoder in ((1,net.model.image_encoder),(2,net.model.text_encoder)):
                    batch=dict(image=[images if state==1 else torch.zeros_like(images)],
                               text=([""]*len(images) if state==1 else text,),missing_type=[state]*len(images))
                    with ExitStack() as hooks:
                        hooks.callback(encoder.transformer.resblocks[-1].register_forward_pre_hook(save_block).remove)
                        if state==2:
                            hooks.callback(encoder.token_embedding.register_forward_pre_hook(save_ids).remove)
                        net.infer(batch)
                    write(f"state{state}",buffer.pop("hidden"),offset)
                    if state==2:
                        write("end",buffer.pop("end"),offset)
                offset+=len(images)
        assert offset==len(ds) and not buffer
        manifest={k:dict(shape=list(a.shape),dtype=str(a.dtype)) for k,a in arrays.items()}
        for array in arrays.values():array.flush()
        json_out(folder/"manifest.json",manifest)
        print(split,manifest,flush=True)


if __name__=="__main__":
    cache(sys.argv[1:])
