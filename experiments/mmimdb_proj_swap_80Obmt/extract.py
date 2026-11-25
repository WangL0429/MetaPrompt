from pathlib import Path
import json
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

JOB = Path(__file__).resolve().parent
EXP = JOB.parent
BASE = EXP / "mmimdb_joint_full_9aNxqw"
MATCH = EXP / "mmimdb_matched_probe_obZdDo"
SHARED = EXP / "mmimdb_joint_head_UR2YBk" / "selected.ckpt"
DPL = json.loads((EXP / "mmimdb_curriculum_20261119/dpl_specialists.json").read_text())["datasets"]["mmimdb"]
CASE = DPL["case_overrides"]["text_90"]
PYTHON = {"shared": "/disk1/longw/conda_envs/mm/bin/python", "dpl": "/home/wanglong/anaconda3/envs/MM_DCP/bin/python"}


def prepare(tmp):
    sys.path.insert(0, str(BASE))
    from cache import Posters
    from model_io import plots
    from torch.utils.data import DataLoader

    with (MATCH / "metadata.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    for arm, checkpoint in (("shared", SHARED), ("dpl", CASE["checkpoint"])):
        weights = torch.load(checkpoint, map_location="cpu")["state_dict"]
        mat = weights["model.image_encoder.proj"].float().numpy()
        assert mat.shape == (768, 512)
        np.save(JOB / (arm + "_matrix.npy"), mat)
        del weights
    for split in ("dev", "test"):
        ds = plots(dict(val_transform_keys=["CLIP_transform"], image_size=224), split)
        old = meta[split]
        assert [str(x).zfill(7) for x in ds.table["image_id"].to_pylist()] == old["ids"]
        rows = np.array([ds.index_mapper[i][0] for i in range(len(ds))])
        np.testing.assert_array_equal(rows, old["rows"])
        np.testing.assert_array_equal(ds.table["label"].to_pylist(), old["labels"])
        first = np.unique(rows, return_index=True)[1].tolist()
        arr = np.lib.format.open_memmap(tmp / (split+".npy"), mode="w+", dtype=np.float16,
                                        shape=(len(first), 3, 224, 224))
        pos = 0
        for batch in DataLoader(Posters(ds, first), batch_size=48, num_workers=4):
            arr[pos:pos+len(batch)] = batch.numpy()
            pos += len(batch)
        assert pos == len(arr)
        arr.flush()
        del arr


def encode(arm, tmp):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if arm == "shared":
        sys.path.insert(0, str(BASE))
        from model_io import load
        model, obj = load(SHARED)
        del obj
        size = 64
        other = "dpl"
    else:
        import pytorch_lightning
        import torchmetrics
        import yaml
        sys.path.insert(0, DPL["root"])
        from clip.modules import CLIPransformerSS
        cfg = yaml.safe_load(Path(CASE["historical_config"]).read_text())["config"]
        model = CLIPransformerSS(dict(cfg, load_path=str(CASE["checkpoint"]), test_only=True))
        size = 48
        other = "shared"
    model.cuda().eval().requires_grad_(False)
    encoder = model.model.image_encoder
    feature = []
    hook = encoder.ln_post.register_forward_hook(lambda module, inputs, output: feature.append(output))
    matrices = {key: torch.from_numpy(np.load(JOB / (key+"_matrix.npy"))).cuda() for key in (arm,other)}
    assert torch.equal(matrices[arm].to(encoder.proj.dtype), encoder.proj)
    arrays, parity = {}, {}
    with torch.no_grad(), torch.cuda.amp.autocast(), np.load(MATCH / (arm+".npz")) as archive:
        for split in ("train", "dev", "test"):
            posters = np.load(BASE / "posters.npy" if split == "train" else tmp / (split+".npy"), mmap_mode="r")
            old = archive[split+"_x"]
            assert len(posters) == len(old)
            raw = np.empty((len(old),768), dtype=np.float16)
            cross = np.empty_like(old)
            native = np.empty_like(old)
            for start in range(0,len(posters),size):
                img = torch.from_numpy(posters[start:start+size].copy()).cuda().float()
                n = len(img)
                z = model.infer(dict(image=[img], text=([""]*n,), missing_type=[1]*n))["cls_feats"]
                assert len(feature) == 1
                h = feature.pop()
                assert h.shape == (n,768) and not z[:,512:].any()
                projected = h @ matrices[arm]
                assert torch.equal(projected, z[:,:512]), (arm, (projected-z[:,:512]).abs().max().item())
                interval = slice(start,start+n)
                raw[interval] = h.half().cpu().numpy()
                native[interval] = projected.half().cpu().numpy()
                cross[interval] = (h @ matrices[other]).half().cpu().numpy()
            diff = native.astype(np.float32) - old.astype(np.float32)
            parity[split] = dict(max_abs=float(abs(diff).max()), rms=float(np.mean(diff**2)**.5))
            assert parity[split]["max_abs"] < .02 and parity[split]["rms"] < .001, (arm,split,parity)
            arrays.update({split+"_raw":raw, split+"_native":native, split+"_swap":cross})
    hook.remove()
    np.savez(JOB / (arm+"_features.npz"), **arrays)
    (JOB / (arm+"_parity.json")).write_text(json.dumps(parity, indent=2))


def launch():
    with tempfile.TemporaryDirectory(prefix="mmimdb_swap_") as scratch:
        tmp = Path(scratch)
        prepare(tmp)
        def worker(arm,gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
            cwd = BASE if arm=="shared" else Path(DPL["root"])
            subprocess.run([PYTHON[arm], __file__, arm, scratch], cwd=cwd, env=env, check=True)
        with ThreadPoolExecutor(2) as pool:
            tasks = [pool.submit(worker,arm,gpu) for arm,gpu in (("shared",4),("dpl",5))]
            for task in tasks:
                task.result()


if __name__ == "__main__":
    torch.set_num_threads(2)
    if len(sys.argv) == 1:
        launch()
    else:
        encode(sys.argv[1], Path(sys.argv[2]))

