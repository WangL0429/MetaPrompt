"""Store paired native MMIMDb predictions, the nine benchmark scores and pure-state scores."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_f1_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP/'mmimdb_joint_full_9aNxqw'))
from model_io import load, plots
from evaluate_mmimdb import MODES, RATES, SEEDS, load_test_masks


def evaluate(stem):
    path = stem.with_suffix('.ckpt').resolve(strict=True)
    output = stem.with_suffix('.npz')
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            assert cached['checkpoint'].item() == str(path)
        return
    torch.set_num_threads(2)
    model, ck = load(path)
    ds = plots(ck['hyper_parameters']['config'], 'test')
    del ck
    model.eval().requires_grad_(False).cuda()
    movies = [str(x).zfill(7) for x in ds.table['image_id'].to_pylist()]
    rows = np.asarray([ds.index_mapper[i][0] for i in range(len(ds))], dtype=np.int32)
    labels = torch.tensor(ds.table['label'].to_pylist(), dtype=torch.long)[rows]
    assert len(set(movies)) == len(movies) == 7799 and labels.shape == (16120,23)
    ids = [movies[row] for row in rows]
    masks = load_test_masks(ids, MODES, RATES)
    loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    chunks = [[], [], []]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for image, text in loader:
            image = image.cuda(non_blocking=True)
            for state in range(3):
                batch = dict(image=[torch.zeros_like(image) if state==2 else image],
                             text=(['']*len(text) if state==1 else text,),missing_type=[state]*len(text))
                z = model.mmimdb_classifier(model.infer(batch)['cls_feats']).float()
                assert z.isfinite().all()
                chunks[state].append(z.cpu())
    probabilities = torch.stack([torch.cat(group) for group in chunks]).sigmoid()
    index = torch.arange(len(rows))
    def score(prob):
        return 100*multilabel_f1_score(prob,labels,num_labels=23,average='macro').item()
    native = np.asarray([[[score(probabilities[masks[m,r,s],index]) for s in SEEDS]
                          for r in RATES] for m in MODES])
    pure = np.asarray([score(p) for p in probabilities])
    bits = np.packbits((probabilities > .5).numpy(), axis=-1)
    tmp = output.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, checkpoint=str(path), predictions=bits, y=labels.numpy().astype(np.uint8),
                            ids=np.asarray(movies), rows=rows, native=native, pure=pure)
    tmp.replace(output)
    print(path, native.shape, flush=True)

if __name__=='__main__':
    evaluate(Path(sys.argv[1]))
