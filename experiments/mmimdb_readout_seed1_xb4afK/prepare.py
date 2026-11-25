"""Recreate the archived experiment hierarchy with independent seed-1 weights."""
import shutil
import sys
from pathlib import Path

JOB = Path(sys.argv[1]).resolve()
ARCHIVE = JOB.parent
ROOT = ARCHIVE.parent
EXP = JOB / "pipeline/experiments"
BASE = "mmimdb_joint_full_9aNxqw"
SEM = "mmimdb_pair_ablation_dZfki9"
HEAD = "mmimdb_head_QeuxLr"
POSTER = "mmimdb_poster_probe_S5KTyc"
JOINT = "mmimdb_joint_head_UR2YBk"
VIS = "mmimdb_visual_prompt_NOSgpO"
PREV = "mmimdb_curriculum_20261119"


def link(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    path.symlink_to(target)


def copy(name, files, source=None):
    dest = EXP / name
    dest.mkdir(exist_ok=True)
    for file in files.split():
        shutil.copy2(ARCHIVE / (source or name) / file, dest / file)


def replace(name, old, new):
    path = EXP / name
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError((path, old))
    path.write_text(source.replace(old, new))


for name in ["clip", "datasets"]:
    link(EXP.parent / name, ROOT / name)
copy(PREV, "evaluate_mmimdb.py")
copy(HEAD, "features.py fit.py")
copy(VIS, "common.py")
copy("mmimdb_cross_meta_k9dMAw", "train_meta.py meta_weights.py")
copy(BASE, "train.py model_io.py course.py rule.py", "mmimdb_early_tail_K1GIlH_s1_original")
copy(BASE, "cache.py")
copy(POSTER, "extract.py fit_probe.py")
copy(JOINT, "extract_complete.py train_head.py")
copy(SEM, "fit_semantics.py semantic_io.py predictor.py")
copy(SEM, "encode.py", "mmimdb_semantics_X7IWAN")
for name in ["mmimdb_matched_probe_obZdDo", "mmimdb_local_pool_Gu55Zp"]:
    link(EXP / name, ARCHIVE / name)
link(EXP / BASE / "posters.npy", ARCHIVE / BASE / "posters.npy")

replace(f"{HEAD}/features.py", "curriculum_seed0", "curriculum_seed1")
replace(f"{HEAD}/features.py", "    model = CLIPransformerSS", "    torch.manual_seed(1)\n    model = CLIPransformerSS")
replace(f"{HEAD}/fit.py", "torch.manual_seed(0)", "torch.manual_seed(1)")
replace(f"{VIS}/common.py", "torch.manual_seed(0)", "torch.manual_seed(1)")
for filename in [f"{POSTER}/extract.py", f"{JOINT}/extract_complete.py"]:
    replace(filename, "assert ck['cross_second_meta']['epoch']==6",
            "import json\n    chosen = json.loads((REF/'selection.json').read_text())['best_epoch']\n"
            "    assert ck.get('cross_second_meta', {}).get('epoch', 0) == chosen")
replace(f"{JOINT}/train_head.py", "archived=json.loads((REF/'selection.json').read_text())['history'][6]['grid']",
        "record=json.loads((REF/'selection.json').read_text())\n    archived=record['history'][record['best_epoch']]['grid']")
# The shared poster input is a read-only archive. Recompute every teacher prediction.
cache = EXP / BASE / "cache.py"
source = cache.read_text()
a, b = source.index("    tmp=JOB/"), source.index("    teacher=")
cache.write_text(source[:a] + """    data=np.load(path,mmap_mode='r')
    old=torch.load(path.resolve().parent/'train_cache.pt',map_location='cpu')
    assert ids==old['ids'] and torch.equal(rows,old['rows'])
    assert torch.equal(tokens,old['tokens']) and torch.equal(y,old['labels'])
    assert data.shape==(len(ids),3,224,224)
    assert np.array_equal(data[0],posters[0].numpy())
    del old
""" + source[b:])
replace(f"{SEM}/semantic_io.py", "torch.manual_seed(0)", "torch.manual_seed(1)")
replace(f"{SEM}/fit_semantics.py", 'STEPS*EPOCHS,0,"curriculum"', 'STEPS*EPOCHS,1,"curriculum"')
replace(f"{SEM}/fit_semantics.py", "1000+epoch", "11000+epoch")
replace(f"{SEM}/fit_semantics.py", "[seed,0]", "[seed,1]")
replace(f"{SEM}/fit_semantics.py", "steps=STEPS,seed=0", "steps=STEPS,seed=1")
