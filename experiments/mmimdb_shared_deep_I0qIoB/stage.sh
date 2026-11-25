set -eu
root=/disk1/longw/work/MetaPrompt_code_1116
cd "$root/experiments/mmimdb_shared_deep_I0qIoB"
prev=../mmimdb_joint_full_9aNxqw
cp "$prev/train.py" "$prev/rule.py" "$prev/report.py" "$prev/run.sh" .
sed 's/from common import INITIAL, load, plots/from common import INITIAL, plots\nfrom shared_init import load/' "$prev/model_io.py" > model_io.py
ln -s "$prev/train_cache.pt" train_cache.pt
ln -s "$prev/posters.npy" posters.npy
cmp train.py "$prev/train.py"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 /disk1/longw/conda_envs/mm/bin/python -u measure.py
