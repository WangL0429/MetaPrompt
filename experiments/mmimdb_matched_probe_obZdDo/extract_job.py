import os, subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

JOB=Path(__file__).resolve().parent
PY='/disk1/longw/conda_envs/mm/bin/python'
DPY='/home/wanglong/anaconda3/envs/MM_DCP/bin/python'
DPL='/disk1/longw/work/mm_sota/DPL/DPL_DCP'

def run(cmd,name,gpu,cwd):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (JOB/(name+'.log')).open('w') as log:
        status=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if status.returncode: raise RuntimeError(name+'\n'+(JOB/(name+'.log')).read_text()[-4200:])
    print(name,'finished',flush=True)

with TemporaryDirectory(prefix='mmimdb-posters-') as tmp:
    run([PY,'-u',str(JOB/'prepare.py'),tmp],'prepare',1,JOB)
    with ThreadPoolExecutor(2) as pool:
        tasks=[pool.submit(run,[py,'-u',str(JOB/'encode.py'),arm,tmp],arm,gpu,cwd)
               for arm,py,gpu,cwd in [('shared',PY,1,JOB.parents[1]),('dpl',DPY,2,DPL)]
               if not (JOB/(arm+'.npz')).exists()]
        for task in tasks: task.result()

