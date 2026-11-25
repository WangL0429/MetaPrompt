"""Fit both encoders' probes concurrently and compute paired intervals from the frozen caches."""
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

JOB = Path(__file__).resolve().parent


def fit(arm, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    subprocess.run([sys.executable, str(JOB/"fit_heads.py"), arm], cwd=JOB, env=env, check=True)


if __name__=="__main__":
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(fit,arm,gpu) for arm,gpu in (("shared",4),("dpl",5))]
        for future in futures:
            future.result()
    subprocess.run([sys.executable, str(JOB/"paired.py")], cwd=JOB, check=True)

