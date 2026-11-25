import json
from pathlib import Path
import sys
JOB=Path(__file__).resolve().parent
CONFIG=json.loads((JOB/'run_config.json').read_text())
PREV=JOB.parent/'mmimdb_joint_head_UR2YBk'
PROBE_TOOLS=JOB.parent/'mmimdb_poster_probe_S5KTyc'
REF=JOB.parent/CONFIG['source']
PROBE=JOB.parent/CONFIG['probe']
CACHE=JOB.parent/CONFIG['complete']
COEF=float(CONFIG['coefficient'])
sys.path.insert(0,str(JOB.parent/'mmimdb_joint_full_9aNxqw'))
from model_io import load
