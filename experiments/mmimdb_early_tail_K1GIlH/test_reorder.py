import unittest
from collections import Counter
from pathlib import Path
import runpy
try:
    from course import early_tail
except ImportError:
    early_tail=None

make_schedule=runpy.run_path(str(Path(__file__).resolve().parents[2]/'clip/datamodules/missing_schedule.py'))['make_schedule']

class ScheduleTest(unittest.TestCase):
    def test_paired_20_epoch_schedule(self):
        self.assertTrue(callable(early_tail),'missing rate-only schedule rearrangement')
        for seed in (0,1):
            base=make_schedule(1280,seed,'curriculum')
            saved=list(base)
            out=early_tail(base,64,seed)
            self.assertEqual(base,saved)
            self.assertEqual(out,early_tail(base,64,seed))
            self.assertEqual(Counter(x[0] for x in out),Counter(x[0] for x in base))
            self.assertEqual([x[1:] for x in out],[x[1:] for x in base])
            self.assertEqual(out[:128],base[:128])
            self.assertEqual(out[1024:],base[1024:])
            self.assertEqual(sum(x!=y for x,y in zip(base,out)),112)
            for e in range(2,9):
                r=[x[0] for x in out[e*64:(e+1)*64]]
                self.assertEqual([sum(v>=.8 for v in r[i:i+8]) for i in range(0,64,8)],[1]*8)
                self.assertEqual(sum(v>=.9 for v in r),2)
            self.assertEqual(sum(v[0]>=.9 for v in out[:384]),8)
