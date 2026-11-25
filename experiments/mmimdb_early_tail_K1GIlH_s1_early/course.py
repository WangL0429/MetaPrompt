import hashlib
import json
import os
import random
from collections import Counter


def early_tail(schedule,batches,seed):
    """Move 6 moderate-tail and 2 extreme-tail draws into each of epochs 3..9."""
    if batches!=64 or len(schedule)!=20*batches:
        raise ValueError('This protocol uses 20 epochs of 64 paired steps')
    values=[s[0] for s in schedule]
    if any(x>=.8 for x in values[2*batches:9*batches]):
        raise ValueError('Early blocks already include tail draws in this seed')
    donor_steps=range(9*batches,16*batches)
    moderate=[i for i in donor_steps if .8<=values[i]<.9]
    extreme=[i for i in donor_steps if values[i]>=.9]
    rng=random.Random(4000+seed)
    donors=(iter(rng.sample(moderate,42)),iter(rng.sample(extreme,14)))
    for epoch in range(2,9):
        for slot in range(8):
            i=epoch*batches+slot*8+3
            j=next(donors[slot in (1,5)])
            values[i],values[j]=values[j],values[i]
    return [(r,m,s) for r,(_,m,s) in zip(values,schedule)]


def digest(schedule):
    return hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest()


def protocol(base,batches,seed,arm):
    if arm not in ('original','early'):
        raise ValueError(arm)
    schedule=early_tail(base,batches,seed) if arm=='early' else base
    assert Counter(x[0] for x in base)==Counter(x[0] for x in schedule)
    assert [x[1:] for x in base]==[x[1:] for x in schedule]
    stats=[]
    for e in range(20):
        r=[s[0] for s in schedule[e*batches:(e+1)*batches]]
        stats.append(dict(epoch=e+1,n80=sum(x>=.8 for x in r),n90=sum(x>=.9 for x in r)))
    return schedule,dict(arm=arm,seed=seed,base_digest=digest(base),used_digest=digest(schedule),
                         swaps=sum(a!=b for a,b in zip(schedule,base))//2,per_epoch=stats)
