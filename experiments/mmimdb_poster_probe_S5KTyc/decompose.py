import torch
from evaluate import JOB, val_grid, counts, f1, write
from fit_probe import predict

def main():
    torch.set_num_threads(2)
    d=torch.load(JOB/'features.pt',map_location='cpu')
    h=torch.load(JOB/'poster_head.pt',map_location='cpu')
    v=d['dev']['val']
    _,inv=torch.unique(v['movie'],return_inverse=True)
    w=torch.cat((h['w'],d['w'][:,512:]),1)
    cases=[('original',d['w'],d['b']),('visual_columns_only',w,d['b']),
           ('raw_bias_only',d['w'],h['b']),('shared_refit_raw',w,h['b']),
           ('calibrated_bias_only',d['w'],h['b']-h['probe_cut']),
           ('shared_refit_calibrated',w,h['b']-h['probe_cut'])]
    baseline=val_grid(v['z'],v['y'])
    rows=[]
    for name,wt,b in cases:
        z=torch.stack([predict(v['x'][i],wt,b) for i in range(3)])
        if name=='visual_columns_only':
            assert torch.equal(z[2],v['z'][2])
        grid=val_grid(z,v['y'])
        poster=f1(counts(z,v['y'],1.,inv).sum(0))
        row=dict(method=name,pure_poster_f1=poster,nine_mean=sum(grid)/9,
                 worst_change_pp=min(a-bb for a,bb in zip(grid,baseline)))
        row.update({f'{k}_{r}':grid[i*3+j] for i,k in enumerate(('both','image','text')) for j,r in enumerate((.5,.7,.9))})
        rows.append(row)
        print(name,round(poster,4),round(sum(grid)/9,4),[round(a-bb,3) for a,bb in zip(grid,baseline)])
    write('component_check.csv',rows)

if __name__=='__main__':
    main()
