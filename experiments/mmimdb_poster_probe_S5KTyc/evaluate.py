import csv
import json
import numpy as np
import torch
from extract import JOB,REF
from common import val_grid
from fit_probe import ap,predict

def write(name,rows):
    with (JOB/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def counts(z,y,rate,movie_index):
    # Condition on movie-level missingness, aggregate plot-level expected counts.
    pred=(1-rate)*(z[0]>0).float()+rate*(z[1]>0).float()
    plot=torch.stack((pred*y,pred,y),dim=1).double()
    movies=int(movie_index.max())+1
    out=torch.zeros(movies,3,23,dtype=torch.float64)
    out.index_add_(0,movie_index,plot)
    return out.numpy()

def f1(count):
    tp,pred,actual=np.moveaxis(count,-2,0)
    return 100*np.mean(2*tp/np.maximum(pred+actual,1e-12),axis=-1)

def main():
    torch.set_num_threads(2)
    d=torch.load(JOB/'features.pt',map_location='cpu')
    probe=torch.load(JOB/'poster_head.pt',map_location='cpu')
    v=d['dev']['val']
    y=v['y']; movie=v['movie']
    _,inv=torch.unique(movie,return_inverse=True)
    first=np.unique(movie.numpy(),return_index=True)[1]
    assert len(first)==1164 and len(movie)==2441
    for i in range(3):
        check=predict(v['x'][i],d['w'],d['b'])
        torch.testing.assert_close(check,v['z'][i],atol=.01,rtol=.001)
    w=torch.cat((probe['w'],d['w'][:,512:]),1)
    b=probe['b']
    original=v['z']
    refit=predict(v['x'][1],w,b)
    def poster_only(z):
        return torch.stack((original[0],z,original[2]))
    cases=dict(original=original,threshold_only=poster_only(original[1]-probe['original_cut']),
               refit_only=poster_only(refit),refit_calibrated=poster_only(refit-probe['probe_cut']),
               shared_head=torch.stack([predict(v['x'][s],w,b-probe['probe_cut']) for s in range(3)]))
    expected=json_load(REF/'selection.json')['history'][6]['grid']
    grid0=val_grid(original,y)
    assert max(abs(a-b) for a,b in zip(grid0,expected))<.015,(grid0,expected)
    numeric={}; statistics=[]; grids=[]; bygenre=[]
    modes=['both','image','text']
    for key,z in cases.items():
        c0=counts(z,y,1.,inv)
        c90=counts(z,y,.9,inv)
        grid=val_grid(z,y)
        pure=f1(c0.sum(0))
        m90=f1(c90.sum(0))
        assert abs(grid[-1]-m90)<.0001
        numeric[key]=[c0,c90]
        apr=ap(z[1],y)
        eq=counts(z[:,first],y[first],1.,torch.arange(len(first)))
        statistics.append(dict(method=key,pure_poster_macro_F1=pure,text90_macro_F1=m90,
                               pure_poster_macro_AP=apr,movie_equal_poster_macro_F1=f1(eq.sum(0)),
                               movie_equal_poster_macro_AP=ap(z[1,first],y[first])))
        for j,value in enumerate(grid):
            grids.append(dict(method=key,missing_type=modes[j//3],missing_rate=[.5,.7,.9][j%3],
                              macro_F1=value,delta_vs_original_pp=value-grid0[j]))
        tp,n,act=c0.sum(0)
        for i,f in enumerate(200*tp/np.maximum(n+act,1e-12)):
            bygenre.append(dict(method=key,label_index=i,validation_positive_movies=int(y[first,i].sum()),
                                pure_poster_F1=f))
    assert abs(statistics[0]['pure_poster_macro_AP']-statistics[1]['pure_poster_macro_AP'])<.0001
    # Paired cluster bootstrap. Fixed trained heads and thresholds in every replicate.
    rng=np.random.default_rng(73)
    boot=rng.multinomial(len(first),np.full(len(first),1/len(first)),size=2000)
    cis=[]
    for i,metric in enumerate(('pure_poster_macro_F1','text90_macro_F1')):
        curves={k:f1((boot@pair[i].reshape(len(first),-1)).reshape(len(boot),3,23)) for k,pair in numeric.items()}
        for k,ref in [('threshold_only','original'),('refit_only','original'),
                      ('refit_calibrated','original'),('refit_calibrated','threshold_only'),('shared_head','original')]:
            diff=curves[k]-curves[ref]
            lo,hi=np.percentile(diff,[2.5,97.5])
            cis.append(dict(metric=metric,method=k,reference=ref,
                            delta_pp=f1(numeric[k][i].sum(0))-f1(numeric[ref][i].sum(0)),
                            paired_movie_bootstrap_95_lo=lo,paired_movie_bootstrap_95_hi=hi))
    write('validation.csv',statistics)
    write('nine_conditions.csv',grids)
    write('paired_uncertainty.csv',cis)
    write('per_genre.csv',bygenre)
    meta=dict(source_checkpoint=str(REF/'selected.ckpt'),source_epoch=6,train_movies=len(d['train']['ids']),
              train_plot_weight_sum=int(d['train']['weight'].sum()),calibration_movies=1444,validation_movies=len(first),
              validation_plots=len(movie),selected_ridge=probe['lambda_selected'],candidates=probe['candidates'],
              native_baseline_error_pp=max(abs(a-b) for a,b in zip(grid0,expected)))
    (JOB/'summary.json').write_text(json.dumps(meta,indent=2))
    for r in statistics: print(r,flush=True)
    print('CI',[(r['metric'],r['method'],r['reference'],round(r['paired_movie_bootstrap_95_lo'],2),round(r['paired_movie_bootstrap_95_hi'],2)) for r in cis],flush=True)
    print('SHARED',[(v['missing_type'],v['missing_rate'],round(v['delta_vs_original_pp'],3)) for v in grids if v['method']=='shared_head'])
def json_load(path):
    return json.loads(path.read_text())

if __name__=='__main__':
    main()
