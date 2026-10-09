"""Read-only DESI diagnostic: validation tuning, independent galaxy test split."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from speclite import filters
from bedcosmo.num_visits.empirical.desi.training_matrix import _extrapolate_powerlaw
OUT=Path(__file__).parent
# Reuse precisely the earlier experimental regularized estimator, without running its driver.
source=(OUT/'try_regularized_slope.py').read_text()
WIDTHS=[100,200,300,500,800]
TAUS=[0.,.01,.03,.1,.3,float('inf')]
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and node.name in ['quadratic_fit','shrink_slope','select_regularization','fit_selected']:
        exec(compile(ast.get_source_segment(source,node),str(OUT/'try_regularized_slope.py'),'exec'))
METHODS=['constant','powerlaw','regularized']
CUTS=[(0.,1)]+[(t,n) for t in [.05,.1,.2] for n in [1,3,5]]
ZBINS=[0,.6,.9,1.2,2.]
FILTERS={s:filters.load_filter('lsst2023-'+b) for s,b in [('blue','g'),('red','z')]}

def endpoint(w,iv,z,threshold,run):
    """Oriented wavelength increases toward the extrapolated edge."""
    if threshold==0 and run==1:
        return len(w)-1
    distance=(w[-1]-w)*(1+z)
    ref=(distance>=50)&(distance<=300)
    if ref.sum()<3:
        raise ValueError('Insufficient reference bins')
    good=iv>=threshold*np.median(iv[ref])
    for k in range(len(w)-1,run-2,-1):
        if good[k-run+1:k+1].all() and np.allclose(np.diff(w[k-run+1:k+1]),10):
            return k
    raise ValueError('No supported endpoint')

def extrapolate(w,f,iv,target,method,side,horizon):
    if method=='constant':
        level=np.mean(median_filter(f,size=5,mode='reflect')[w>=w[-1]-100])
        return np.full(len(target),level)
    if method=='powerlaw':
        m=w>=w[-1]-500
        sign=-1 if side=='blue' else 1
        return _extrapolate_powerlaw(sign*w[m],f[m],iv[m],sign*w[-1],sign*target)
    return fit_selected(w,f,iv,target,horizon)[0]

# Identity and edge-selection checks before touching the real sample.
sw=np.arange(1000.,3001.,10.)
assert endpoint(sw,np.ones(len(sw)),.5,0,1)==len(sw)-1
assert endpoint(sw,np.ones(len(sw)),.5,.1,5)==len(sw)-1
si=np.ones(len(sw)); si[-2:]=.001
assert endpoint(sw,si,.5,.1,3)==len(sw)-3
assert np.allclose(extrapolate(sw,np.full(len(sw),2.),np.ones(len(sw))*100,np.array([3010.,3200.]),'regularized','red',200),2)
assert np.dot(np.array([1.,2.,3.]),np.zeros(3))==0
print('Five synthetic checks passed',flush=True)
D0=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
D={k:D0[k] for k in ['wave_rest_aa','flux','relative_ivar','redshift','targetid']}; D0.close()
W=D['wave_rest_aa']
exclude=set(pd.read_csv(OUT/'stratified-edge-errors.csv').row.astype(int))
exclude.update(pd.read_csv(OUT/'regularized-slope-errors.csv').row.astype(int))
exclude.update([8235,8357,3302,5285,11428,7666])
excluded_ids=set(D['targetid'][list(exclude)].tolist())

def arrays(row,side):
    m=D['relative_ivar'][row]>0
    w=W[m]; f=D['flux'][row,m]; iv=D['relative_ivar'][row,m]
    if side=='blue':
        return -w[::-1],f[::-1],iv[::-1]
    return w,f,iv

def eligible(row):
    z=D['redshift'][row]
    for side,hobs in [('blue',600),('red',800)]:
        w,f,iv=arrays(row,side); h=hobs/(1+z)
        if w[-1]-w[0]<4*h+150:
            return False
        # No flux/SNR screening in the edge holdout itself.
        sign=-1 if side=='blue' else 1
        obs=sign*w*(1+z); filt=FILTERS[side]
        a=obs*np.interp(obs,filt.wavelength,filt.response)*10*(1+z)
        truth=np.dot(a,f); sigma=np.sqrt(np.sum(a*a/iv))
        coverage=a.sum()/np.trapz(filt.wavelength*filt.response,filt.wavelength)
        if coverage<.995 or truth<=3*sigma:
            return False
    return True

rng=np.random.default_rng(20261010); splits={'validation':[],'test':[]}; used_ids=set(excluded_ids)
for b,(lo,hi) in enumerate(zip(ZBINS[:-1],ZBINS[1:])):
    pool=np.flatnonzero((D['redshift']>=lo)&(D['redshift']<hi))
    accepted=[]
    for row in rng.permutation(pool):
        tid=int(D['targetid'][row])
        if tid in used_ids or not eligible(row):
            continue
        used_ids.add(tid); accepted.append(int(row))
        if len(accepted)==200:
            break
    if len(accepted)!=200:
        raise ValueError(f'Insufficient eligible galaxies in z bin {b}: {len(accepted)}')
    splits['validation']+=accepted[:100]; splits['test']+=accepted[100:]
assert set(D['targetid'][splits['validation']]).isdisjoint(D['targetid'][splits['test']])
assert set(D['targetid'][splits['test']]).isdisjoint(excluded_ids)
protocol=dict(seed=20261010,splits={s:[dict(row=r,targetid=int(D['targetid'][r]),z=float(D['redshift'][r])) for r in rows] for s,rows in splits.items()},excluded_previous_example_rows=sorted(exclude),cuts=CUTS,methods=METHODS,holdout_observed_aa={'blue':600,'red':800},reference_observed_aa=[50,300],band={'blue':'g','red':'z'},minimum_measured_photon_response=.995,band_snr_minimum=3,primary_score='median absolute fractional change in DESI-covered synthetic band flux',secondary_score='RMS of inverse-variance weighted residuals in 50-rest-Angstrom bins; flux normalized by saved per-galaxy scale',regularization='existing estimator: three inner holdouts select width and shrinkage; candidate quality cut applied at outer fit endpoint only',limitations='Artificial cutoffs do not reproduce physical DESI endpoint noise; no ground truth for missing u/y; reference is noisy measured data, not latent continuum; no NMF or visit-prior validation',selection='For each side and method, minimize validation median band-flux error; exact ties prefer lower threshold then fewer bins; final winning method selected separately per side using validation only')
(OUT/'edge-tuning-protocol.json').write_text(json.dumps(protocol,indent=2))
print('Sample split saved: 400 validation and 400 test; previous examples excluded',flush=True)

def evaluate(rows,cuts_by_side):
    records=[]
    for counter,row in enumerate(rows):
        z=float(D['redshift'][row])
        for side,hobs in [('blue',600),('red',800)]:
            w,f,iv=arrays(row,side); h=hobs/(1+z)
            base=np.flatnonzero(w<w[-1]-h)[-1]
            sign=-1 if side=='blue' else 1; obs=sign*w*(1+z); filt=FILTERS[side]
            a=obs*np.interp(obs,filt.wavelength,filt.response)*10*(1+z)
            truth=float(np.dot(a,f)); cache={}
            for threshold,run,method,label in cuts_by_side[side]:
                try:
                    k=endpoint(w[:base+1],iv[:base+1],z,threshold,run)
                    key=(k,method)
                    if key not in cache:
                        target=w[k+1:]
                        pred=extrapolate(w[:k+1],f[:k+1],iv[:k+1],target,method,side,w[-1]-w[k])
                        residual=pred-f[k+1:]
                        band_error=float(abs(np.dot(a[k+1:],residual))/abs(truth))
                        # Include discarded training bins as well as the fixed holdout.
                        means=[]
                        for left in np.arange(w[k+1],w[-1]+1,50):
                            m=(target>=left)&(target<left+50)
                            if m.any():
                                means.append(np.average(residual[m],weights=iv[k+1:][m]))
                        cache[key]=dict(band_error=band_error,continuum_rmse=float(np.sqrt(np.mean(np.square(means)))),negative=bool(np.any(pred<0)),extreme=bool(np.max(np.abs(pred))>10*np.median(np.abs(f[:base+1]))),trim_observed_aa=float((w[base]-w[k])*(1+z)),failure='')
                    stats=cache[key]
                except ValueError as exc:
                    stats=dict(band_error=float('inf'),continuum_rmse=float('inf'),negative=False,extreme=True,trim_observed_aa=float('nan'),failure=str(exc))
                records.append(dict(row=row,targetid=int(D['targetid'][row]),z=z,zbin=int(np.searchsorted(ZBINS,z,side='right')-1),side=side,method=method,threshold=threshold,run=run,label=label,**stats))
        if (counter+1)%25==0:
            print('Completed',counter+1,'/',len(rows),'galaxies',flush=True)
    return pd.DataFrame(records)

validation_cuts={s:[(t,n,m,'grid') for m in METHODS for t,n in CUTS] for s in FILTERS}
V=evaluate(splits['validation'],validation_cuts); V.to_csv(OUT/'edge-tuning-validation.csv',index=False)
agg=V.groupby(['side','method','threshold','run'],as_index=False).agg(median_band_error=('band_error','median'),p90_band_error=('band_error',lambda x:x.quantile(.9)),median_continuum_rmse=('continuum_rmse','median'),negative_fraction=('negative','mean'),failure_count=('failure',lambda x:(x!='').sum()),trim_fraction=('trim_observed_aa',lambda x:(x>0).mean()))
agg.to_csv(OUT/'edge-tuning-validation-summary.csv',index=False)
selected={}; test_cuts={s:[] for s in FILTERS}
for side in FILTERS:
    selected[side]={}
    for method in METHODS:
        candidates=agg[(agg.side==side)&(agg.method==method)&(agg.failure_count==0)]
        best=candidates.sort_values(['median_band_error','threshold','run']).iloc[0]
        selected[side][method]=dict(threshold=float(best.threshold),run=int(best.run),validation_median_band_error=float(best.median_band_error))
        test_cuts[side].append((float(best.threshold),int(best.run),method,'tuned '+method))
    selected[side]['winner']=min(METHODS,key=lambda m:selected[side][m]['validation_median_band_error'])
    test_cuts[side]+=[(0.,1,'constant','no cut constant'),(.1,3,'constant','trial cut constant')]
(OUT/'edge-tuning-selected.json').write_text(json.dumps(selected,indent=2))
print('Validation choices frozen BEFORE test scoring:',json.dumps(selected),flush=True)
T=evaluate(splits['test'],test_cuts); T.to_csv(OUT/'edge-tuning-test.csv',index=False)
summary=T.groupby(['side','label'],as_index=False).agg(n=('row','size'),median_band_error=('band_error','median'),p90_band_error=('band_error',lambda x:x.quantile(.9)),median_continuum_rmse=('continuum_rmse','median'),negative_fraction=('negative','mean'),extreme_fraction=('extreme','mean'),trim_fraction=('trim_observed_aa',lambda x:(x>0).mean()),median_trim_observed_aa=('trim_observed_aa','median'),failure_count=('failure',lambda x:(x!='').sum()))
summary.to_csv(OUT/'edge-tuning-test-summary.csv',index=False)
print(summary.to_string(index=False),flush=True)
