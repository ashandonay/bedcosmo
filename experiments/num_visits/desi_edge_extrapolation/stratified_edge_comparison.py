"""Stratified exploratory red-edge comparison; saved matrix is read-only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from bedcosmo.num_visits.empirical.desi.training_matrix import _extrapolate_powerlaw
OUT=Path(__file__).parent
WIDTHS=[100,200,300,500,800]
TAUS=[0.,.01,.03,.1,.3,float('inf')]
def quadratic_fit(w,f,iv,width):
    edge=w[-1]
    m=w>edge-width
    x=(w[m]-edge)/width
    if len(x)<5:
        raise ValueError('Too few points for local quadratic')
    kernel=(1-np.abs(x)**3)**3
    X=np.column_stack([np.ones_like(x),x,x*x])
    root=np.sqrt(iv[m]*kernel)
    beta=np.linalg.lstsq(X*root[:,None],f[m]*root,rcond=None)[0]
    # Soft-L1 IRLS: outliers are assessed in measured noise units, before tapering.
    for _ in range(100):
        r=(X@beta-f[m])*np.sqrt(iv[m])
        robust=1/np.sqrt(1+r*r)
        weights=iv[m]*kernel*robust
        root=np.sqrt(weights)
        updated=np.linalg.lstsq(X*root[:,None],f[m]*root,rcond=None)[0]
        if np.linalg.norm(updated-beta)<1e-9*(1+np.linalg.norm(beta)):
            beta=updated
            break
        beta=updated
    # Approximate conditional fit uncertainty; excludes window/model selection uncertainty.
    cov=np.linalg.inv(X.T@(weights[:,None]*X))
    return beta, np.sqrt(cov[1,1])/width

def predict(w,f,iv,target,width,method):
    edge=w[-1]
    if method=='local_quadratic':
        beta,se=quadratic_fit(w,f,iv,width)
        return beta[0]+beta[1]*(target-edge)/width, dict(level=float(beta[0]),slope=float(beta[1]/width),slope_se=float(se),beta=beta.tolist())
    m=w>=edge-width
    if method=='constant':
        filtered=median_filter(f,size=min(5,len(f)//2*2-1),mode='reflect')
        level=np.mean(filtered[m])
        return np.full(len(target),level),dict(level=float(level))
    if method=='powerlaw':
        return _extrapolate_powerlaw(w[m],f[m],iv[m],edge,target),{}
    raise ValueError(method)

def shrink_slope(slope,se,level,tau_fraction):
    if tau_fraction==0:
        return 0.,0.
    if np.isinf(tau_fraction):
        return slope,1.
    tau=abs(level)*tau_fraction/100
    factor=tau*tau/(tau*tau+se*se)
    return slope*factor,float(factor)

def select_regularization(w,f,iv,horizon):
    """Choose width and zero-centered slope prior using only inner holdouts."""
    scores=np.zeros((len(WIDTHS),len(TAUS)))
    for i,width in enumerate(WIDTHS):
        total=np.zeros(len(TAUS)); norm=0.
        for j in [1,2,3]:
            cutoff=w[-1]-j*horizon
            train=w<cutoff
            valid=(w>=cutoff)&(w<cutoff+horizon+1e-7)
            if train.sum()<5 or valid.sum()<3:
                raise ValueError('Insufficient inner validation coverage')
            beta,se=quadratic_fit(w[train],f[train],iv[train],width)
            raw_slope=beta[1]/width
            slopes=np.array([shrink_slope(raw_slope,se,beta[0],tau)[0] for tau in TAUS])
            for lo in np.arange(cutoff,cutoff+horizon,50):
                block=valid&(w>=lo)&(w<lo+50)
                if not block.any():
                    continue
                weight=iv[block].sum()
                distance=np.average(w[block],weights=iv[block])-w[train][-1]
                mean=np.average(f[block],weights=iv[block])
                total+=weight*(beta[0]+slopes*distance-mean)**2
                norm+=weight
        scores[i]=total/norm
    i,j=np.unravel_index(np.argmin(scores),scores.shape)
    return WIDTHS[i],TAUS[j],scores

def fit_selected(w,f,iv,target,horizon):
    width,tau,scores=select_regularization(w,f,iv,horizon)
    beta,se=quadratic_fit(w,f,iv,width)
    raw_slope=float(beta[1]/width)
    slope,factor=shrink_slope(raw_slope,se,beta[0],tau)
    p=beta[0]+slope*(target-w[-1])
    info=dict(width=width,tau_fraction=None if np.isinf(tau) else tau,unregularized=bool(np.isinf(tau)),raw_slope=raw_slope,slope=float(slope),slope_se=float(se),retained_slope_fraction=factor,level=float(beta[0]),beta=beta.tolist(),inner_scores=scores.tolist())
    return p,info

with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    D={key:saved[key] for key in ['wave_rest_aa','redshift','flux','relative_ivar','targetid']}
w=D['wave_rest_aa']; zs=D['redshift']; bins=[0,.6,.9,1.2,2]
rng=np.random.default_rng(20261008)
# Apply the same measured-mean > 3-sigma rule used in the earlier diagnostic.
eligible=[]
insufficient_coverage=0
for row in range(len(zs)):
    iv=D['relative_ivar'][row]; measured=iv>0; end=w[measured][-1]
    held=measured&(w>=end-800/(1+zs[row]))
    mean=np.average(D['flux'][row,held],weights=iv[held]); sigma=np.sqrt(1/iv[held].sum())
    if mean>3*sigma:
        tw=w[measured&~held]
        horizon=800/(1+zs[row])
        enough=True
        for j in [0,1,2,3]:
            cutoff=tw[-1]-j*horizon
            train=tw if j==0 else tw[tw<cutoff]
            valid=tw[(tw>=cutoff)&(tw<cutoff+horizon+1e-7)]
            if len(train)<5 or (j>0 and len(valid)<3):
                enough=False
                break
            if (train>train[-1]-min(WIDTHS)).sum()<5:
                enough=False
                break
        if enough:
            eligible.append(row)
        else:
            insufficient_coverage+=1
eligible=np.array(eligible)
rows=[]; populations=[]
for lo,hi in zip(bins[:-1],bins[1:]):
    candidates=eligible[(zs[eligible]>=lo)&(zs[eligible]<hi)]
    assert len(candidates)>=100
    rows.extend(rng.choice(candidates,100,replace=False).tolist())
    populations.append(dict(z_min=lo,z_max=hi,eligible_population=len(candidates),selected=100))
assert len(rows)==len(set(rows))==400
print('Selected 100 eligible galaxies per bin:',populations,flush=True)
records=[]; selections=[]
for k,row in enumerate(rows):
    f=D['flux'][row]; iv=D['relative_ivar'][row]; z=float(zs[row]); measured=iv>0; end=w[measured][-1]
    held=measured&(w>=end-800/(1+z)); available=measured&~held
    tw=w[available]; tf=f[available]; ti=iv[available]; target=w[held]
    regularized,info=fit_selected(tw,tf,ti,target,800/(1+z))
    selections.append(dict(row=int(row),targetid=int(D['targetid'][row]),**info))
    predictions={'regularized':regularized,'constant':predict(tw,tf,ti,target,100,'constant')[0],'powerlaw':predict(tw,tf,ti,target,500,'powerlaw')[0]}
    truth=np.average(f[held],weights=iv[held]); sigma=np.sqrt(1/iv[held].sum())
    for method,p in predictions.items():
        assert np.all(np.isfinite(p))
        pm=np.average(p,weights=iv[held])
        records.append(dict(row=int(row),targetid=int(D['targetid'][row]),method=method,z=z,true_mean=float(truth),pred_mean=float(pm),mean_sigma=float(sigma),mean_error=float(abs(pm-truth)/abs(truth)),negative_bins=int((p<0).sum()),min_prediction=float(p.min())))
    if (k+1)%25==0:
        print('Completed',k+1,'of 400',flush=True)
T=pd.DataFrame(records)
T.to_csv(OUT/'stratified-edge-errors.csv',index=False)
(OUT/'stratified-edge-selections.json').write_text(json.dumps(selections,indent=2,allow_nan=False))
meta=dict(seed=20261008,n_per_bin=100,redshift_bins=bins,eligibility='Measured mean of red holdout exceeds 3 sigma and sufficient measured bins for all inner validation fits',insufficient_inner_coverage_population=insufficient_coverage,populations=populations,outer_width_observed_aa=800,inner_blocks=3,windows_rest_aa=WIDTHS,prior_fractional_slope_std_per_100_rest_aa=[0,.01,.03,.1,.3,'infinity'],negativity='No positivity constraint or clipping',purpose='Exploratory stratified comparison, not a final independent test')
(OUT/'stratified-edge-provenance.json').write_text(json.dumps(meta,indent=2))
print(T.groupby('method').mean_error.median().to_string(),flush=True)
