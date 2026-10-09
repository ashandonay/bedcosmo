"""Exploratory slope shrinkage; input matrix is read-only. No package changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bedcosmo.num_visits.empirical.desi.training_matrix import _extrapolate_powerlaw

OUT=Path(__file__).parent
WIDTHS=[100,200,300,500,800]
# Prior standard deviation of fractional slope per 100 rest-frame Angstroms.
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

# Synthetic checks: zero/infinite prior limits, uncertain slopes shrink more,
# exact quadratic recovery, and a genuinely flat continuum stays flat.
assert shrink_slope(.01,.001,1.,0)==(0.,0.)
assert shrink_slope(.01,.001,1.,float('inf'))==(.01,1.)
assert shrink_slope(.01,.003,1.,.1)[1]<shrink_slope(.01,.001,1.,.1)[1]
sw=np.arange(1000.,3001.,10.); sx=(sw-sw[-1])/500
beta,se=quadratic_fit(sw,2+.4*sx-.3*sx*sx,np.ones(len(sw))*100,500)
assert np.allclose(beta,[2,.4,-.3],atol=1e-8)
p,info=fit_selected(sw,np.ones(len(sw))*2,np.ones(len(sw))*100,np.array([3010.,3100.]),200)
assert np.allclose(p,2.,atol=1e-8)
print('Five synthetic checks passed',flush=True)

D=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w=D['wave_rest_aa']; rows=np.random.default_rng(42).choice(len(D['redshift']),100,replace=False)
previous=json.loads((OUT/'local-quadratic-selections.json').read_text())
previous={r['row']:r for r in previous if r['method']=='local_quadratic'}
records=[]; selections=[]; curves={}
for k,row in enumerate(rows):
    f=D['flux'][row]; iv=D['relative_ivar'][row]; z=float(D['redshift'][row]); measured=iv>0
    end=w[measured][-1]; held=measured&(w>=end-800/(1+z)); available=measured&~held
    tw=w[available]; tf=f[available]; ti=iv[available]; target=w[held]
    p,info=fit_selected(tw,tf,ti,target,800/(1+z))
    selections.append(dict(row=int(row),targetid=int(D['targetid'][row]),**info))
    curves[(int(row),'regularized')]=p
    baselines={}
    for method,width in [('constant',100),('powerlaw',500),('local_quadratic',previous[int(row)]['width'])]:
        baselines[method]=predict(tw,tf,ti,target,width,method)[0]
    baselines['regularized']=p
    truth=np.average(f[held],weights=iv[held]); sigma=np.sqrt(1/iv[held].sum())
    for method,prediction in baselines.items():
        curves[(int(row),method)]=prediction
        pm=np.average(prediction,weights=iv[held])
        records.append(dict(row=int(row),method=method,z=z,true_mean=truth,pred_mean=pm,mean_sigma=sigma,mean_error=abs(pm-truth)/abs(truth) if truth>3*sigma else np.nan,negative_bins=int((prediction<0).sum()),min_prediction=float(prediction.min())))
    if (k+1)%25==0: print('Completed',k+1,'galaxies',flush=True)
T=pd.DataFrame(records); T.to_csv(OUT/'regularized-slope-errors.csv',index=False)
(OUT/'regularized-slope-selections.json').write_text(json.dumps(selections,indent=2,allow_nan=False))
P=T.pivot(index='row',columns='method',values='mean_error').dropna()
summary={method:dict(n=len(P),median_mean_error=float(P[method].median()),p90_mean_error=float(P[method].quantile(.9)),negative_prediction_galaxies=int((T[T.method==method].set_index('row').loc[P.index].negative_bins>0).sum())) for method in P.columns}
summary['comparison']=dict(regularized_beats_constant_fraction=float((P.regularized<P.constant).mean()),regularized_beats_unregularized_fraction=float((P.regularized<P.local_quadratic).mean()),median_retained_slope_fraction=float(np.median([s['retained_slope_fraction'] for s in selections])),zero_slope_galaxies=sum(s['retained_slope_fraction']==0 for s in selections),unregularized_galaxies=sum(s['unregularized'] for s in selections))
summary['protocol']=dict(seed=42,outer_width_observed_aa=800,inner_blocks=3,window_candidates_rest_aa=WIDTHS,prior_fractional_slope_std_per_100_rest_aa=[0,.01,.03,.1,.3,'infinity'],selection_score='inverse-variance-weighted squared residual of 50-rest-Angstrom mean-flux bins',uncertainty='approximate conditional slope uncertainty; excludes model/window-selection uncertainty',positivity='no flux clipping or positivity constraint')
(OUT/'regularized-slope-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2),flush=True)

fig,axs=plt.subplots(3,1,figsize=(12,11.5))
lookup={s['row']:s for s in selections}
styles={'constant':('#286bb5','--','Constant, 100 Å'),'powerlaw':('#d87915','--','Power law, 500 Å'),'local_quadratic':('#918a9d',':','Unregularized tangent'),'regularized':('#823bb2','--','Regularized tangent')}
for ax,row in zip(axs,[5230,5684,8955]):
    f=D['flux'][row]; iv=D['relative_ivar'][row]; z=float(D['redshift'][row]); end=w[iv>0][-1]
    held=(iv>0)&(w>=end-800/(1+z)); available=(iv>0)&~held; edge=w[available][-1]; info=lookup[row]
    ax.axvspan(edge-info['width'],edge+5,color='#823bb2',alpha=.06)
    ax.axvspan(edge+5,end+5,color='#299568',alpha=.09)
    visible=(iv>0)&(w>=edge-850)
    ax.plot(w[visible],f[visible],color='#a8adb4',lw=.6,alpha=.45)
    for mask,color,label in [(visible&~held,'#414852','Available data, 50 Å bins'),(held,'#16845c','Withheld data, 50 Å bins')]:
        xx=[]; yy=[]; se=[]
        for lo in np.arange(edge-850,end+50,50):
            m=mask&(w>=lo)&(w<lo+50)
            if m.any():
                xx.append(np.average(w[m],weights=iv[m])); yy.append(np.average(f[m],weights=iv[m])); se.append(np.sqrt(1/iv[m].sum()))
        ax.errorbar(xx,yy,yerr=se,color=color,fmt='o-',ms=4,lw=1,capsize=2,label=label,zorder=4)
    texts=[]
    for method,(color,ls,label) in styles.items():
        pred=curves[(row,method)]
        err=T[(T.row==row)&(T.method==method)].mean_error.iloc[0]
        ax.plot(w[held],pred,color=color,ls=ls,lw=2.3,label=label)
        texts.append(f'{label}: {err:.1%}')
    ax.text(.02,.96,'\n'.join(texts),transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    ax.axvline(edge+5,color='#545c66',ls=':',lw=1.2)
    ax.set_title(f'z = {z:.3f}  •  Selected window {info["width"]} Å  •  Retains {info["retained_slope_fraction"]:.0%} of fitted slope',loc='left',fontsize=12)
    ax.set_xlim(edge-850,end+15); ax.set_ylabel('Normalized flux density'); ax.set_xlabel('Rest-frame wavelength [Å]')
    ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Shrink uncertain endpoint slopes toward zero',fontsize=17,y=.99)
fig.text(.5,.952,'Window and shrinkage are chosen on inner wavelength holdouts. Green data are excluded from selection and fitting.',ha='center',fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.012),fontsize=10)
fig.tight_layout(rect=(0,.09,1,.94),h_pad=2)
fig.savefig(OUT/'regularized-slope-heldout-examples.png',dpi=165)
