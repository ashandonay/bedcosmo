"""Exploratory red-edge test. Reads the saved matrix; writes only beside this script."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bedcosmo.num_visits.empirical.desi.training_matrix import _extrapolate_powerlaw

OUT=Path(__file__).parent
WIDTHS=[100,200,300,500,800]

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

def select_window(w,f,iv,horizon,method):
    """Three adjacent inner holdouts; all inputs precede the outer holdout."""
    score=[]
    for width in WIDTHS:
        total=0.; norm=0.
        for j in [1,2,3]:
            cutoff=w[-1]-j*horizon
            train=w<cutoff
            valid=(w>=cutoff)&(w<cutoff+horizon+1e-7)
            if train.sum()<5 or valid.sum()<3:
                raise ValueError('Insufficient inner validation coverage')
            prediction,_=predict(w[train],f[train],iv[train],w[valid],width,method)
            # Predict 50-Angstrom continuum bins, including their measured noise.
            for lo in np.arange(cutoff,cutoff+horizon,50):
                block=valid&(w>=lo)&(w<lo+50)
                if not block.any():
                    continue
                idx=(w[valid]>=lo)&(w[valid]<lo+50)
                mean_res=np.average(prediction[idx]-f[block],weights=iv[block])
                total+=iv[block].sum()*mean_res**2
                norm+=iv[block].sum()
        score.append(total/norm)
    return WIDTHS[int(np.argmin(score))],score

# Deterministic checks on the fit and endpoint tangent.
sw=np.arange(1000.,1601.,10.); sx=(sw-sw[-1])/500
sf=2+.4*sx-.3*sx*sx
beta,se=quadratic_fit(sw,sf,np.ones(len(sw))*100,500)
assert np.allclose(beta,[2,.4,-.3],atol=1e-8)
pr,_=predict(sw,sf,np.ones(len(sw))*100,np.array([1610.,1700.]),500,'local_quadratic')
assert np.allclose(pr,[2.008,2.08],atol=1e-8)
assert se>0

D=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w=D['wave_rest_aa']; rows=np.random.default_rng(42).choice(len(D['redshift']),100,replace=False)
records=[]; selections=[]; failures=[]; curves={}
for k,row in enumerate(rows):
    flux=D['flux'][row]; iv=D['relative_ivar'][row]; z=float(D['redshift'][row]); observed=iv>0
    end=w[observed][-1]; outer=observed&(w>=end-800/(1+z)); available=observed&~outer
    tw=w[available]; tf=flux[available]; ti=iv[available]; target=w[outer]
    horizon=800/(1+z)
    truth=np.average(flux[outer],weights=iv[outer]); sigma=np.sqrt(1/iv[outer].sum())
    for method in ['constant','powerlaw','local_quadratic']:
        try:
            width,scores=select_window(tw,tf,ti,horizon,method)
            p,info=predict(tw,tf,ti,target,width,method)
        except ValueError as exc:
            failures.append(dict(row=int(row),method=method,error=str(exc)))
            continue
        selections.append(dict(row=int(row),method=method,width=width,inner_scores=scores,**info))
        curves[(int(row),method)]=(width,p,info)
        pm=np.average(p,weights=iv[outer])
        records.append(dict(row=int(row),method='selected_'+method,width=width,z=z,true_mean=truth,mean_sigma=sigma,mean_error=abs(pm-truth)/abs(truth) if truth>3*sigma else np.nan,negative_bins=int((p<0).sum()),min_prediction=float(p.min())))
    for method,width in [('constant',100),('powerlaw',500)]:
        p,_=predict(tw,tf,ti,target,width,method)
        pm=np.average(p,weights=iv[outer])
        records.append(dict(row=int(row),method=f'fixed_{method}_{width}',width=width,z=z,true_mean=truth,mean_sigma=sigma,mean_error=abs(pm-truth)/abs(truth) if truth>3*sigma else np.nan,negative_bins=int((p<0).sum()),min_prediction=float(p.min())))
    if (k+1)%25==0: print('Completed',k+1,'galaxies',flush=True)
T=pd.DataFrame(records); T.to_csv(OUT/'local-quadratic-errors.csv',index=False)
(OUT/'local-quadratic-selections.json').write_text(json.dumps(selections,indent=2))
(OUT/'local-quadratic-failures.json').write_text(json.dumps(failures,indent=2))
P=T.pivot(index='row',columns='method',values='mean_error').dropna()
summary={name:dict(n=len(P),median_mean_error=float(P[name].median()),negative_prediction_galaxies=int((T[T.method==name].set_index('row').loc[P.index].negative_bins>0).sum())) for name in P.columns}
summary['details']=dict(outer_observed_width_aa=800,inner_blocks=3,windows_rest_aa=WIDTHS,selection_score='inverse-variance-weighted squared residual of 50-rest-Angstrom mean-flux bins',failed_configurations=failures)
(OUT/'local-quadratic-summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)

# The same illustrative spectra used previously; no selection based on this test's outcomes.
fig,axs=plt.subplots(3,1,figsize=(12,12))
styles={'constant':('#286bb5','--'),'powerlaw':('#d87915','--'),'local_quadratic':('#823bb2','--')}
for ax,row in zip(axs,[5230,5684,8955]):
    f=D['flux'][row]; iv=D['relative_ivar'][row]; z=float(D['redshift'][row]); end=w[iv>0][-1]
    held=(iv>0)&(w>=end-800/(1+z)); train=(iv>0)&~held; edge=w[train][-1]
    width,p,info=curves[(row,'local_quadratic')]
    ax.axvspan(edge-width,edge+5,color='#823bb2',alpha=.07)
    ax.axvspan(edge+5,end+5,color='#299568',alpha=.09)
    visible=(iv>0)&(w>=edge-850)
    ax.plot(w[visible],f[visible],color='#a8adb4',lw=.6,alpha=.45)
    for mask,color,label in [(visible&~held,'#414852','Available data'),(held,'#16845c','Withheld data')]:
        xx=[]; yy=[]; se=[]
        for lo in np.arange(edge-850,end+50,50):
            m=mask&(w>=lo)&(w<lo+50)
            if m.any():
                xx.append(np.average(w[m],weights=iv[m])); yy.append(np.average(f[m],weights=iv[m])); se.append(np.sqrt(1/iv[m].sum()))
        ax.errorbar(xx,yy,yerr=se,color=color,fmt='o-',ms=4,lw=1,capsize=2,label=label,zorder=4)
    fit=w[train]>(edge-width)
    fw=w[train][fit]; bx=(fw-edge)/width; b=np.array(info['beta'])
    ax.plot(fw,b[0]+b[1]*bx+b[2]*bx*bx,color='#823bb2',lw=2,label='Selected quadratic fit')
    texts=[]
    for method,(color,ls) in styles.items():
        width,p,inf=curves[(row,method)]
        err=float(T[(T.row==row)&(T.method=='selected_'+method)].mean_error.iloc[0])
        name={'constant':'Constant','powerlaw':'Power law','local_quadratic':'Quadratic → tangent'}[method]
        ax.plot(w[held],p,color=color,ls=ls,lw=2.2,label=name)
        texts.append(f'{name}: {err:.1%} (window {width} Å)')
    ax.text(.02,.96,'\n'.join(texts),transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    ax.axvline(edge+5,color='#545c66',ls=':',lw=1.3)
    ax.set_title(f'z = {z:.3f}  •  Target {int(D["targetid"][row])}  •  Endpoint slope {info["slope"]*100:+.3f} ± {info["slope_se"]*100:.3f} per 100 Å*',loc='left',fontsize=11)
    ax.set_xlim(edge-850,end+15); ax.set_ylabel('Normalized flux density'); ax.set_xlabel('Rest-frame wavelength [Å]')
    ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Let the measured continuum bend; extrapolate its endpoint tangent',fontsize=16,y=.99)
fig.text(.5,.955,'Each method selects its window using three inner holdouts. Green outer holdout is excluded from all selection and fitting.',ha='center',fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025),fontsize=10)
fig.text(.5,.012,'*Approximate fit uncertainty conditional on the selected window; excludes selection uncertainty. Flux is not clipped to stay positive.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.1,1,.945),h_pad=2)
fig.savefig(OUT/'local-quadratic-heldout-examples.png',dpi=165)
