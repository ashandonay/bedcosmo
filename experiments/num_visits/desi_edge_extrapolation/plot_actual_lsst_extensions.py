"""Actual DESI-edge extensions to LSST bandpass bounds; no outer holdout."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from speclite import filters
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
    d={k:saved[k] for k in ['wave_rest_aa','redshift','flux','relative_ivar','targetid']}
w=d['wave_rest_aa']; zall=d['redshift']; good=d['relative_ivar']>0
median_weight=np.nanmedian(np.where(good,d['relative_ivar'],np.nan),axis=1)
rng=np.random.default_rng(20261009)
rows=[]
for lo,hi in [(0,.2),(.2,.4),(.4,.7),(.7,1),(1,1.3),(1.3,1.7)]:
    candidates=np.flatnonzero((zall>=lo)&(zall<hi))
    threshold=np.quantile(median_weight[candidates],.75)
    rows.append(int(rng.choice(candidates[median_weight[candidates]>=threshold])))
lsst=filters.load_filters(*[f'lsst2023-{band}' for band in 'ugrizy'])
blue=min(f.wavelength.min() for f in lsst); red=max(f.wavelength.max() for f in lsst)
colors={'constant':'#286bb5','powerlaw':'#d87915','regularized':'#823bb2'}
labels={'constant':'Constant, 100 Å','powerlaw':'Power law, 500 Å','regularized':'Regularized local slope'}
all_curves={}; records=[]
for row in rows:
    z=float(zall[row]); measured=good[row]; mw=w[measured]; mf=d['flux'][row,measured]; mi=d['relative_ivar'][row,measured]
    smooth=median_filter(mf,size=5,mode='reflect')
    item=dict(row=row,targetid=int(d['targetid'][row]),redshift=z,measured_observed_bounds=(mw[[0,-1]]*(1+z)).tolist(),edges={})
    for side,bound in [('blue',blue/(1+z)),('red',red/(1+z))]:
        edge=mw[0] if side=='blue' else mw[-1]
        if (side=='blue' and bound>=edge) or (side=='red' and bound<=edge):
            continue
        target=np.linspace(bound,edge,130) if side=='blue' else np.linspace(edge,bound,130)
        window=abs(mw-edge)<=100
        const=np.full(len(target),np.mean(smooth[window]))
        window=abs(mw-edge)<=500
        power=_extrapolate_powerlaw(mw[window],mf[window],mi[window],edge,target)
        # Reflect the blue wavelength axis so both edges use the same outward direction.
        rw,rf,ri,rt=(-mw[::-1],mf[::-1],mi[::-1],-target[::-1]) if side=='blue' else (mw,mf,mi,target)
        horizon=abs(bound-edge)
        reg,info=fit_selected(rw,rf,ri,rt,horizon)
        if side=='blue': reg=reg[::-1]
        all_curves[(row,side)]=(target*(1+z),{'constant':const,'powerlaw':power,'regularized':reg})
        item['edges'][side]=dict(required_rest_width_aa=horizon,selected_window_rest_aa=info['width'],retained_slope_fraction=info['retained_slope_fraction'],prior_fractional_slope_std_per_100_aa=info['tau_fraction'],regularized_min_flux=float(reg.min()))
        assert target.min()>=blue/(1+z)-1e-6 and target.max()<=red/(1+z)+1e-6
        assert all(np.all(np.isfinite(a)) for a in [const,power,reg])
    records.append(item)

for page in range(2):
    fig,axs=plt.subplots(4,1,figsize=(13,11.5),sharex=True,gridspec_kw={'height_ratios':[.5,1,1,1]},layout='constrained')
    for band in lsst:
        axs[0].plot(band.wavelength,band.response,lw=1.3,label=band.name[-1])
    axs[0].set_ylabel('LSST throughput'); axs[0].legend(ncol=6,frameon=False,loc='upper right'); axs[0].set_ylim(0,1)
    for ax,row in zip(axs[1:],rows[page*3:page*3+3]):
        z=float(zall[row]); measured=good[row]; x=w*(1+z)
        # NaNs break internal gaps; all measured matrix bins remain unchanged.
        displayed=np.where(measured,d['flux'][row],np.nan)
        ax.plot(x,displayed,color='#494e56',lw=.65,alpha=.7,label='Measured DESI (all bins)')
        details=[]
        for side in ['blue','red']:
            if (row,side) not in all_curves: continue
            target,preds=all_curves[(row,side)]
            measured_edge=x[measured][0 if side=='blue' else -1]
            ax.axvspan(target.min(),target.max(),color='#747c88',alpha=.09)
            ax.axvline(measured_edge,color='#444a52',ls=':',lw=1)
            ax.plot([measured_edge],[d['flux'][row,measured][0 if side=='blue' else -1]],'o',color='#252a32',ms=4,zorder=5)
            for method in colors:
                ax.plot(target,preds[method],color=colors[method],lw=2.3,ls='-',label=labels[method] if side=='blue' else None)
            info=next(r for r in records if r['row']==row)['edges'][side]
            details.append(f'{side}: {info["selected_window_rest_aa"]} Å, retains {info["retained_slope_fraction"]:.0%} slope')
        ax.set_title(f'z = {z:.3f}  •  Target {int(d["targetid"][row])}  •  Regularized {"; ".join(details)}',loc='left',fontsize=10)
        ax.set_ylabel('Normalized flux density'); ax.grid(alpha=.12); ax.spines[['top','right']].set_visible(False)
        # Autoscaling includes every measured point and all predictions, including negatives.
        plotted=[displayed[(x>=blue)&(x<=red)]]
        for side in ['blue','red']:
            if (row,side) in all_curves: plotted.extend(all_curves[(row,side)][1].values())
        values=np.concatenate(plotted); values=values[np.isfinite(values)]
        pad=.08*max(np.ptp(values),.1); ax.set_ylim(values.min()-pad,values.max()+pad)
    axs[1].legend(ncol=4,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.02))
    axs[-1].set_xlim(blue,red); axs[-1].set_xlabel('Observed wavelength [Å]')
    fig.suptitle(f'Actual DESI extensions across LSST ugrizy — examples {page*3+1}–{page*3+3}\nShaded tails are inferred; dotted boundaries mark the actual measured endpoints.',fontsize=14)
    fig.savefig(OUT/f'actual-lsst-three-methods-{page+1}.png',dpi=165)
(OUT/'actual-lsst-three-methods-provenance.json').write_text(json.dumps(dict(selection='One seeded random galaxy from the upper quartile of median measured inverse variance in each of six redshift bins; selected before fitting',lsst_observed_bounds_aa=[float(blue),float(red)],regularized_selection='Three inner blocks, each as wide as the required extension at that edge; blue axis reflected; fit final model to all measured data',measured_data='All valid bins retained unchanged; no outer holdout; no flux clipping',examples=records),indent=2,allow_nan=False))
print(json.dumps(records,indent=2))
