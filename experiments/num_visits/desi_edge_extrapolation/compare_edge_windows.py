from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bedcosmo.num_visits.empirical.desi.training_matrix import extrapolate_spectrum_edges, _extrapolate_powerlaw

out=Path(__file__).parent
d=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w=d['wave_rest_aa']; windows=[100,200,500]; records=[]; failures=[]
def predict_red(w, f, train, held, method, width):
    # Restrict evaluation to the red holdout, avoiding unrelated blue/far-red tails.
    lo=np.flatnonzero(train>0)[0]; hi=np.flatnonzero(held)[-1]+1
    p,_=extrapolate_spectrum_edges(w[lo:hi],f[lo:hi],train[lo:hi],method=method,window_aa=width,continuum_window_aa=width)
    result=f.copy(); result[lo:hi]=p
    return result

rows=np.random.default_rng(42).choice(len(d['redshift']),100,replace=False)
for row in rows:
    f=d['flux'][row]; iv=d['relative_ivar'][row]; z=float(d['redshift'][row]); measured=iv>0; end=w[measured][-1]
    for offset,horizon in [(800,800),(800,400),(600,400),(400,400)]:
        cutoff=end-offset/(1+z)
        train=iv.copy(); train[w>=cutoff]=0
        held=measured&(w>=cutoff)&(w<cutoff+horizon/(1+z)+1e-7)
        if not held.any():
            raise ValueError('Empty held-out region')
        truth=np.average(f[held],weights=iv[held]); sigma=np.sqrt(1/iv[held].sum())
        for width in windows:
            for method in ['constant','powerlaw']:
                try:
                    pred=predict_red(w,f,train,held,method,width)
                except ValueError as exc:
                    failures.append(dict(row=int(row),offset=offset,horizon=horizon,window=width,method=method,error=str(exc)))
                    continue
                pm=np.average(pred[held],weights=iv[held])
                records.append(dict(row=int(row),targetid=int(d['targetid'][row]),z=z,offset=offset,horizon=horizon,window=width,method=method,true_mean=truth,pred_mean=pm,mean_sigma=sigma,error=abs(pm-truth)/abs(truth) if truth>3*sigma else np.nan))
t=pd.DataFrame(records); t.to_csv(out/'matched-edge-window-errors.csv',index=False)
# Use the same eligible galaxies in every cutoff/window/method of each horizon.
paired=[]
for horizon in [400,800]:
    block=t[t.horizon==horizon]
    pivot=block.pivot(index='row',columns=['offset','window','method'],values='error').dropna()
    paired.append(block[block.row.isin(pivot.index)])
summary=pd.concat(paired).groupby(['offset','horizon','window','method']).error.agg(['median','count']).reset_index()
summary.to_csv(out/'matched-edge-window-summary.csv',index=False)
(out/'matched-edge-window-failures.json').write_text(json.dumps(failures,indent=2))
print(summary.to_string(index=False)); print('Failures:',len(failures))

# Keep the same three illustrative galaxies and original held-out region.
fig,axs=plt.subplots(3,3,figsize=(16,11),sharex='row',sharey='row')
colors={'constant':'#286bb5','powerlaw':'#d87915'}
for i,row in enumerate([5230,5684,8955]):
    f=d['flux'][row]; iv=d['relative_ivar'][row]; z=float(d['redshift'][row]); end=w[iv>0][-1]
    hidden=(iv>0)&(w>=end-800/(1+z)); train=iv.copy(); train[hidden]=0; edge=w[train>0][-1]
    for j,width in enumerate(windows):
        ax=axs[i,j]; ax.axvspan(edge+5,end+5,color='#299568',alpha=.09); ax.axvspan(edge-width,edge+5,color='#ed8a23',alpha=.08)
        visible=(iv>0)&(w>=edge-550)
        ax.plot(w[visible],f[visible],color='#a8adb4',lw=.6,alpha=.45)
        for mask,color in [(visible&~hidden,'#414852'),(hidden,'#16845c')]:
            xx=[]; yy=[]; se=[]
            for lo in np.arange(edge-550,end+50,50):
                m=mask&(w>=lo)&(w<lo+50)
                if m.any():
                    xx.append(np.average(w[m],weights=iv[m])); yy.append(np.average(f[m],weights=iv[m])); se.append(np.sqrt(1/iv[m].sum()))
            ax.errorbar(xx,yy,yerr=se,color=color,fmt='o-',ms=3,lw=.9,capsize=2)
        for method in colors:
            p=predict_red(w,f,train,hidden,method,width)
            fit=(train>0)&(w>=edge-width)
            fitted=np.full(fit.sum(),p[hidden][0]) if method=='constant' else _extrapolate_powerlaw(w[fit],f[fit],iv[fit],edge,w[fit])
            ax.plot(w[fit],fitted,color=colors[method],lw=1.6)
            ax.plot(w[hidden],p[hidden],ls='--',color=colors[method],lw=2,label=method)
        sub=t[(t.row==row)&(t.offset==800)&(t.horizon==800)&(t.window==width)].set_index('method')
        ax.text(.03,.96,f'Constant {sub.loc["constant","error"]:.1%}\nPower law {sub.loc["powerlaw","error"]:.1%}',transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax.axvline(edge+5,color='#545c66',ls=':',lw=1)
        ax.set_xlim(edge-550,end+10); ax.grid(alpha=.12); ax.spines[['top','right']].set_visible(False)
        if i==0: ax.set_title(f'Both fit the last {width} rest-frame Å',fontsize=12)
        if j==0: ax.set_ylabel(f'z = {z:.3f}\nNormalized flux density')
        if i==2: ax.set_xlabel('Rest-frame wavelength [Å]')
fig.suptitle('Same spectra and cutoff; change only the fitting-window width',fontsize=17,y=.99)
fig.text(.5,.95,'Green: measured data withheld from fits • Blue dashed: constant • Orange dashed: power law • Labels: mean-flux error',ha='center',fontsize=11)
fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(out/'matched-edge-window-examples.png',dpi=160)

# Compare cutoff positions at a fixed 400-observed-Angstrom prediction horizon.
fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
for ax,offset in zip(axs,[800,600,400]):
    for method in colors:
        sub=summary[(summary.offset==offset)&(summary.horizon==400)&(summary.method==method)].sort_values('window')
        ax.plot(sub.window,100*sub['median'],'o-',color=colors[method],label=method,lw=2)
        for x,y in zip(sub.window,100*sub['median']): ax.annotate(f'{y:.1f}%',(x,y),xytext=(0,8 if y>float(summary[(summary.offset==offset)&(summary.horizon==400)&(summary.window==x)&(summary.method!=method)]['median'].iloc[0])*100 else -16),textcoords='offset points',ha='center',fontsize=10)
    ax.set_title(f'Cutoff {offset} Å before measured end\n(n = {int(sub["count"].iloc[0])} common galaxies)',fontsize=11)
    ax.set_xticks(windows); ax.set_xlim(65,540); ax.set_ylim(0,35); ax.set_xlabel('Fitting window [rest-frame Å]'); ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Median absolute mean-flux error [%]'); axs[-1].legend(frameon=False)
fig.suptitle('Shift the cutoff redward, keeping the prediction horizon fixed',fontsize=15)
fig.text(.5,.9,'100 random galaxies; common eligible subset at all cutoffs. Predict next 400 observed-frame Å; measured mean > 3σ.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.86)); fig.savefig(out/'matched-edge-cutoff-summary.png',dpi=170)
