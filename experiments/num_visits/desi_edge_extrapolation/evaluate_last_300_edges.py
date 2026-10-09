"""Direct last-300-observed-Angstrom holdout comparison, with fixed methods."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from bedcosmo.num_visits.empirical.desi.training_matrix import trim_spectrum_edges,_extrapolate_powerlaw
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
RESEARCH=OUT
with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    d={k:saved[k] for k in ['wave_rest_aa','flux','relative_ivar','redshift','targetid']}
sample=json.loads((RESEARCH/'edge-tuning-protocol.json').read_text())['splits']['test']
methods=['constant','powerlaw']; records=[]; curves={}; selection=[]
for counter,item in enumerate(sample):
    row=item['row']; z=float(d['redshift'][row]); w=d['wave_rest_aa']; obs=w*(1+z)
    original_f=d['flux'][row]; original_iv=d['relative_ivar'][row]
    f,iv=trim_spectrum_edges(w,original_f,original_iv,z); measured=iv>0
    first,last=np.flatnonzero(measured)[[0,-1]]
    for side,edge in [('blue',first),('red',last)]:
        distance=(obs-obs[edge]) if side=='blue' else (obs[edge]-obs)
        held=measured&(distance>=0)&(distance<300)
        available=measured&~held
        # Apply the same fixed production cut to the available training data.
        train_f,train_iv=trim_spectrum_edges(w,np.where(available,f,0),np.where(available,iv,0),z)
        training=train_iv>0; mw=w[training]; mf=train_f[training]; mi=train_iv[training]; target=w[held]
        # All estimators use only available data; their existing fit widths stay fixed.
        smooth=median_filter(mf,size=5,mode='reflect')
        boundary=mw[0] if side=='blue' else mw[-1]
        window=np.abs(mw-boundary)<=100
        predictions={'constant':np.full(len(target),np.mean(smooth[window]))}
        window=np.abs(mw-boundary)<=500
        predictions['powerlaw']=_extrapolate_powerlaw(mw[window],mf[window],mi[window],boundary,target)
        truth=float(np.mean(f[held])); sigma=float(np.sqrt(np.sum(1/iv[held]))/held.sum())
        for method,pred in predictions.items():
            pm=float(np.mean(pred)); difference=pm-truth
            records.append(dict(row=row,targetid=int(d['targetid'][row]),z=z,zbin=int(np.searchsorted([0,.6,.9,1.2,2],z,side='right')-1),side=side,method=method,n_held_bins=int(held.sum()),true_mean=truth,mean_sigma=sigma,predicted_mean=pm,signed_mean_error=difference,absolute_mean_error=abs(difference),fractional_mean_error=abs(difference)/abs(truth) if truth>3*sigma else np.nan,negative=bool(np.any(pred<0)),pixel_rmse=float(np.sqrt(np.mean((pred-f[held])**2)))))
            curves[(row,side,method)]=pred
        curves[(row,side,'data')]=(obs[held],f[held],1/np.sqrt(iv[held]),boundary*(1+z),obs[training],train_f[training])
    if (counter+1)%100==0: print('Completed',counter+1,'/ 400 galaxies',flush=True)
T=pd.DataFrame(records); T.to_csv(OUT/'last-300-edge-errors.csv',index=False)
(OUT/'last-300-edge-selections.json').write_text(json.dumps(selection,indent=2,allow_nan=False))
rng=np.random.default_rng(20261012); summary=[]; comparison=[]
for side in ['blue','red']:
    q=T[T.side==side].pivot(index='row',columns='method',values='absolute_mean_error')
    ix=rng.integers(0,len(q),(10000,len(q)))
    boots={m:np.median(q[m].to_numpy()[ix],axis=1) for m in methods}
    for method in methods:
        rows=T[(T.side==side)&(T.method==method)]
        lo,hi=np.quantile(boots[method],[.025,.975])
        summary.append(dict(side=side,method=method,n=len(rows),median_absolute_mean_error=float(q[method].median()),ci_low=float(lo),ci_high=float(hi),p90_absolute_mean_error=float(q[method].quantile(.9)),positive_3sigma_n=int(rows.fractional_mean_error.notna().sum()),median_fractional_mean_error=float(rows.fractional_mean_error.median()),negative_fraction=float(rows.negative.mean()),median_truth_sigma=float(rows.mean_sigma.median())))
        diff=boots[method]-boots['constant']; lo,hi=np.quantile(diff,[.025,.975])
        comparison.append(dict(side=side,method=method,median_error_difference_vs_constant=float(q[method].median()-q['constant'].median()),difference_ci_low=float(lo),difference_ci_high=float(hi),fraction_better_than_constant=float((q[method]<q['constant']).mean())))
S=pd.DataFrame(summary); S.to_csv(OUT/'last-300-edge-summary.csv',index=False)
C=pd.DataFrame(comparison); C.to_csv(OUT/'last-300-edge-paired-comparison.csv',index=False)
protocol=dict(sample='Reuse the fixed 400-galaxy test sample from the earlier study, 100 per redshift bin; no new per-edge S/N screening',rows=[i['row'] for i in sample],holdout_observed_aa=300,quality='Fixed production 5% / 3-bin cut before defining physical endpoints and again on training-only pixels',constant_window_rest_aa=100,powerlaw_window_rest_aa=500,primary='Absolute difference of unweighted predicted and observed mean flux in withheld bins; flux normalized by original robust per-galaxy normalization',secondary='Fractional mean error only where observed held-out mean is positive and >3 sigma; report subset size',uncertainty='10,000 paired galaxy bootstrap resamples; noisy observed reference, not known latent continuum; errors in rebinning treated as independent',limits='300 observed Angstrom is shorter than the actual missing LSST tails. The fixed sample was originally screened for g/z coverage and S/N. This sample has already informed prior exploratory discussion; this is a new diagnostic, not a new untouched final test. No u/y band integration or NMF validation.',synthetic='No production jobs or matrix writes')
(OUT/'last-300-edge-protocol.json').write_text(json.dumps(protocol,indent=2))
print(S.to_string(index=False)); print(C.to_string(index=False))
styles={'constant':('#286bb5','Constant, 100 rest Å'),'powerlaw':('#d87915','Power law, 500 rest Å')}
fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),sharey=True)
for ax,side in zip(axes,['blue','red']):
    for k,method in enumerate(methods):
        r=S[(S.side==side)&(S.method==method)].iloc[0]
        ax.errorbar(k,r.median_absolute_mean_error,yerr=[[r.median_absolute_mean_error-r.ci_low],[r.ci_high-r.median_absolute_mean_error]],fmt='o',color=styles[method][0],capsize=6,ms=8,lw=2)
        ax.annotate(f'{r.median_absolute_mean_error:.3f}',(k,r.ci_high),xytext=(0,8),textcoords='offset points',ha='center',fontsize=11)
    ax.set_xticks(range(2),['Constant','Power law']); ax.set_xlim(-.5,1.5); ax.set_ylim(bottom=0)
    ax.set_title(f'{side.capitalize()} DESI edge · 400 galaxies',loc='left',fontsize=12)
    ax.grid(axis='y',alpha=.18); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Median absolute error in withheld mean flux\n[in units of each galaxy’s flux normalization]')
fig.suptitle('Predict only the last 300 observed Å at each DESI edge',fontsize=16)
fig.text(.5,.89,'Fit using data inward of the cutoff; score the withheld region directly. Whiskers: 95% galaxy-bootstrap intervals.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.87)); fig.savefig(OUT/'last-300-edge-method-comparison.png',dpi=180)
# Choose examples by predefined redshift-bin position, not extrapolation outcome.
example_rows=[sample[0]['row'],sample[200]['row'],sample[300]['row']]
fig,axes=plt.subplots(3,2,figsize=(12,10))
for axrow,row in zip(axes,example_rows):
    z=float(d['redshift'][row])
    for ax,side in zip(axrow,['blue','red']):
        x,true,sigma,boundary,tx,tf=curves[(row,side,'data')]
        lo=x.min()-200 if side=='red' else x.min()-10
        hi=x.max()+200 if side=='blue' else x.max()+10
        inside=(tx>=lo)&(tx<=hi)
        ax.plot(tx[inside],tf[inside],color='#8d969f',lw=1,label='Available DESI')
        ax.axvspan(x.min()-5,x.max()+5,color='#48a17c',alpha=.10)
        ax.errorbar(x,true,yerr=sigma,fmt='o',ms=4,color='#238363',capsize=3,label='Withheld data ±1σ')
        for method in methods:
            ax.plot(x,curves[(row,side,method)],color=styles[method][0],lw=2,label=styles[method][1])
        ax.axvline(boundary,color='#64717e',ls=':',lw=1)
        ax.set_xlim(lo,hi); ax.set_title(f'{side.capitalize()} edge · z = {z:.3f}',loc='left',fontsize=11)
        ax.set_ylabel('Normalized flux'); ax.set_xlabel('Observed wavelength [Å]')
        ax.grid(axis='y',alpha=.14); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Last-300-Å holdout: predictions against actual withheld DESI data',fontsize=15,y=.995)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0),fontsize=10)
fig.tight_layout(rect=(0,.085,1,.97)); fig.savefig(OUT/'last-300-edge-examples.png',dpi=170)
