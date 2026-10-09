"""Change only the robust power-law fit width in the last-100-AA diagnostic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from bedcosmo.num_visits.empirical.desi.training_matrix import trim_spectrum_edges,_extrapolate_powerlaw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
OUT=Path(__file__).parent
PREVIOUS=pd.read_csv(OUT/'last-100-edge-errors.csv')
WIDTHS=[500,1000]
with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    d={k:saved[k] for k in ['wave_rest_aa','flux','relative_ivar','redshift','targetid']}
rows=PREVIOUS.row.drop_duplicates().astype(int).tolist(); records=[]
for counter,row in enumerate(rows):
    w=d['wave_rest_aa']; z=float(d['redshift'][row]); obs=w*(1+z)
    f,iv=trim_spectrum_edges(w,d['flux'][row],d['relative_ivar'][row],z); measured=iv>0
    first,last=np.flatnonzero(measured)[[0,-1]]
    for side,edge in [('blue',first),('red',last)]:
        distance=obs-obs[edge] if side=='blue' else obs[edge]-obs
        held=measured&(distance>=0)&(distance<100); available=measured&~held
        tf,ti=trim_spectrum_edges(w,np.where(available,f,0),np.where(available,iv,0),z)
        training=ti>0; mw=w[training]; mf=tf[training]; mi=ti[training]; target=w[held]
        boundary=mw[0] if side=='blue' else mw[-1]
        truth=float(np.mean(f[held])); sigma=float(np.sqrt(np.sum(1/iv[held]))/held.sum())
        for width in WIDTHS:
            window=np.abs(mw-boundary)<=width
            try:
                pred=_extrapolate_powerlaw(mw[window],mf[window],mi[window],boundary,target)
                pm=float(np.mean(pred)); error=pm-truth; failure=''
            except ValueError as exc:
                pm=float('nan'); error=float('inf'); failure=str(exc)
            records.append(dict(row=row,targetid=int(d['targetid'][row]),z=z,side=side,width_rest_aa=width,method=f'powerlaw{width}',n_fit_bins=int(window.sum()),n_held_bins=int(held.sum()),true_mean=truth,mean_sigma=sigma,predicted_mean=pm,signed_mean_error=error,absolute_mean_error=abs(error),failure=failure))
    if (counter+1)%100==0: print('Completed',counter+1,'/ 400 galaxies',flush=True)
T=pd.DataFrame(records)
# Exactly reproduce the previous 500-AA fit and holdout references.
old=PREVIOUS[PREVIOUS.method=='powerlaw'].set_index(['row','side'])
new=T[T.width_rest_aa==500].set_index(['row','side']).loc[old.index]
np.testing.assert_allclose(new.predicted_mean,old.predicted_mean,rtol=1e-10,atol=1e-10)
np.testing.assert_allclose(new.true_mean,old.true_mean,rtol=1e-10,atol=1e-10)
T.to_csv(OUT/'powerlaw-1000-window-errors.csv',index=False)
constant=PREVIOUS[PREVIOUS.method=='constant'].copy(); constant['method']='constant'; constant['failure']=''; constant['width_rest_aa']=100
T=pd.concat([T,constant],ignore_index=True)
method_order=['constant']+[f'powerlaw{x}' for x in WIDTHS]
summary=[]; paired=[]; rng=np.random.default_rng(20261013)
for side in ['blue','red']:
    q=T[T.side==side].pivot(index='row',columns='method',values='absolute_mean_error')
    indices=rng.integers(0,len(q),(10000,len(q)))
    ref=np.median(q.powerlaw500.to_numpy()[indices],axis=1)
    for method in method_order:
        values=q[method].to_numpy(); boot=np.median(values[indices],axis=1)
        ci=np.quantile(boot-ref,[.025,.975])
        summary.append(dict(side=side,method=method,n=len(values),failures=int(np.isinf(values).sum()),median=float(np.median(values)),p90=float(np.quantile(values,.90)),p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)),maximum=float(np.max(values))))
        paired.append(dict(side=side,method=method,median_difference_vs_500=float(np.median(values)-np.median(q.powerlaw500)),ci_low=float(ci[0]),ci_high=float(ci[1]),fraction_better_than_500=float(np.mean(values<q.powerlaw500))))
S=pd.DataFrame(summary); S.to_csv(OUT/'powerlaw-1000-window-summary.csv',index=False)
P=pd.DataFrame(paired); P.to_csv(OUT/'powerlaw-1000-window-paired-comparison.csv',index=False)
protocol=dict(sample='Same 400 galaxies and quality/holdout definitions as last-100-edge-protocol.json; not a new independent final test',rows=rows,holdout_observed_aa=100,fit_windows_rest_aa=WIDTHS,smoothing='None for power law; raw signed measured flux with ivar-weighted soft-L1 fit',quality='Fixed production 5% / three-consecutive-bin cut',primary='Absolute error in unweighted mean of withheld normalized flux',baseline='Existing constant100 result, five-bin median smoothing',failures='Record fit failures as infinite error, without removing their galaxies from denominators',verification='500-AA predictions and withheld means reproduce prior comparison to 1e-10 relative/absolute tolerance',limits='Noisy reference, especially blue; only a 100-observed-AA horizon, shorter than actual missing u/y coverage; choosing a window on these results requires a fresh subsequent test')
(OUT/'powerlaw-1000-window-protocol.json').write_text(json.dumps(protocol,indent=2))
print(S.to_string(index=False)); print(P.to_string(index=False));print('500-AA reproduction checks passed.')
styles={'constant':('#68737f','Constant, 100 Å'),'powerlaw100':('#269679','Power law, 100 Å'),'powerlaw200':('#286bb5','Power law, 200 Å'),'powerlaw300':('#823bb2','Power law, 300 Å'),'powerlaw500':('#d87915','Power law, 500 Å'),'powerlaw1000':('#b43256','Power law, 1000 Å')}
fig,axes=plt.subplots(2,2,figsize=(12,8.7),sharex='col')
for col,side in enumerate(['blue','red']):
    frame=T[T.side==side]; allvalues=frame.absolute_mean_error.to_numpy(); finite=allvalues[np.isfinite(allvalues)]
    low=np.nextafter(finite[finite>0].min(),0); high=np.nextafter(finite.max(),np.inf)
    bins=np.geomspace(low,high,31)
    for method in method_order:
        color,label=styles[method]; values=frame[frame.method==method].absolute_mean_error.to_numpy(); finite=values[np.isfinite(values)]; failures=int(np.isinf(values).sum())
        if failures: label+=f' ({failures} failures)'
        axes[0,col].hist(finite,bins=bins,weights=np.full(len(finite),100/len(values)),histtype='step',lw=1.5,color=color,label=label)
        x,counts=np.unique(np.sort(finite),return_counts=True); prob=(np.cumsum(counts[::-1])[::-1]+failures)*100/len(values)
        axes[1,col].step(x,prob,where='pre',lw=1.8,color=color)
        axes[1,col].plot(x[-1],prob[-1],'.',color=color,ms=4)
    axes[0,col].set_title(f'{side.capitalize()} edge · 400 galaxies per fit window',loc='left',fontsize=12)
    axes[0,col].set_ylabel('Galaxies per logarithmic bin [%]')
    axes[1,col].set_ylabel('Galaxies with error ≥ x [%]')
    axes[1,col].set_xlabel('Absolute error in withheld mean flux\n[normalized flux units; logarithmic axis]')
    axes[1,col].set_yscale('log'); axes[1,col].set_ylim(.2,110)
    axes[1,col].set_yticks([.25,1,5,10,50,100]); axes[1,col].yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    for ax in axes[:,col]:
        ax.set_xscale('log'); ax.set_xlim(low/1.1,high*1.1); ax.grid(axis='y',alpha=.14); ax.spines[['top','right']].set_visible(False)
    axes[1,col].axhline(1,color='#9ba4ae',ls=':',lw=.7)
fig.suptitle('Does a 1000 Å power-law fitting window help?',fontsize=17,y=.99)
fig.text(.5,.949,'Fit raw flux with ivar weighting and robust loss · withhold the last 100 observed Å · fitting windows below are rest-frame Å',ha='center',fontsize=10)
fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.929),fontsize=10)
fig.text(.5,.015,'All galaxies and finite outliers shown. Lower tail curves mean fewer large errors. Blue/red x ranges differ; errors include measurement noise.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.85),h_pad=2.2,w_pad=2)
fig.savefig(OUT/'powerlaw-1000-window-error-distributions.png',dpi=180)
fig.savefig(OUT/'powerlaw-1000-window-error-distributions.pdf')
