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
methods=['constant','powerlaw']; records=[]; curves={}; selection=[]; slices=[]; rms_records=[]
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
            bin_errors=[]
            outward=300-distance[held]
            for number in range(3):
                mask=(outward>number*100)&(outward<=(number+1)*100)
                if not mask.any():
                    raise ValueError(f'Empty slice: {row} {side} {number}')
                actual=float(np.mean(f[held][mask])); predicted=float(np.mean(pred[mask]))
                error=predicted-actual; bin_errors.append(error)
                slices.append(dict(row=row,side=side,method=method,slice=number+1,n_pixels=int(mask.sum()),true_mean=actual,predicted_mean=predicted,signed_error=error,mean_sigma=float(np.sqrt(np.sum(1/iv[held][mask]))/mask.sum()),gap_from_nominal_cutoff_observed_aa=float(abs(boundary*(1+z)-obs[edge])-300)))
            rms_records.append(dict(row=row,side=side,method=method,rms=float(np.sqrt(np.mean(np.square(bin_errors))))))
            pm=float(np.mean(pred)); difference=pm-truth
            records.append(dict(row=row,targetid=int(d['targetid'][row]),z=z,zbin=int(np.searchsorted([0,.6,.9,1.2,2],z,side='right')-1),side=side,method=method,n_held_bins=int(held.sum()),true_mean=truth,mean_sigma=sigma,predicted_mean=pm,signed_mean_error=difference,absolute_mean_error=abs(difference),fractional_mean_error=abs(difference)/abs(truth) if truth>3*sigma else np.nan,negative=bool(np.any(pred<0)),pixel_rmse=float(np.sqrt(np.mean((pred-f[held])**2)))))
            curves[(row,side,method)]=pred
        curves[(row,side,'data')]=(obs[held],f[held],1/np.sqrt(iv[held]),boundary*(1+z),obs[training],train_f[training])
    if (counter+1)%100==0: print('Completed',counter+1,'/ 400 galaxies',flush=True)
B=pd.DataFrame(slices); R=pd.DataFrame(rms_records)
B.to_csv(OUT/'binned-300-edge-signed-errors.csv',index=False)
R.to_csv(OUT/'binned-300-edge-rms-errors.csv',index=False)
# Verify identical fits and held-out means to the preceding whole-region evaluation.
old=pd.read_csv(OUT/'last-300-edge-errors.csv').set_index(['row','side','method'])
new=pd.DataFrame(records).set_index(['row','side','method']).loc[old.index]
np.testing.assert_allclose(new.predicted_mean,old.predicted_mean,rtol=1e-10,atol=1e-10)
np.testing.assert_allclose(new.true_mean,old.true_mean,rtol=1e-10,atol=1e-10)
assert len(B)==4800 and len(R)==1600 and np.isfinite(R.rms).all()
rng=np.random.default_rng(20261009); summaries=[]
for side in ['blue','red']:
    q=R[R.side==side].pivot(index='row',columns='method',values='rms')
    ix=rng.integers(0,len(q),(10000,len(q)))
    delta=np.median(q.powerlaw.to_numpy()[ix],axis=1)-np.median(q.constant.to_numpy()[ix],axis=1)
    summaries.append(dict(side=side,constant_median=float(q.constant.median()),powerlaw_median=float(q.powerlaw.median()),constant_p90=float(q.constant.quantile(.9)),powerlaw_p90=float(q.powerlaw.quantile(.9)),median_difference=float(q.powerlaw.median()-q.constant.median()),difference_ci=np.quantile(delta,[.025,.975]).tolist(),powerlaw_win_fraction=float((q.powerlaw<q.constant).mean())))
protocol=dict(sample='Same exploratory 400 galaxies; no additional exclusions',holdout_observed_aa=300,slice_width_observed_aa=100,slice_order='Near to far from nominal cutoff: 0–100, 100–200, 200–300 outward. Actual retained training edge can be further inward after quality trimming.',fit_windows_rest_aa=dict(constant=100,powerlaw=500),quality='Unchanged fixed production 5% cut, including training-only re-trim',reference='Unsmoothed measured flux; unweighted mean within each slice',rms='sqrt(mean of squared signed slice-mean errors)); each of three slices has equal weight',limits='Reference noise and spectral features remain; slice means are not latent continuum truth. Reused exploratory sample, no full LSST-band validation.',verification='1600 fits/4800 slice errors; all finite, no empty slices; whole-region predictions and means reproduce preceding 300-AA evaluation at 1e-10',summary=summaries)
(OUT/'binned-300-edge-protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps(summaries,indent=2)); print('Reproduction and completeness checks passed.')
styles={'constant':('#286bb5','Constant · 100 rest Å'),'powerlaw':('#d87915','Power law · 500 rest Å')}
from matplotlib.ticker import FuncFormatter
fig,axes=plt.subplots(3,2,figsize=(12,11.5))
for col,side in enumerate(['blue','red']):
    frame=B[B.side==side]; ax=axes[0,col]
    for method,(color,label) in styles.items():
        offset=-.13 if method=='constant' else .13
        groups=[frame[(frame.method==method)&(frame.slice==i)].signed_error.to_numpy() for i in [1,2,3]]
        positions=np.arange(3)+offset
        parts=ax.violinplot(groups,positions=positions,widths=.24,showextrema=False)
        for body in parts['bodies']: body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.3)
        for pos,values in zip(positions,groups):
            lo,med,hi=np.quantile(values,[.1,.5,.9]); ax.plot([pos,pos],[lo,hi],color=color,lw=2); ax.plot(pos,med,'o',color=color,ms=4)
    ax.axhline(0,color='#7f8991',lw=.8)
    ax.set_xticks(range(3),['0–100','100–200','200–300'])
    ax.set_xlabel('Outward distance from nominal cutoff [observed Å]')
    ax.set_ylabel('Predicted − measured slice mean\n[normalized flux units]')
    ax.set_title(f'{side.capitalize()} edge · signed errors by slice',loc='left')
    ax.set_yscale('symlog',linthresh=.1)
    q=R[R.side==side]; allvalues=q.rms.to_numpy(); low=allvalues.min()/1.1; high=allvalues.max()*1.1
    bins=np.geomspace(low,high,30)
    for method,(color,label) in styles.items():
        values=q[q.method==method].rms.to_numpy()
        axes[1,col].hist(values,bins=bins,weights=np.full(len(values),.25),histtype='step',lw=1.8,color=color,label=label)
        x,counts=np.unique(np.sort(values),return_counts=True); probability=np.cumsum(counts[::-1])[::-1]*.25
        axes[2,col].step(x,probability,where='pre',color=color,lw=2)
        axes[2,col].plot(x[-1],probability[-1],'.',color=color)
    axes[1,col].set_title('RMS of the three slice errors',loc='left'); axes[1,col].set_ylabel('Galaxies per log bin [%]')
    axes[2,col].set_ylabel('Galaxies with RMS ≥ x [%]'); axes[2,col].set_yscale('log'); axes[2,col].set_ylim(.2,110)
    axes[2,col].set_yticks([.25,1,5,10,50,100]); axes[2,col].yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    for ax in axes[1:,col]: ax.set_xscale('log'); ax.set_xlim(low,high); ax.set_xlabel('RMS slice-mean error [normalized flux units]')
    for ax in axes[:,col]: ax.grid(axis='y',alpha=.15); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Continuum check: three 100 Å slices across the 300 Å holdout',fontsize=17,y=.99)
fig.text(.5,.956,'Same 400 galaxies · unchanged fits · slice means use unsmoothed measured flux',ha='center',fontsize=11)
fig.legend(*axes[1,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.text(.5,.017,'Violins show full signed-error distributions; dots: medians, bars: 10–90%. RMS prevents cancellation between slices. All outliers included.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.045,1,.88),h_pad=2,w_pad=2)
fig.savefig(OUT/'binned-300-edge-error-distributions.png',dpi=180)
fig.savefig(OUT/'binned-300-edge-error-distributions.pdf')
