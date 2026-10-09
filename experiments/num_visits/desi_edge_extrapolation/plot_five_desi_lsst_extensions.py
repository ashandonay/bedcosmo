"""Five real DESI galaxies: constant and power-law continuations with the agreed 5% quality cut."""
from pathlib import Path
import json
import numpy as np
from speclite import filters
from bedcosmo.num_visits.empirical.desi.training_matrix import trim_spectrum_edges,extrapolate_spectrum_edges,_extrapolate_powerlaw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    d={k:saved[k] for k in ['wave_rest_aa','flux','relative_ivar','redshift','targetid']}
# Reuse five previously shown objects, spaced across redshift, without selecting on fit performance.
rows=[8235,3302,5285,11428,7666]
bank=filters.load_filters(*['lsst2023-'+b for b in 'ugrizy'])
limits=np.array([min(b.wavelength.min() for b in bank),max(b.wavelength.max() for b in bank)])
styles={'constant':('#286bb5','Constant · 100 Å'),'powerlaw':('#d87915','Power law · 500 Å')}
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':10,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(14,13))
grid=fig.add_gridspec(6,3,height_ratios=[.50,1,1,1,1,1],width_ratios=[1,2.25,1],hspace=.51,wspace=.24)
filter_ax=fig.add_subplot(grid[0,1])
for b,color in zip(bank,['#8b67af','#4e8cad','#59a079','#c3914c','#bd7068','#986b83']):
    filter_ax.plot(b.wavelength,b.response,color=color,lw=1.5)
    peak=np.argmax(b.response)
    filter_ax.text(b.wavelength[peak],b.response[peak]+.045,b.name[-1],ha='center',fontsize=11,color=color)
filter_ax.set_xlim(*limits); filter_ax.set_ylim(0,max(b.response.max() for b in bank)*1.27)
filter_ax.set_ylabel('LSST\nthroughput'); filter_ax.set_xticks([]); filter_ax.set_yticks([])
filter_ax.spines[['top','right','bottom','left']].set_visible(False)
records=[]
for rownum,row in enumerate(rows,1):
    w=d['wave_rest_aa']; f=d['flux'][row]; iv=d['relative_ivar'][row]; z=float(d['redshift'][row]); obs=w*(1+z)
    tf,ti=trim_spectrum_edges(w,f,iv,z); measured=ti>0
    mw=w[measured]; mf=tf[measured]; mi=ti[measured]
    blue,red=mw[[0,-1]]*(1+z)
    tails={'blue':np.linspace(limits[0]/(1+z),mw[0],180),'red':np.linspace(mw[-1],limits[1]/(1+z),180)}
    # Dense tail evaluation uses the package implementations, preserving original data centers.
    ew=np.unique(np.r_[w,tails['blue'][:-1],tails['red'][1:]])
    at=np.searchsorted(ew,w); ef=np.zeros(len(ew)); ei=np.zeros(len(ew)); ef[at]=tf; ei[at]=ti
    curves={side:{} for side in tails}
    for method in ['constant','powerlaw']:
        extended,_=extrapolate_spectrum_edges(ew,ef,ei,method=method)
        for side,target in tails.items():
            mask=target<mw[0] if side=='blue' else target>mw[-1]
            target_idx=np.searchsorted(ew,target[mask])
            p=extended[target_idx]
            # Draw the fitted continuum up to the boundary, separately from the
            # measured endpoint. Do not create artificial connectors to its noise.
            if method=='powerlaw':
                edge=mw[0] if side=='blue' else mw[-1]
                window=np.abs(mw-edge)<=500
                p=_extrapolate_powerlaw(mw[window],mf[window],mi[window],edge,target)
            elif side=='blue': p=np.r_[p,p[-1]]
            else: p=np.r_[p[0],p]
            curves[side][method]=p
    old=np.flatnonzero(iv>0); new=np.flatnonzero(measured)
    records.append(dict(row=row,targetid=int(d['targetid'][row]),redshift=z,blue_trim_observed_aa=float((w[new[0]]-w[old[0]])*(1+z)),red_trim_observed_aa=float((w[old[-1]]-w[new[-1]])*(1+z))))
    for col,(low,high) in enumerate([(limits[0],blue+420),(limits[0],limits[1]),(red-420,limits[1])]):
        ax=fig.add_subplot(grid[rownum,col])
        ax.axvspan(limits[0],blue,color='#7e90a2',alpha=.08)
        ax.axvspan(red,limits[1],color='#7e90a2',alpha=.08)
        ax.plot(obs,np.where(measured,f,np.nan),color='#8a949e',lw=.65,alpha=.9,label='Measured DESI')
        ax.axhline(0,color='#aeb4bb',lw=.6,zorder=0)
        for edge in [blue,red]: ax.axvline(edge,color='#6c7782',ls=':',lw=.8)
        plotted=[f[measured&(obs>=low)&(obs<=high)]]
        for side,target in tails.items():
            x=target*(1+z); visible=(x>=low)&(x<=high)
            for method,(color,label) in styles.items():
                ax.plot(x,curves[side][method],color=color,lw=2,ls='-',label=label if side=='blue' else None)
                plotted.append(curves[side][method][visible])
        values=np.concatenate(plotted); pad=.09*max(np.ptp(values),.1)
        ax.set_ylim(values.min()-pad,values.max()+pad); ax.set_xlim(low,high)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.14)
        ax.tick_params(axis='both',labelsize=9)
        if col==0: ax.set_ylabel(f'z = {z:.3f}\nNormalized flux')
        if rownum==1: ax.set_title(['Blue edge','Full LSST wavelength range','Red edge'][col],pad=10)
        if rownum==5: ax.set_xlabel('Observed wavelength [Å]')
        if col==1:
            ax.text(.02,.94,f'TARGETID {int(d["targetid"][row])}',transform=ax.transAxes,va='top',fontsize=8,color='#4b5661',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        if rownum==1 and col==1: handles,labels=ax.get_legend_handles_labels()
fig.suptitle('DESI spectra extended to LSST ugrizy',fontsize=20,y=.987)
fig.text(.5,.958,'Constant and power-law continuation · five real galaxies · 5% edge-quality cut',ha='center',fontsize=11)
fig.legend(handles,labels,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,.944))
fig.text(.5,.016,'Shaded regions are extrapolated. Side panels enlarge the edges; each panel includes all values in its wavelength range. Window sizes are rest-frame Å.',ha='center',fontsize=9)
fig.subplots_adjust(left=.075,right=.985,top=.897,bottom=.065)
fig.savefig(OUT/'five-desi-lsst-extrapolation-methods.png',dpi=175)
fig.savefig(OUT/'five-desi-lsst-extrapolation-methods.pdf')
(OUT/'five-desi-lsst-extrapolation-provenance.json').write_text(json.dumps(dict(selection='Five previously shown examples, spaced across redshift, not selected on extrapolation results',quality_cut='production trim_spectrum_edges: 5% local median ivar and three consecutive bins',lsst_observed_bounds_aa=limits.tolist(),holdout=False,methods=['constant','powerlaw'],constant_window_rest_aa=100,powerlaw_window_rest_aa=500,examples=records),indent=2,allow_nan=False))
print('Saved PNG, PDF, and provenance for five spectra.',flush=True)
