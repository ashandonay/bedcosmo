from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bedcosmo.num_visits.empirical.desi.training_matrix import extrapolate_spectrum_edges, _extrapolate_powerlaw

out=Path(__file__).parent
d=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w=d['wave_rest_aa']
fig,axs=plt.subplots(3,1,figsize=(12,12))
records=[]
for ax,row in zip(axs,[5230,5684,8955]):
    f=d['flux'][row]; iv=d['relative_ivar'][row]; z=float(d['redshift'][row]); measured=iv>0
    end=w[measured][-1]
    hidden=measured & (w>=end-800/(1+z))
    train=iv.copy(); train[hidden]=0
    edge=w[train>0][-1]; boundary=edge+5
    visible=measured & (w>=edge-600)
    fit=(train>0)&(w>=edge-500)
    const,_=extrapolate_spectrum_edges(w,f,train,method='constant')
    power,_=extrapolate_spectrum_edges(w,f,train,method='powerlaw')
    truth=np.average(f[hidden],weights=iv[hidden])
    ec=abs(np.average(const[hidden],weights=iv[hidden])-truth)/abs(truth)
    ep=abs(np.average(power[hidden],weights=iv[hidden])-truth)/abs(truth)
    ax.axvspan(edge-500,boundary,color='#ed8a23',alpha=.08)
    ax.axvspan(edge-100,boundary,color='#3475c5',alpha=.12)
    ax.axvspan(boundary,end+5,color='#299568',alpha=.09)
    ax.plot(w[visible],f[visible],color='#a8adb4',lw=.7,alpha=.55,label='Measured spectrum (raw)')
    # Weighted 50-Angstrom bins show the continuum without hiding the raw data.
    for mask,color,label in [(visible&~hidden,'#414852','Available data, 50 Å bins'),(hidden,'#16845c','Withheld data, 50 Å bins')]:
        xx=[]; yy=[]; se=[]
        for lo in np.arange(edge-600,end+50,50):
            m=mask&(w>=lo)&(w<lo+50)
            if m.any():
                xx.append(np.average(w[m],weights=iv[m])); yy.append(np.average(f[m],weights=iv[m])); se.append(np.sqrt(1/iv[m].sum()))
        ax.errorbar(xx,yy,yerr=se,color=color,fmt='o-',ms=4,lw=1.1,capsize=2,label=label,zorder=4)
    ax.plot(w[fit],_extrapolate_powerlaw(w[fit],f[fit],iv[fit],edge,w[fit]),color='#d87915',lw=2,label='Power-law fit: last 500 Å')
    ax.plot([edge-100,edge],[const[hidden][0]]*2,color='#286bb5',lw=2,label='Constant fit: last 100 Å')
    ax.plot(np.r_[edge,w[hidden]],np.r_[const[hidden][0],const[hidden]],'--',color='#286bb5',lw=2.5,label='Constant prediction')
    ax.plot(np.r_[edge,w[hidden]],np.r_[_extrapolate_powerlaw(w[fit],f[fit],iv[fit],edge,np.array([edge])),power[hidden]],'--',color='#d87915',lw=2.5,label='Power-law prediction')
    ax.axvline(boundary,color='#545c66',ls=':',lw=1.5)
    ax.text(.98,.96,f'Mean-flux error: constant {ec:.1%}  |  power law {ep:.1%}',transform=ax.transAxes,ha='right',va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.9),fontsize=11)
    ax.set_title(f'{"Constant wins" if ec<ep else "Power law wins"}  •  z = {z:.3f}  •  DESI target {int(d["targetid"][row])}',loc='left',fontsize=12,pad=10)
    ax.set_xlim(edge-600,end+15)
    ax.set_xlabel('Rest-frame wavelength [Å]'); ax.set_ylabel('Normalized flux density')
    ax.grid(alpha=.14); ax.spines[['top','right']].set_visible(False)
    records.append(dict(row=row,targetid=int(d['targetid'][row]),redshift=z,constant_error=ec,powerlaw_error=ep))
fig.suptitle('Predicting a withheld red edge of real DESI spectra',fontsize=17,y=.985)
fig.text(.5,.952,'Green region: last 800 Å in the observed frame, excluded from both fits. Dashed curves are predictions.',ha='center',fontsize=11)
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=10,frameon=False,bbox_to_anchor=(.5,.012))
fig.tight_layout(rect=(0,.085,1,.945),h_pad=2)
fig.savefig(out/'red-edge-heldout-examples.png',dpi=170)
import json
(out/'red-edge-heldout-examples.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
