"""Plot archived quality-cut scores and actual DESI endpoint trimming."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
OUT=Path(__file__).parent
DATA=OUT
V=pd.read_csv(DATA/'edge-tuning-validation-summary.csv')
E=pd.read_csv(DATA/'edge-tuning-native-trimming-summary.csv')
E=E[E.split=='test']
cuts=[(0.,1)]+[(t,n) for t in [.05,.1,.2] for n in [1,3,5]]
labels=['No\ncut']+[f'{100*t:.0f}%\n{n} bins' for t,n in cuts[1:]]
styles={'constant':('#286bb5','Constant, 100 Å'),'powerlaw':('#d87915','Power law, 500 Å'),'regularized':('#823bb2','Regularized slope')}
fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for ax,side,band,nmoved in zip(axes,['blue','red'],['g','z'],[0,2]):
    for method,(color,label) in styles.items():
        subset=V[(V.side==side)&(V.method==method)].set_index(['threshold','run'])
        values=np.array([subset.loc[c,'median_band_error'] for c in cuts])*100
        ax.plot(range(10),values,'o-',color=color,lw=1.8,ms=4,label=label)
    ax.set_xticks(range(10),labels,fontsize=9)
    ax.set_xlim(-.3,9.3); ax.set_ylim(0,2.8)
    ax.set_title(f'{side.capitalize()} artificial edge → LSST {band}',loc='left',fontsize=12)
    ax.set_xlabel('Ivar threshold / required consecutive bins')
    ax.text(.02,.95,f'At most {nmoved}/400 artificial cutoffs move',transform=ax.transAxes,va='top',fontsize=10)
    ax.grid(axis='y',alpha=.18); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Median absolute reconstructed band-flux error [%]')
fig.suptitle('Quality-cut settings barely affect validation error',fontsize=16)
fig.text(.5,.89,'400 validation galaxies • blue holdout: 600 observed Å • red holdout: 800 observed Å',ha='center',fontsize=10)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0))
fig.tight_layout(rect=(0,.09,1,.87))
fig.savefig(OUT/'quality-cut-validation-comparison.png',dpi=180)

fig,axes=plt.subplots(2,2,figsize=(10,7.5))
for col,side in enumerate(['blue','red']):
    q=E[E.side==side].set_index(['threshold','run'])
    for row,(metric,title,scale,vmax,unit) in enumerate([
        ('fraction_moved','Spectra whose endpoint is trimmed',100,100,'%'),
        ('p90_lost_observed_aa','90th percentile of discarded coverage',1,50,' Å'),
    ]):
        ax=axes[row,col]
        values=np.array([[q.loc[(t,n),metric]*scale for t in [.05,.1,.2]] for n in [1,3,5]])
        im=ax.imshow(values,cmap='Blues',norm=Normalize(0,vmax),aspect='auto')
        for j in range(3):
            for i in range(3):
                value=values[j,i]
                ax.text(i,j,f'{value:.1f}{unit}',ha='center',va='center',fontsize=12,color='white' if value>vmax*.55 else '#172332')
        ax.set_xticks(range(3),['5%','10%','20%']); ax.set_yticks(range(3),['1','3','5'])
        ax.set_xlabel('Threshold relative to nearby median ivar')
        ax.set_ylabel('Required consecutive bins')
        ax.set_title(f'{side.capitalize()} edge: {title}',loc='left',fontsize=11)
        # Identify the trial without hiding other candidate values.
        ax.add_patch(plt.Rectangle((.5,.5),1,1,fill=False,edgecolor='#d87915',lw=2))
        fig.colorbar(im,ax=ax,pad=.025,label='Spectra [%]' if row==0 else 'Observed wavelength [Å]')
fig.suptitle('Quality cuts do change the physical DESI endpoints',fontsize=16)
fig.text(.5,.918,'400 separate test galaxies • outlined cell: trial 10% / 3 bins • no-cut baseline discards 0 Å',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.90),h_pad=2)
fig.savefig(OUT/'quality-cut-native-endpoint-comparison.png',dpi=180)
