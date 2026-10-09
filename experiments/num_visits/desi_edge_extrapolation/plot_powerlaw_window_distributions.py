"""Show the bulk and complete tails separately when narrow fits explode."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
OUT=Path(__file__).parent
T=pd.read_csv(OUT/'powerlaw-window-errors.csv'); C=pd.read_csv(OUT/'last-100-edge-errors.csv'); C=C[C.method=='constant']; T=pd.concat([T,C],ignore_index=True)
styles={'constant':('#68737f','Constant, 100 Å'),'powerlaw100':('#269679','Power law, 100 Å'),'powerlaw200':('#286bb5','Power law, 200 Å'),'powerlaw300':('#823bb2','Power law, 300 Å'),'powerlaw500':('#d87915','Power law, 500 Å')}
fig,axes=plt.subplots(3,2,figsize=(12,11.4))
for col,side in enumerate(['blue','red']):
    frame=T[T.side==side]; values_all=frame.absolute_mean_error.to_numpy()
    low=np.nextafter(values_all[values_all>0].min(),0)
    reference=frame[frame.method.isin(['constant','powerlaw300','powerlaw500'])]
    focus=max(reference.groupby('method').absolute_mean_error.quantile(.99))*1.3
    bins=np.geomspace(low,focus,29)
    overflow=[]
    for method,(color,label) in styles.items():
        values=frame[frame.method==method].absolute_mean_error.to_numpy()
        axes[0,col].hist(values,bins=bins,weights=np.full(len(values),100/len(values)),histtype='step',lw=1.6,color=color,label=label)
        x,counts=np.unique(np.sort(values),return_counts=True); prob=np.cumsum(counts[::-1])[::-1]*100/len(values)
        for ax in axes[1:,col]:
            ax.step(x,prob,where='pre',lw=1.8,color=color)
        axes[2,col].plot(x[-1],prob[-1],'.',ms=5,color=color)
        overflow.append(f'{label.replace("Power law, ","")}: {np.sum(values>focus)}/400')
    axes[0,col].set_title(f'{side.capitalize()} edge · main distribution',loc='left',fontsize=12)
    axes[0,col].set_ylabel('Galaxies per log bin [%]')
    axes[0,col].text(.02,.95,'Beyond this view:\n'+'\n'.join(overflow),transform=axes[0,col].transAxes,va='top',fontsize=8.5,color='#4b5661')
    for row in [0,1]:
        axes[row,col].set_xscale('log'); axes[row,col].set_xlim(low/1.1,focus)
    for row in [1,2]:
        axes[row,col].set_yscale('log'); axes[row,col].set_ylim(.2,110)
        axes[row,col].set_yticks([.25,1,5,10,50,100]); axes[row,col].yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
        axes[row,col].set_ylabel('Galaxies with error ≥ x [%]')
        axes[row,col].axhline(1,color='#9ba4ae',ls=':',lw=.7)
    axes[1,col].set_title('Tail probabilities in the same main range',loc='left',fontsize=11)
    axes[1,col].set_xlabel('Absolute error in withheld mean flux [normalized units]')
    axes[2,col].set_xscale('log'); axes[2,col].set_xlim(.1,values_all.max()*5)
    axes[2,col].set_title('Full tail — every extreme value included',loc='left',fontsize=11)
    axes[2,col].set_xlabel('Absolute error [normalized units; full logarithmic range]')
    axes[2,col].set_xticks([1,1e10,1e20,1e30,1e40,1e50])
    for ax in axes[:,col]:
        ax.grid(axis='y',alpha=.15); ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=9)
fig.suptitle('Shorter power-law windows: distributions and extreme tails',fontsize=17,y=.995)
fig.text(.5,.962,'400 galaxies · last 100 observed Å withheld · raw signed flux, ivar weighting, robust loss · fit windows in rest-frame Å',ha='center',fontsize=10)
fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,.944),fontsize=10)
fig.text(.5,.013,'Upper two rows enlarge the main range; counts list errors outside it. Bottom row includes the full range. One galaxy = 0.25%; reference flux contains noise.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.88),h_pad=2.2,w_pad=2)
fig.savefig(OUT/'powerlaw-window-error-distributions.png',dpi=180)
fig.savefig(OUT/'powerlaw-window-error-distributions.pdf')
print('Saved focused distributions and full-range tails.')
