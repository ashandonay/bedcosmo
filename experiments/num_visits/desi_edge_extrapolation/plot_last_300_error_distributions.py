"""Empirical distributions of last-300-Angstrom errors, including every galaxy."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,LogLocator,FuncFormatter
OUT=Path(__file__).parent
D=pd.read_csv(OUT/'last-300-edge-errors.csv')
styles={'constant':('#286bb5','Constant'),'powerlaw':('#d87915','Power law')}
D=D[D.method.isin(styles)]
fig,axes=plt.subplots(2,2,figsize=(12,8.6),sharex='col')
quantiles=[]
for col,side in enumerate(['blue','red']):
    frame=D[D.side==side]
    allvalues=frame.absolute_mean_error.to_numpy()
    # Common logarithmic bin edges for all methods within each edge comparison.
    low=np.nextafter(allvalues.min(),0); high=np.nextafter(allvalues.max(),np.inf)
    edges=np.geomspace(low,high,31)
    for method,(color,label) in styles.items():
        values=frame[frame.method==method].absolute_mean_error.to_numpy()
        axes[0,col].hist(values,bins=edges,weights=np.full(len(values),100/len(values)),histtype='step',lw=1.9,color=color,label=label)
        x,counts=np.unique(np.sort(values),return_counts=True)
        probability=np.cumsum(counts[::-1])[::-1]*100/len(values)
        axes[1,col].step(x,probability,where='pre',lw=2,color=color,label=label)
        axes[1,col].plot(x[-1],probability[-1],'.',color=color,ms=5)
        quantiles.append(dict(side=side,method=method,n=len(values),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),p99=float(np.quantile(values,.99)),maximum=float(np.max(values)),fraction_above_1=float(np.mean(values>=1)),fraction_above_5=float(np.mean(values>=5))))
    axes[0,col].set_title(f'{side.capitalize()} edge · 400 galaxies per method',loc='left',fontsize=13)
    axes[0,col].set_ylabel('Galaxies per logarithmic bin [%]')
    axes[1,col].set_ylabel('Galaxies with error ≥ x [%]')
    axes[1,col].set_yscale('log'); axes[1,col].set_ylim(.2,110)
    axes[1,col].set_yticks([.25,1,5,10,50,100]); axes[1,col].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    axes[1,col].set_xlabel('Absolute error in withheld mean flux\n[normalized flux units; logarithmic axis]')
    for ax in axes[:,col]:
        ax.set_xscale('log'); ax.set_xlim(low/1.12,high*1.15)
        ax.grid(axis='y',alpha=.16); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[1,col].axhline(1,color='#99a2ac',ls=':',lw=.8)
    axes[1,col].text(.02,.04,'One galaxy = 0.25%',transform=axes[1,col].transAxes,fontsize=9,color='#64707b')
fig.suptitle('Full error distributions — predict the last 300 observed Å',fontsize=17,y=.99)
fig.text(.5,.948,'All 400 galaxies retained · fixed 5% edge cut · no errors or outliers clipped',ha='center',fontsize=11)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.927),ncol=3,frameon=False,fontsize=11)
fig.text(.5,.015,'Lower tail-probability curves mean fewer large errors. Blue and red use different x ranges; errors include noise in the withheld measurements.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.055,1,.88),h_pad=2.3,w_pad=2)
fig.savefig(OUT/'last-300-edge-error-distributions.png',dpi=180)
fig.savefig(OUT/'last-300-edge-error-distributions.pdf')
Q=pd.DataFrame(quantiles);Q.to_csv(OUT/'last-300-edge-tail-quantiles.csv',index=False)
print(Q.to_string(index=False))
