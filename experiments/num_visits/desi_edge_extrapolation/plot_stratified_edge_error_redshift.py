from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).parent
t=pd.read_csv(out/'stratified-edge-errors.csv')
methods=['constant','powerlaw','regularized']
p=t[t.method.isin(methods)].pivot(index='row',columns='method',values='mean_error').dropna()
z=t.groupby('row').z.first().loc[p.index]
bins=[0,.6,.9,1.2,2]
assert z.min()>=bins[0] and z.max()<bins[-1]
assert len(p)==400
rng=np.random.default_rng(1108)
colors={'constant':'#286bb5','powerlaw':'#d87915','regularized':'#823bb2'}
names={'constant':'Constant, 100 Å','powerlaw':'Power law, 500 Å','regularized':'Regularized local slope'}
records=[]; labels=[]
for lo,hi in zip(bins[:-1],bins[1:]):
    sub=p[(z>=lo)&(z<hi)]; n=len(sub)
    assert n==100
    labels.append(f'{lo:g} ≤ z < {hi:g}\n(n = {n})')
    # Resample galaxies together across methods to preserve their pairing.
    bootstrap=rng.integers(0,n,size=(10000,n))
    for method in methods:
        values=sub[method].to_numpy()
        median=float(np.median(values))
        lower,upper=np.quantile(np.median(values[bootstrap],axis=1),[.025,.975])
        assert 0<=lower<=median<=upper
        records.append(dict(z_min=lo,z_max=hi,n=n,method=method,median_error=median,median_bootstrap_low=float(lower),median_bootstrap_high=float(upper)))
fig,ax=plt.subplots(figsize=(11,6.2))
for method,shift in zip(methods,[-.07,0,.07]):
    recs=[r for r in records if r['method']==method]
    med=100*np.array([r['median_error'] for r in recs])
    lower=100*np.array([r['median_bootstrap_low'] for r in recs])
    upper=100*np.array([r['median_bootstrap_high'] for r in recs])
    ax.errorbar(np.arange(4)+shift,med,yerr=np.array([med-lower,upper-med]),fmt='o-',color=colors[method],lw=1.8,capsize=5,label=names[method])
ax.set_ylabel('Median absolute mean-flux error [%]')
ax.legend(frameon=False,ncol=3,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.01))
ax.set_ylim(bottom=0); ax.set_xticks(range(4),labels); ax.set_xlabel('Redshift bin')
ax.grid(alpha=.15); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Red-edge extrapolation error by redshift',fontsize=16,y=.985)
fig.text(.5,.925,'Whiskers: 95% bootstrap intervals for the median, from resampling galaxies within each bin.',ha='center',fontsize=10)
fig.text(.5,.025,'400 galaxies, randomly sampled within redshift bins; last 800 observed-frame Å withheld.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,.9))
fig.savefig(out/'stratified-edge-error-versus-redshift.png',dpi=170)
pd.DataFrame(records).to_csv(out/'stratified-edge-error-versus-redshift.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
