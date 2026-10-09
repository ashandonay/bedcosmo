"""Summarize frozen choices, galaxy bootstrap, and physical endpoint loss."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
P=json.loads((OUT/'edge-tuning-protocol.json').read_text())
S=json.loads((OUT/'edge-tuning-selected.json').read_text())
T=pd.read_csv(OUT/'edge-tuning-test.csv')
V=pd.read_csv(OUT/'edge-tuning-validation-summary.csv')
rng=np.random.default_rng(20261011)
reports={}; cis=[]
for side in ['blue','red']:
    Q=T[T.side==side].pivot(index='row',columns='label',values='band_error')
    ix=rng.integers(0,len(Q),(10000,len(Q)))
    medians={c:np.median(Q[c].to_numpy()[ix],axis=1) for c in Q}
    winner='tuned '+S[side]['winner']; base='no cut constant'
    diff=medians[winner]-medians[base]
    reports[side]=dict(validation_winner=S[side]['winner'],n=len(Q),test_median_band_error={c:float(Q[c].median()) for c in Q},test_median_difference_vs_no_cut_constant=float(Q[winner].median()-Q[base].median()),difference_95_bootstrap_ci=np.quantile(diff,[.025,.975]).tolist(),fraction_galaxies_winner_beats_constant=float((Q[winner]<Q[base]).mean()),trial_cut_changed_test_predictions=int(np.count_nonzero(~np.isclose(Q['trial cut constant'],Q[base],rtol=1e-12,atol=1e-15))),validation_cut_range_by_method={m:[float(V[(V.side==side)&(V.method==m)].median_band_error.min()),float(V[(V.side==side)&(V.method==m)].median_band_error.max())] for m in ['constant','powerlaw','regularized']})
    for b in range(4):
        q=T[(T.side==side)&(T.zbin==b)].pivot(index='row',columns='label',values='band_error')
        bix=rng.integers(0,len(q),(10000,len(q)))
        for method in ['constant','powerlaw','regularized']:
            label='tuned '+method; values=q[label].to_numpy()
            low,high=np.quantile(np.median(values[bix],axis=1),[.025,.975])
            cis.append(dict(side=side,zbin=b,method=method,n=len(q),median=float(np.median(values)),low=float(low),high=float(high)))
(OUT/'edge-tuning-test-inference.json').write_text(json.dumps(reports,indent=2))
C=pd.DataFrame(cis); C.to_csv(OUT/'edge-tuning-test-confidence.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,5.1))
styles={'constant':('#286bb5','Constant, 100 Å'),'powerlaw':('#d87915','Power law, 500 Å'),'regularized':('#823bb2','Regularized slope')}
for ax,side,band,horizon in zip(axes,['blue','red'],['g','z'],[600,800]):
    for offset,(method,(color,label)) in zip([-.15,0,.15],styles.items()):
        q=C[(C.side==side)&(C.method==method)].sort_values('zbin')
        y=q['median'].to_numpy()*100; low=q.low.to_numpy()*100; high=q.high.to_numpy()*100
        ax.errorbar(np.arange(4)+offset,y,yerr=np.array([y-low,high-y]),fmt='o-',color=color,lw=1.7,ms=5,capsize=4,label=label)
    ax.set_xticks(range(4),['0–0.6','0.6–0.9','0.9–1.2','1.2–2.0'])
    ax.set_xlabel('Galaxy redshift bin (100 test galaxies each)')
    ax.set_ylabel(f'Median absolute LSST {band}-flux error [%]')
    ax.set_title(f'{side.capitalize()} edge: hide {horizon} observed Å',loc='left',fontsize=13)
    ax.set_ylim(bottom=0); ax.grid(alpha=.17); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Extrapolation on 400 separate test galaxies',fontsize=17)
fig.text(.5,.905,'Quality-cut settings selected on 400 validation galaxies; whiskers: 95% galaxy-bootstrap intervals',ha='center',fontsize=10)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.025))
fig.tight_layout(rect=(0,.1,1,.88))
fig.savefig(OUT/'edge-tuning-test-redshift.png',dpi=170)
# Physical measured endpoints: apply the same grid without a wavelength holdout.
source=(OUT/'tune_edge_quality.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='endpoint')
exec(compile(ast.get_source_segment(source,node),'endpoint','exec'))
d0=np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz')
w=d0['wave_rest_aa']; iv=d0['relative_ivar']; z=d0['redshift']; d0.close()
records=[]
for split in ['validation','test']:
    for item in P['splits'][split]:
        row=item['row']; measured=iv[row]>0
        for side in ['blue','red']:
            x=w[measured]; v=iv[row,measured]
            if side=='blue': x=-x[::-1]; v=v[::-1]
            for threshold,run in P['cuts']:
                k=endpoint(x,v,z[row],threshold,run)
                records.append(dict(split=split,row=row,side=side,threshold=threshold,run=run,lost_observed_aa=float((x[-1]-x[k])*(1+z[row]))))
E=pd.DataFrame(records); E.to_csv(OUT/'edge-tuning-native-trimming.csv',index=False)
A=E.groupby(['split','side','threshold','run'],as_index=False).agg(n=('row','size'),fraction_moved=('lost_observed_aa',lambda x:(x>0).mean()),median_lost_observed_aa=('lost_observed_aa','median'),p90_lost_observed_aa=('lost_observed_aa',lambda x:x.quantile(.9)),max_lost_observed_aa=('lost_observed_aa','max'))
A.to_csv(OUT/'edge-tuning-native-trimming-summary.csv',index=False)
print(json.dumps(reports,indent=2))
print('Trial physical-endpoint trimming on separate test galaxies:')
print(A[(A.split=='test')&(A.threshold==.1)&(A.run==3)].to_string(index=False))
assert len(T)==400*2*5
assert (C.n==100).all()
assert (T.failure.fillna('')=='').all()
assert T.groupby(['side','label']).row.nunique().eq(400).all()
print('Result integrity checks passed; plot saved.')
