from pathlib import Path
import json
import numpy as np
import pandas as pd

out=Path(__file__).parent
with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    d={k:saved[k] for k in ['wave_rest_aa','redshift','flux','relative_ivar','targetid']}
w=d['wave_rest_aa']; thresholds=[.01,.05,.1,.2]; examples=[8235,8357,3302,5285,11428,7666]
records=[]
def usable_endpoint(w,iv,z,side,threshold):
    indices=np.flatnonzero(iv>0)
    if side=='red': indices=indices[::-1]
    edge=indices[0]
    distance=np.abs(w-w[edge])*(1+z)
    # Reference ignores the first 50 observed Å to avoid partially filled edge bins.
    ref=(iv>0)&(distance>=50)&(distance<=300)
    if ref.sum()<3:
        raise ValueError('Insufficient local reference coverage')
    scale=np.median(iv[ref])
    adequate=(iv>=threshold*scale)&(iv>0)
    direction=1 if side=='blue' else -1
    # Require three consecutive grid bins passing the weight threshold.
    for idx in indices:
        run=idx+direction*np.arange(3)
        if run.min()<0 or run.max()>=len(w): continue
        if adequate[run].all():
            return int(idx),float(scale)
    raise ValueError('No supported endpoint')
for row,z in enumerate(d['redshift']):
    iv=d['relative_ivar'][row]; f=d['flux'][row]; measured=np.flatnonzero(iv>0)
    for side in ['blue','red']:
        old=int(measured[0 if side=='blue' else -1])
        for threshold in thresholds:
            idx,ref=usable_endpoint(w,iv,z,side,threshold)
            records.append(dict(row=row,targetid=int(d['targetid'][row]),z=float(z),side=side,threshold=threshold,old_index=old,new_index=idx,original_edge_weight_ratio=float(iv[old]/ref),coverage_lost_observed_aa=float(abs(w[idx]-w[old])*(1+z)),original_flux=float(f[old]),original_sigma=float(1/np.sqrt(iv[old])),new_flux=float(f[idx]),new_sigma=float(1/np.sqrt(iv[idx]))))
t=pd.DataFrame(records)
t.to_csv(out/'edge-ivar-threshold-diagnostic.csv',index=False)
summary=t.groupby(['side','threshold']).agg(n=('row','size'),fraction_moved=('coverage_lost_observed_aa',lambda x:float((x>0).mean())),median_lost_observed_aa=('coverage_lost_observed_aa','median'),p90_lost_observed_aa=('coverage_lost_observed_aa',lambda x:float(x.quantile(.9))),max_lost_observed_aa=('coverage_lost_observed_aa','max')).reset_index()
summary.to_csv(out/'edge-ivar-threshold-summary.csv',index=False)
print('ALL GALAXIES:',summary.to_string(index=False))
selected=t[(t.row.isin(examples))&(t.threshold==.1)]
print('SIX EXAMPLES, threshold 10%:',selected[['row','z','side','original_edge_weight_ratio','coverage_lost_observed_aa','original_flux','original_sigma','new_flux','new_sigma']].to_string(index=False))
selected.to_csv(out/'edge-ivar-six-examples.csv',index=False)
(out/'edge-ivar-threshold-protocol.json').write_text(json.dumps(dict(reference='Median positive inverse variance 50–300 observed Angstroms inward from original edge',thresholds=thresholds,edge_requirement='Three consecutive 10-rest-Angstrom grid bins passing threshold; discard bins exterior to that endpoint',scope='Existing saved matrix only; no production changes',n_spectra=len(d['redshift'])),indent=2))
