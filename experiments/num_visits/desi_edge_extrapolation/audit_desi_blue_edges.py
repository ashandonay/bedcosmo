from pathlib import Path
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from bedcosmo.num_visits.empirical.desi.training_matrix import bin_rest_frame_spectrum
from bedcosmo.num_visits.empirical.desi_data import get_local_desi_paths

out=Path(__file__).parent
root=Path('/home/ashandonay/scratch/bedcosmo/desi/tiny_dr1')
with np.load('/home/ashandonay/scratch/bedcosmo/num_visits/desi_training_data/desi_rest_frame_training_matrix.npz') as saved:
    d={k:saved[k] for k in saved.files}
w=d['wave_rest_aa']; step=np.median(np.diff(w)); records=[]; pixels=[]
for row in [8235,8357,3302,5285,11428,7666]:
    target=int(d['targetid'][row]); z=float(d['redshift'][row]); hp=int(d['healpix'][row])
    path,_=get_local_desi_paths(root,'iron','main','dark',hp)
    with fits.open(path,memmap=True) as hdul:
        ids=hdul['FIBERMAP'].data['TARGETID']; match=np.flatnonzero(ids==target)
        assert len(match)==1
        idx=int(match[0])
        nw=np.concatenate([np.array(hdul[f'{arm}_WAVELENGTH'].data,dtype=float) for arm in 'BRZ'])
        nf=np.concatenate([np.array(hdul[f'{arm}_FLUX'].data[idx],dtype=float) for arm in 'BRZ'])
        ni=np.concatenate([np.array(hdul[f'{arm}_IVAR'].data[idx],dtype=float) for arm in 'BRZ'])
        nm=np.concatenate([np.array(hdul[f'{arm}_MASK'].data[idx]) for arm in 'BRZ'])
    rebinned,weight,scale=bin_rest_frame_spectrum(nw,nf,ni,nm,z,w)
    measured=d['relative_ivar'][row]>0
    max_flux_diff=float(np.max(np.abs(rebinned[measured]-d['flux'][row,measured])))
    max_weight_diff=float(np.max(np.abs(weight[measured]-d['relative_ivar'][row,measured])))
    first=np.flatnonzero(measured)[0]
    good=np.isfinite(nf)&np.isfinite(ni)&(ni>0)&(nm==0)&(nw/(1+z)>=w[0])&(nw/(1+z)<=w[-1])
    index=np.rint((nw/(1+z)-w[0])/step).astype(int)
    selected=good&(index==first)
    direct_mean=float(np.average(nf[selected],weights=ni[selected])/scale)
    sigma=float(np.sqrt(1/ni[selected].sum())/scale)
    rec=dict(row=row,targetid=target,z=z,coadd=str(path),scale=float(scale),saved_scale=float(d['normalization_scale'][row]),max_measured_flux_difference=max_flux_diff,max_measured_weight_difference=max_weight_diff,first_bin_rest_aa=float(w[first]),first_bin_observed_center_aa=float(w[first]*(1+z)),first_bin_saved_flux=float(d['flux'][row,first]),first_bin_recomputed_flux=direct_mean,first_bin_sigma=sigma,native_first_bin_pixel_count=int(selected.sum()),native_first_bin_observed_min=float(nw[selected].min()),native_first_bin_observed_max=float(nw[selected].max()),native_first_bin_flux_min_normalized=float((nf[selected]/scale).min()),native_first_bin_flux_max_normalized=float((nf[selected]/scale).max()),native_first_bin_masks=np.unique(nm[selected]).astype(int).tolist())
    records.append(rec)
    nearby=(nw>=3580)&(nw<=3800)
    for i in np.flatnonzero(nearby):
        pixels.append(dict(row=row,targetid=target,z=z,wave_observed_aa=float(nw[i]),flux_native=float(nf[i]),flux_normalized=float(nf[i]/scale),ivar_native=float(ni[i]),sigma_normalized=float(1/np.sqrt(ni[i])/scale) if ni[i]>0 else None,mask=int(nm[i]),used=bool(good[i]),in_first_bin=bool(selected[i])))
    assert np.allclose(rebinned[measured],d['flux'][row,measured],rtol=1e-6,atol=1e-7)
    assert np.allclose(weight[measured],d['relative_ivar'][row,measured],rtol=1e-6,atol=1e-7)
(out/'desi-blue-edge-audit.json').write_text(json.dumps(records,indent=2,allow_nan=False))
pd.DataFrame(pixels).to_csv(out/'desi-blue-edge-native-pixels.csv',index=False)
print(json.dumps(records,indent=2),flush=True)
