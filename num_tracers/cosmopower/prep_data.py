import traceback
import camb
import numpy as np
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from cosmopower import cosmopower_NN
from train_cp import cosmopower_train

def generate_latin_hypercube(bounds, n_samples):
    keys = list(bounds.keys())
    lower = np.array([bounds[k][0] for k in keys])
    upper = np.array([bounds[k][1] for k in keys])
    sampler = qmc.LatinHypercube(d=len(keys))
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample, lower, upper)
    param_list = []
    for row in scaled:
        param_dict = {k: v for k, v in zip(keys, row)}
        param_list.append(param_dict)
    return param_list

def get_camb_cls(params, lmax=2507):
    """
    Computes TT, EE, and TE CMB angular power spectra from CAMB.
    
    Parameters:
        params (dict): dictionary of cosmological parameters
        lmax (int): maximum multipole to compute Cls up to
        
    Returns:
        dict: with keys 'ell', 'cl_tt', 'cl_ee', 'cl_te', all in μK²
    """
    # Unpack parameters
    ombh2 = params['ombh2']
    omch2 = params['omch2']
    ns = params['ns']
    logA = params['logA']
    tau = params['tau']
    mnu = params.get('mnu', 0.06)
    nnu = params.get('nnu', 3.044)
    w = params.get('w', -1.0)
    wa = params.get('wa', 0.0)
    theta_MC_100 = params['theta_MC_100']
    
    # Convert logA to As
    As = 1e-10 * np.exp(logA)
    cosmomc_theta = 1e-2 * theta_MC_100

    try:
        cp = camb.CAMBparams()
        cp.set_cosmology(
            ombh2=ombh2,
            omch2=omch2,
            tau=tau,
            mnu=mnu,
            nnu=nnu,
            cosmomc_theta=cosmomc_theta,
            theta_H0_range=[20, 100],
            num_massive_neutrinos=1
        )
        cp.set_dark_energy(w=w, wa=wa)
        cp.InitPower.set_params(As=As, ns=ns)
        cp.set_for_lmax(lmax, lens_potential_accuracy=0)
        
        results = camb.get_results(cp)
    except Exception as e:
        raise e
    powers = results.get_cmb_power_spectra(cp, CMB_unit='muK', lmax=2508)
    cl = powers['total']
    cl_tt = cl[:, 0]
    cl_ee = cl[:, 1]
    cl_te = cl[:, 3]
    ell = np.arange(cl.shape[0])

    return {
        'ell': ell,
        'cl_tt': cl_tt,
        'cl_ee': cl_ee,
        'cl_te': cl_te
    }


def prepare_data(cls_data, param_samples, prior_bounds, test_size=0.2, save_path='.', spectrum='TT', log=False):

    print("Preparing data for", spectrum)
    print("Logarithmic transformation:", log)
    if spectrum == 'TT':
        if log:
            cl = np.log10(np.array([c['cl_tt'][2:] for c in cls_data]))
        else:
            cl = np.array([c['cl_tt'][2:] for c in cls_data])
        ell_modes = cls_data[0]['ell'][2:]
    elif spectrum == 'EE':
        if log:
            cl = np.log10(np.array([c['cl_ee'][2:1997] for c in cls_data]))
        else:
            cl = np.array([c['cl_ee'][2:1997] for c in cls_data])
        ell_modes = cls_data[0]['ell'][2:1997]
    elif spectrum == 'TE':
        if log:
            cl = np.log10(np.array([c['cl_te'][2:1997] for c in cls_data]))
        else:
            cl = np.array([c['cl_te'][2:1997] for c in cls_data])
        ell_modes = cls_data[0]['ell'][2:1997]

    params = np.array([[p[k] for k in prior_bounds.keys()] for p in param_samples])

    cl_train, cl_test, params_train, params_test = train_test_split(cl, params, test_size=test_size, random_state=1)
    if log:
        np.savez(f"{save_path}/cmb_log_{spectrum}_train.npz", modes=ell_modes, features=cl_train)
        np.savez(f"{save_path}/cmb_log_{spectrum}_test.npz", modes=ell_modes, features=cl_test)
    else:
        np.savez(f"{save_path}/cmb_{spectrum}_train.npz", modes=ell_modes, features=cl_train)
        np.savez(f"{save_path}/cmb_{spectrum}_test.npz", modes=ell_modes, features=cl_test)

    param_names = list(prior_bounds.keys())
    params_train_dict = {param: params_train[:, i] for i, param in enumerate(param_names)}
    params_test_dict = {param: params_test[:, i] for i, param in enumerate(param_names)}
    np.savez(f"{save_path}/cmb_params_train.npz", **params_train_dict)
    np.savez(f"{save_path}/cmb_params_test.npz", **params_test_dict)
    
if __name__ == "__main__":
    prior_bounds = {
        "ombh2": (0.005, 0.1),
        "omch2": (0.001, 0.99),
        "ns": (0.8, 1.2),
        "logA": (1.61, 3.91),
        "tau": (0.01, 0.8),
        "nnu": (0.05, 10.0),
        "theta_MC_100": (0.5, 10.0)
    }

    n_samples = 10000
    param_samples = []
    cls_data = []
    total_attempts = 0
    batch_size = 100
    dim = len(prior_bounds)
    seed = 0

    while len(param_samples) < n_samples:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        sample = sampler.random(n=batch_size)
        scaled = qmc.scale(sample,
                           [prior_bounds[k][0] for k in prior_bounds],
                           [prior_bounds[k][1] for k in prior_bounds])
        for row in scaled:
            param = {k: v for k, v in zip(prior_bounds.keys(), row)}
            try:
                cls = get_camb_cls(param, lmax=2508)
                cls_data.append(cls)
                param_samples.append(param)
            except Exception:
                pass
            total_attempts += 1
            if len(param_samples) >= n_samples:
                break
        #print(f"Successfully generated {len(param_samples)} samples, failed {total_attempts - len(param_samples)} times")
        seed += 1  # Ensure new LHS draws differ
    save_path = '/home/ashandonay/data/cosmopower'
    for s in ['TT', 'EE', 'TE']:
        # Only the accepted samples are included; samples and cls_data are already matched
        prepare_data(cls_data, param_samples, prior_bounds, test_size=0.2, save_path=save_path, spectrum=s)
    
    lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    for s in ['TT', 'EE', 'TE']:
        cosmopower_train(spectrum=s, epochs=2000, save_path=save_path, lrs=lrs, batch_size=100)
