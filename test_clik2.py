import eval_clik
import numpy as np
from camb import CAMBparams, get_results

def compute_log_likelihood(ombh2, h, A_planck=1.0, n_samples=100):
    # Initialize the Commander likelihood
    clik_path = "/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"
    lkl = eval_clik.eval_clik(clik_path)

    # Get the lmax
    lmax = lkl.get_lmax()[0]
    n_nuisance = len(lkl.get_extra_parameter_names())

    # Replace the CAMB parameter initialization section
    pars = CAMBparams()
    pars.set_cosmology(H0=h * 100, ombh2=ombh2, omch2=0.120, mnu=0.06, omk=0, tau=0.054)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)

    # Make sure to disable tensors and use a realistic lmax
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantCls = True
    pars.WantScalars = True
    pars.WantTensors = False

    # Generate the power spectrum
    results = get_results(pars)
    cl_dict = results.get_cmb_power_spectra(pars, lmax=lmax, raw_cl=False)

    # Extract the TT spectrum directly
    cl_tt = cl_dict['total'][:lmax + 1, 0] * 1e12  # Apply 1e12 scaling for correct amplitude
    print("Raw TT Cls (First 10):", cl_tt[:10])

    # Sanity check: Ensure no negative values
    if np.any(cl_tt < 0):
        print("Error: Negative Cl values detected")
        return -np.inf

    # Sanity check: Ensure the spectrum is within a reasonable range
    if np.max(cl_tt) > 1e5 or np.min(cl_tt) < 1e-6:
        print("Error: Cl values are out of expected range")
        return -np.inf

    # Apply a minimum floor to avoid numerical underflow
    cls = np.maximum(cl_tt, 1e-6)

    # Apply A_planck scaling
    params = np.zeros(lmax + 1 + n_nuisance)
    params[:lmax + 1] = cls * A_planck
    params[-1] = A_planck

    # Final sanity check on the full parameter vector
    print("Final Spectrum Sent to Commander (First 10):", params[:10])
    if np.any(np.isnan(params)) or np.any(params == 0):
        print("Error: NaNs or zero values in the parameter vector")
        return -np.inf

    # Compute the log-likelihood
    logl = lkl(params)
    return logl

# Test the function
print(compute_log_likelihood(0.022, 0.67, A_planck=1.0))