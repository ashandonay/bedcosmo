import sys
sys.path.insert(0, '/home/ashandonay/cobaya/packages/code/planck/clik-main/lib/python/site-packages')
import clipy as clik
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist

# Placeholder for a CosmoPower-like emulator
# This should take cosmological parameters and return power spectra
def get_cosmopower_spectra(params):
    """
    Placeholder for a cosmopower emulator.
    This should take a dictionary of cosmological parameters and return
    a dictionary of power spectra (Cls).

    For now, it returns fiducial values, ignoring the input params.
    You will need to replace this with your actual emulator.
    """
    # These are just placeholders, corresponding to some fiducial cosmology
    # The shape should be (n_ell,)
    tt_ell = np.load("/home/ashandonay/data/cosmopower/tt_ell.npy")
    dl_tt_fiducial = np.load("/home/ashandonay/data/cosmopower/dl_tt.npy")[0]
    
    # We need C_ls, not D_ls
    cl_tt_fiducial = (dl_tt_fiducial * 2 * np.pi) / (tt_ell * (tt_ell + 1))
    
    # For a real implementation, you would use `params` to compute the spectra
    # e.g., Om = params['Om']
    
    # The output should be a tensor on the correct device
    return torch.from_numpy(cl_tt_fiducial)

class CMBLikelihood:
    def __init__(self, device="cuda:0"):
        self.device = device
        
        # Define priors for the cosmological parameters you want to constrain
        # These should match the parameters your emulator expects
        self.priors = {
            'Om': dist.Uniform(0.1, 0.5), 
            'h': dist.Uniform(0.6, 0.8),
            # Add other parameters like 's8', 'ns', etc. as needed by your emulator
        }
        self.cosmo_params = list(self.priors.keys())
        self.latex_labels = ['\Omega_m', 'h'] # Add corresponding latex labels

        # Load fiducial spectra and create a placeholder covariance matrix
        # For a real scenario, the covariance would be non-diagonal and complex
        tt_ell = np.load("/home/ashandonay/data/cosmopower/tt_ell.npy")
        dl_tt_fiducial = np.load("/home/ashandonay/data/cosmopower/dl_tt.npy")[0]
        cl_tt_fiducial = (dl_tt_fiducial * 2 * np.pi) / (tt_ell * (tt_ell + 1))

        # Placeholder for covariance: assume 1% diagonal relative error on C_ls
        cl_variance = (0.01 * cl_tt_fiducial)**2
        self.covariance = torch.diag(torch.from_numpy(cl_variance)).to(device)
        self.mean_fiducial = torch.from_numpy(cl_tt_fiducial).to(device)
        
        # Define the labels for the data vector and parameters
        self.target_labels = self.cosmo_params

    def pyro_model(self, design):
        # The design parameter is not used for the CMB likelihood's physics, 
        # but its shape determines the batch size for sampling.
        batch_shape = design.shape[:-1]

        # The plate ensures that we can ask for batches of samples
        with pyro.plate_stack("plate", batch_shape):
            # Sample cosmological parameters from their priors
            params = {
                k: pyro.sample(k, v) for k, v in self.priors.items()
            }

            # Get the theoretical power spectrum (mean of the likelihood)
            # from the emulator for the sampled parameters.
            # NOTE: You MUST replace `get_cosmopower_spectra` with your actual emulator.
            mean_cls = get_cosmopower_spectra(params)
            mean_cls = mean_cls.to(self.device)

            # The "observed" data y is a sample from the likelihood distribution.
            # This simulates the outcome of a CMB experiment.
            # The log-probability of this sample is automatically tracked by Pyro.
            return pyro.sample(
                "y",
                dist.MultivariateNormal(mean_cls, self.covariance)
            )

def compute_clik_log_likelihood(A_planck=1.0, n_samples=100):
    # Initialize the Commander likelihood
    tt_lowl_lkl = clik.clik("/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik")
    ee_lowl_lkl = clik.clik("/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik")
    highl_lkl = clik.clik("/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik/")
    #highl_lkl = clik.clik("/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik")
    lmax = tt_lowl_lkl.get_lmax()[0]
    n_nuisance = len(tt_lowl_lkl.get_extra_parameter_names())

    # Define the multipole range for plotting C_l values, which start from l=2
    plot_ell = np.arange(2, lmax + 1)

    tt_ell = np.load("/home/ashandonay/data/cosmopower/tt_ell.npy")
    dl_tt = np.load("/home/ashandonay/data/cosmopower/dl_tt.npy")
    te_ell = np.load("/home/ashandonay/data/cosmopower/te_ell.npy")
    dl_te = np.load("/home/ashandonay/data/cosmopower/dl_te.npy")
    ee_ell = np.load("/home/ashandonay/data/cosmopower/ee_ell.npy")
    dl_ee = np.load("/home/ashandonay/data/cosmopower/dl_ee.npy")
    n_samples = dl_tt.shape[0]
    
    # Convert D_l to C_l for TT, EE, TE
    # C_l = 2 * pi * D_l / (l * (l + 1))
    cl_tt = (dl_tt * 2 * np.pi) / (tt_ell * (tt_ell + 1))
    cl_ee = (dl_ee * 2 * np.pi) / (ee_ell * (ee_ell + 1))
    cl_te = (dl_te * 2 * np.pi) / (te_ell * (te_ell + 1))


    # Prepare params for tt_lowl_lkl (TT only, l=2 to 29)
    lmax_tt_lowl = tt_lowl_lkl.get_lmax()[0]
    n_nuisance_tt_lowl = len(tt_lowl_lkl.get_extra_parameter_names())
    print("TT nuisance parameters:", tt_lowl_lkl.get_extra_parameter_names())
    params_tt_lowl = np.zeros((n_samples, lmax_tt_lowl + 1 + n_nuisance_tt_lowl))
    # TT from l=2 to 29
    tt_indices = np.where(np.arange(2, 30) == tt_ell[:, np.newaxis])[0]
    params_tt_lowl[:, 2:lmax_tt_lowl+1] = cl_tt[:, tt_indices]
    if n_nuisance_tt_lowl > 0:
        params_tt_lowl[:, -n_nuisance_tt_lowl:] = A_planck

    # Prepare params for ee_lowl_lkl (EE only, l=2 to 29)
    lmax_ee_lowl = ee_lowl_lkl.get_lmax()[1]
    n_nuisance_ee_lowl = len(ee_lowl_lkl.get_extra_parameter_names())
    print("EE nuisance parameters:", ee_lowl_lkl.get_extra_parameter_names())
    params_ee_lowl = np.zeros((n_samples, lmax_ee_lowl + 1 + n_nuisance_ee_lowl))  # +1 because we need l=0 to lmax
    # l=0 and l=1 are already set to 0 by np.zeros
    # EE from l=2 to 29
    ee_indices = np.where(np.arange(2, 30) == ee_ell[:, np.newaxis])[0]
    params_ee_lowl[:, 2:lmax_ee_lowl+1] = cl_ee[:, ee_indices]  # Start at index 2 for l=2
    if n_nuisance_ee_lowl > 0:
        params_ee_lowl[:, -n_nuisance_ee_lowl:] = A_planck

    # Prepare params for highl_lkl (TT only, l=0 to lmax)
    lmax_highl = highl_lkl.get_lmax()[0]
    n_nuisance_highl = len(highl_lkl.get_extra_parameter_names())
    
    # Create array with correct shape: (n_samples, lmax_highl + 1 + n_nuisance_highl)
    # The +1 is because we need l=0 to lmax_highl inclusive
    # Values for l<30 are unused by the likelihood but must be present in the array
    params_highl = np.zeros((n_samples, lmax_highl + 1 + n_nuisance_highl))
    
    # Fill in TT values from l=30 to lmax_highl
    tt_highl = np.where(np.arange(30, lmax_highl + 1) == tt_ell[:, np.newaxis])[0]
    # Fill in the values starting at index 30 (which corresponds to l=30)
    # Values for l<30 are unused by the likelihood but must be present in the array
    params_highl[:, 30:lmax_highl+1] = cl_tt[:, tt_highl]
    
    # Add nuisance parameters at the end
    if n_nuisance_highl > 0:
        params_highl[:, -n_nuisance_highl:] = [
            67,        # A_cib_217
            -1.3,      # cib_index
            0,         # xi_sz_cib
            7,         # A_sz
            257,       # ps_A_100_100
            47,        # ps_A_143_143
            40,        # ps_A_143_217
            104,       # ps_A_217_217
            0,         # ksz_norm
            7,         # gal545_A_100
            9,         # gal545_A_143
            21,        # gal545_A_143_217
            80,        # gal545_A_217
            1.,        # A_sbpx_100_100_TT
            1.,        # A_sbpx_143_143_TT
            1.,        # A_sbpx_143_217_TT
            1.,        # A_sbpx_217_217_TT
            1.0002,    # calib_100T
            0.99805,   # calib_217T
            A_planck   # A_planck
        ]
    # Compute log-likelihoods separately and add them
    logl_tt_lowl = tt_lowl_lkl(params_tt_lowl)
    logl_ee_lowl = ee_lowl_lkl(params_ee_lowl)
    logl_highl = highl_lkl(params_highl)
    logl = logl_tt_lowl + logl_ee_lowl + logl_highl

    # Now calculate for the fiducial cosmology
    print("\n--- Calculating for Fiducial Cosmology ---")
    # This file is created by the 'clik_get_selfcheck' command-line tool.
    # We re-initialize the likelihood to get the clean diagnostic printout.
    lkl_fiducial = clik.clik("/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik")
    params_fiducial = np.loadtxt("/home/ashandonay/data/cosmopower/fiducial_cosmology.txt")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(plot_ell, params_fiducial[2:lmax+1]*(plot_ell*(plot_ell+1))/(2*np.pi), color="black", label="Fiducial")
    ax2.plot(plot_ell, params_fiducial[2:lmax+1], color="black")
    logl_fiducial = lkl_fiducial(params_fiducial)
    print(f"Log-likelihood for fiducial data: {logl_fiducial}")
    ax1.set_ylabel('$\\frac{\ell(\ell+1)}{2 \pi} C_\ell$')
    ax1.set_xlabel('$\ell$')
    ax2.set_ylabel('$C_\ell$')
    ax2.set_xlabel('$\ell$')
    ax1.legend()
    plt.suptitle("Planck TT Commander (low-l)")
    plt.tight_layout()
    plt.savefig("tt_cl.png")
    plt.close()

    return np.array(logl)

if __name__ == "__main__":

    logl = compute_clik_log_likelihood(A_planck=1.0)
    print(f"A few log-likelihoods from your data: {logl[:5]}")

    # To convert log-likelihoods to a normalized likelihood distribution,
    # it's good practice to first shift them to avoid numerical errors
    # from exponentiating very large negative numbers. This gives the
    # relative likelihood.
    likelihoods = np.exp(logl - np.max(logl))

    # Now, we can plot a histogram of these likelihoods.
    # Using density=True normalizes the histogram so that the area of
    # the bars sums to 1, effectively creating a probability density plot.
    plt.figure()
    plt.hist(likelihoods, bins=20, density=True)
    plt.xlabel("Relative Likelihood")
    plt.ylabel("Probability Density")
    plt.title("Normalized Likelihood Distribution")
    plt.savefig("likelihood_hist.png")
    plt.close()