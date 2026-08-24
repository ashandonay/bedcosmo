import eval_clik
import numpy as np
import clik.cldf

print("Successfully imported clik")

# Try to create a simple test likelihood
try:
    # Create a simple test array
    test_array = np.ones(10)
    print("\nTest array created successfully")
    
    # Try to print some basic clik information
    print("\nAvailable clik functions:")
    print([func for func in dir(eval_clik) if not func.startswith('_')])
    
    print("\nclik installation test completed successfully!")

    # Path to your .clik file
    clik_path = "/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"

    # Open the clik file as a cldf file
    clf = eval_clik.cldf.File(clik_path)

    # Explore the keys to find where the samples might be stored
    print("\nTop-level keys:", clf.keys())
    
    # Function to recursively print all keys and their shapes
    def explore_structure(d, prefix=''):
        if hasattr(d, 'keys'):
            for k in d.keys():
                try:
                    if hasattr(d[k], 'shape'):
                        print(f"{prefix}{k}: shape {d[k].shape}")
                    else:
                        print(f"{prefix}{k}")
                    explore_structure(d[k], prefix + k + '/')
                except Exception as e:
                    print(f"{prefix}{k}: {str(e)}")

    print("\nExploring file structure:")
    explore_structure(clf['clik'])

    # Load the likelihood
    lkl = eval_clik.eval_clik(clik_path)
    
    # Get the lmax for TT (first element of the lmax array)
    lmax = lkl.get_lmax()[0]  # Should be 29 for commander
    print(f"\nTT spectrum lmax: {lmax}")
    
    # Create a fiducial spectrum (all ones for testing)
    n_cls = lmax + 1  # Number of Cl values (0 to lmax inclusive)
    n_nuisance = len(lkl.get_extra_parameter_names())
    print(f"Number of nuisance parameters: {n_nuisance}")
    
    # Create a fiducial input vector
    fiducial = np.ones(n_cls + n_nuisance)
    
    # Compute the likelihood at the fiducial point
    logl_fiducial = lkl(fiducial)
    print(f"\nLog-likelihood at fiducial point: {logl_fiducial}")
    
    # To compute the covariance, we would need to:
    # 1. Sample the likelihood around the fiducial point
    # 2. Compute the covariance of these samples
    print("\nTo compute the covariance, we would need to:")
    print("1. Sample the likelihood around the fiducial point")
    print("2. Compute the covariance of these samples")
    print("\nThis would require:")
    print("- A good initial guess for the fiducial spectrum")
    print("- A sampling strategy (e.g., MCMC or importance sampling)")
    print("- Enough samples to get a reliable covariance estimate")

except Exception as e:
    print(f"\nError during test: {str(e)}")

class CommanderLikelihood:
    def __init__(self, clik_path):
        """Initialize the Commander likelihood wrapper.
        
        Args:
            clik_path: Path to the commander_dx12_v3_2_29.clik file
        """
        self.lkl = eval_clik.eval_clik(clik_path)
        self.lmax = self.lkl.get_lmax()[0]  # Should be 29 for commander
        self.n_cls = self.lmax + 1  # Number of Cl values (0 to lmax inclusive)
        self.n_nuisance = len(self.lkl.get_extra_parameter_names())  # Should be 1 (calibration)
        self.total_params = self.n_cls + self.n_nuisance
        
        print(f"Initialized Commander likelihood:")
        print(f"- TT spectrum range: l=0 to {self.lmax}")
        print(f"- Number of nuisance parameters: {self.n_nuisance}")
        print(f"- Total parameters: {self.total_params}")
    
    def log_likelihood(self, params):
        """Compute the log-likelihood for a given set of parameters.
        
        Args:
            params: Array of shape (n_samples, n_params) or (n_params,)
                   containing the TT spectrum (l=0 to lmax) and nuisance parameters
                   
        Returns:
            Array of log-likelihood values
        """
        # Ensure params is 2D
        if params.ndim == 1:
            params = params[None, :]
            
        # Check shape
        if params.shape[1] != self.total_params:
            raise ValueError(f"Expected {self.total_params} parameters, got {params.shape[1]}")
            
        # Compute log-likelihood for each sample
        logl = np.array([self.lkl(p) for p in params])
        return logl

# Example usage:
if __name__ == "__main__":
    # Path to your .clik file
    clik_path = "/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"
    
    # Initialize the likelihood
    commander = CommanderLikelihood(clik_path)
    
    # Create more realistic test parameters
    n_test_samples = 5
    
    # Create a reasonable TT spectrum
    # l=0,1 should be zero, then increasing values for higher l
    test_spectrum = np.zeros(commander.n_cls)
    test_spectrum[2:] = np.linspace(1000, 2000, commander.n_cls-2)  # Example values
    
    # Create test samples with small variations
    test_params = np.zeros((n_test_samples, commander.total_params))
    for i in range(n_test_samples):
        # Add the spectrum
        test_params[i, :commander.n_cls] = test_spectrum
        # Add calibration parameter (close to 1.0)
        test_params[i, -1] = 1.0 + 0.01 * np.random.randn()
    
    # Compute log-likelihood
    logl = commander.log_likelihood(test_params)
    print("\nTest log-likelihood values:")
    print(logl)
    
    print("\nParameter ranges used in test:")
    print(f"TT spectrum: {test_spectrum[2:5]}... (first two elements are zero)")
    print(f"Calibration: {test_params[:, -1]}")
    
    print("\nThis likelihood wrapper can now be used with your generative model!")
    print("Just pass samples from your prior to the log_likelihood method.") 