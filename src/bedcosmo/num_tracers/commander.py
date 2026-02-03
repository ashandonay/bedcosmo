import numpy as np
from astropy.io import fits
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class CommanderLowLTransformer:
    def __init__(self, sigma_fits_path, cov_low, lmin=2, lmax=29):
        self.lmin = lmin
        self.lmax = lmax
        self.ells = np.arange(lmin, lmax + 1)

        # Load spline data
        hdul = fits.open(sigma_fits_path)
        spline_data = hdul[0].data  # shape: (3, 249, 1000)

        # Extract per-ℓ grid, transform, and derivative vectors
        n_ell = spline_data.shape[1]
        self.cl_grids = [spline_data[0, i, :] for i in range(n_ell)]
        transform_vals = [spline_data[1, i, :] for i in range(n_ell)]
        derivative_vals = [spline_data[2, i, :] for i in range(n_ell)]
        print("Spline data shape:", spline_data.shape)
        # Build Hermite splines for forward transform
        self.forward_splines = [
            CubicHermiteSpline(self.cl_grids[i], transform_vals[i], derivative_vals[i])
            for i in range(n_ell)
        ]
        # Diagnostic min/max of each spline
        self.spline_mins = np.array([
            self.forward_splines[i](self.cl_grids[i]).min() for i in range(n_ell)
        ])
        self.spline_maxs = np.array([
            self.forward_splines[i](self.cl_grids[i]).max() for i in range(n_ell)
        ])
        # Use covariance matrix provided
        self.cov_low = cov_low


    def transform(self, cl_array):
        """Transform C_ell values to Commander Gaussianized space."""
        # Directly interpolate (with extrapolation) on the provided C_ell values
        x_out = np.zeros_like(cl_array, dtype=float)
        for i, ell in enumerate(self.ells):
            cl_val = cl_array[i]
            # Optionally warn if outside the original grid range
            grid_min, grid_max = self.cl_grids[i].min(), self.cl_grids[i].max()
            if cl_val < grid_min or cl_val > grid_max:
                warnings.warn(f"ℓ={ell}: C_ell={cl_val} outside grid range [{grid_min}, {grid_max}]. Extrapolating.")
            x_out[i] = self.forward_splines[i](cl_val)
        return x_out


    def inverse_transform(self, x_array):
        """Invert transformed variables back to C_ell values using numerical root finding."""
        cl_out = np.zeros_like(x_array)
        for i, ell in enumerate(self.ells):
            # Clamp target to valid interpolation range
            x_target = x_array[i]
            x_min, x_max = self.spline_mins[i], self.spline_maxs[i]
            if x_target < x_min or x_target > x_max:
                warnings.warn(f"ℓ={ell}: x_target={x_target} out of bounds. Clamping to [{x_min}, {x_max}].")
            # Solve g(C) = x_target for C via root finding
            spline = self.forward_splines[ell - 2]
            cl_min = self.cl_grids[i].min()
            cl_max = self.cl_grids[i].max()
            cl_out[i] = brentq(lambda cl: spline(cl) - x_target, cl_min, cl_max)
        return cl_out

    def apply_noise(self, cl_array):
        """Add Commander-modeled noise to C_ell values."""
        x_mean = self.transform(cl_array)
        x_samples = np.random.multivariate_normal(x_mean, self.cov_low)
        plt.figure()
        #plt.plot(cl_array, label="original C_ell")
        plt.plot(x_mean, label="transformed x mean")
        #plt.yscale("log")
        plt.xlabel("ℓ")
        plt.ylabel("x")
        plt.title("x mean")
        plt.grid(True)
        #plt.plot(x_samples)
        plt.savefig("x_samples.png", dpi=300)
        plt.close()
        cl_noisy = self.inverse_transform(x_samples)

        return cl_noisy

# Example usage
if __name__ == "__main__":
    # Path to sigma.fits
    sigma_path = "/home/ashandonay/cobaya/packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik/clik/lkl_0/_external/sigma.fits"

    # Load external covariance matrix (ℓ = 2–29 block)
    cov_full = fits.open(sigma_path)[2].data
    std_devs = fits.open(sigma_path)[1].data
    corr = cov_full / (np.outer(std_devs, std_devs) + 1e-30)
    cov = corr * np.outer(std_devs, std_devs)
    ells_full = np.arange(2, 251)
    mask = ells_full <= 29
    cov_low = cov[mask][:, mask]
    plt.figure()
    plt.imshow(cov_low, norm=LogNorm())
    plt.xlabel("ℓ")
    plt.ylabel("ℓ")
    plt.title("Commander covariance matrix")
    plt.colorbar()
    plt.savefig("cov_low.png", dpi=300)
    plt.close()

    # Create transformer
    transformer = CommanderLowLTransformer(sigma_path, cov_low)

    # Hermite spline fidelity test: compare forward_splines to raw transform arrays
    hdul = fits.open(sigma_path)
    spline_data = hdul[0].data  # shape: (3, n_ell, n_grid)


    lowl_TT_dl = np.load("/home/ashandonay/data/cosmopower/dl_tt.npy")[0, 2:30]
    lowl_TT_ell = np.load("/home/ashandonay/data/cosmopower/tt_ell.npy")[2:30]

    plt.figure()
    # Apply noise in D_ell domain
    for i in range(5):
        noisy_Dl = transformer.apply_noise(lowl_TT_dl)
        plt.scatter(lowl_TT_ell, noisy_Dl, alpha=0.2, color="tab:blue")
    plt.plot(np.arange(2, 30), lowl_TT_dl, label="original $D_\\ell$", color="black")
    #plt.plot(np.arange(2, 30), noisy_Dl, '--', label="noisy $D_\\ell$")
    plt.xlabel("$\\ell$")
    plt.ylabel("$D_\\ell$ [$\\mu K^2$]")
    plt.legend()
    plt.title("Commander-simulated low-$\\ell$ TT noise")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("commander_noisy_cls.png", dpi=300)