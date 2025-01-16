import torch
import numpy as np
import pyro
from pyro import distributions as dist
from astropy.cosmology import Planck18
from astropy import constants
from scipy.integrate import trapezoid

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

class NumTracers:

    def __init__(self, desi_data, nominal_cov, priors, obs_labels, eff=True, include_D_M=False, device=device):
        self.priors = priors
        self.cosmo_params = set(priors.keys())
        self.obs_labels = obs_labels
        self.r_drag = 149.77
        self.H0 = Planck18.H0.value
        self.coeff = constants.c.to('km/s').value/(self.H0*self.r_drag)
        self.nominal_cov = nominal_cov
        self.include_D_M = include_D_M
        self.device = device

        if eff:
            self.efficiency = torch.tensor(desi_data["efficiency"].tolist(), device=device)
        else:
            self.efficiency = torch.ones(len(desi_data), device=device)

        self.sigmas = torch.tensor(desi_data["std"].tolist(), device=device)
        self.z_eff = torch.tensor(desi_data["z"][::2].tolist(), device=device)
        nominal_num = torch.tensor(desi_data["num"].tolist(), device=device)
        N_tot = (nominal_num[::2]/self.efficiency[::2]).sum()
        self.nominal_n_ratios = nominal_num[::2]/N_tot
        print(f'z_eff: {self.z_eff}\n'
            f'sigmas: {self.sigmas}\n'
            f'efficiency: {self.efficiency}\n'
            f'nominal_n_ratios: {self.nominal_n_ratios}')
    
    def D_H_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            return self.coeff / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu())).cpu()

        elif self.cosmo_params == {'Om', 'w0'}:
            return self.coeff / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu()) * (1+z.cpu())**(3*(1+w0.cpu()))).cpu()

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return self.coeff / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu()) * (1+z.cpu())**(3*(1+(w0.cpu()+wa.cpu()*(z.cpu()/(1+z.cpu())))))).cpu()
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def D_M_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()) * (1 + z.cpu())**(3 * (1 + w0.unsqueeze(-1).cpu())))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()) * (1 + z.cpu())**(3 * (1 + (w0.unsqueeze(-1).cpu() + wa.unsqueeze(-1).cpu() * (z.cpu() / (1 + z.cpu()))))))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")
    
    def pyro_model(self, tracers_ratio):
        with pyro.plate_stack("plate", tracers_ratio.shape[:-1]):
            parameters = {}
            for i, (k, v) in enumerate(self.priors.items()):
                if isinstance(v, dist.Distribution):
                    parameters[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    parameters[k] = v
            means = torch.zeros(tracers_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas = torch.zeros(tracers_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)

            z = self.z_eff.reshape((len(self.cosmo_params)-1)*[1] + [-1])
            means[:, :, 1::2] = self.D_H_func(z, **parameters)
            rescaled_sigmas[:, :, 1::2] = self.sigmas[1::2] * torch.sqrt((self.efficiency[1::2]*tracers_ratio)/self.nominal_n_ratios)
            if self.include_D_M:
                z_array = self.z_eff.unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
                means[:, :, ::2] = self.D_M_func(z, **parameters).cpu()
                rescaled_sigmas[:, :, ::2] = self.sigmas[::2] * torch.sqrt((self.efficiency[::2]*tracers_ratio)/self.nominal_n_ratios)

            # extract correlation matrix from DESI covariance matrix
            self.corr_matrix = torch.tensor(self.nominal_cov/np.sqrt(np.outer(np.diag(self.nominal_cov), np.diag(self.nominal_cov))), device=self.device)
            if self.include_D_M:
                means = means.to(self.device)
                # convert correlation matrix to covariance matrix using rescaled sigmas
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                # only use D_H values for mean and covariance matrix
                means = means[:, :, 1::2].to(self.device)
                covariance_matrix = self.corr_matrix * (rescaled_sigmas[:, :, 1::2].unsqueeze(-1) * rescaled_sigmas[:, :, 1::2].unsqueeze(-2))
                
            return pyro.sample("y", dist.MultivariateNormal(means, covariance_matrix))

    def unnorm_lfunc(self, params, features, designs):
        parameters = { }
        for key in params.names:
            parameters[key] = torch.tensor(getattr(params, key), device=self.device)
        likelihood = 1
        for i in range(len(self.z_eff)):
            z = self.z_eff[i].reshape((len(self.cosmo_params)-1)*[1] + [-1])
            D_H_mean = self.D_H_func(z, **parameters)
            D_H_diff = getattr(features, features.names[i]) - D_H_mean.cpu().numpy()
            D_H_sigma = self.sigmas[1::2].cpu().numpy()[i] * np.sqrt((self.efficiency[1::2].cpu().numpy()[i]*getattr(designs, designs.names[i]))/self.nominal_n_ratios[i].cpu().numpy())
            likelihood = np.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood
            
            if self.include_D_M:
                z_array = self.z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
                D_M_mean = self.D_M_func(z, **parameters)
                D_M_diff = getattr(features, features.names[i+len(self.z_eff)]) - D_M_mean.cpu().numpy()
                D_M_sigma = self.sigmas[::2].cpu().numpy()[i] * np.sqrt((self.efficiency[::2].cpu().numpy()[i]*getattr(designs, designs.names[i]))/self.nominal_n_ratios[i].cpu().numpy())
                likelihood = np.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        return likelihood
