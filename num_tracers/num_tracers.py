import torch
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
from astropy.cosmology import Planck18
from astropy import constants
from scipy.integrate import trapezoid
from bed.grid import GridStack

class NumTracers:

    def __init__(self, desi_data, desi_tracers, nominal_cov, priors, obs_labels, device, eff=True, include_D_M=False):
        self.priors = priors
        self.desi_tracers = desi_tracers
        self.cosmo_params = set(priors.keys())
        self.obs_labels = obs_labels
        self.rdrag = 149.77
        self.hrdrag = Planck18.h*self.rdrag
        self.c = constants.c.to('km/s').value
        self.nominal_cov = nominal_cov
        self.corr_matrix = torch.tensor(self.nominal_cov/np.sqrt(np.outer(np.diag(self.nominal_cov), np.diag(self.nominal_cov))), device=device)
        self.include_D_M = include_D_M
        self.device = device

        if eff:
            self.efficiency = torch.tensor(desi_data["efficiency"].tolist()[::2], device=device)
        else:
            self.efficiency = torch.ones(5, device=device)

        self.sigmas = torch.tensor(desi_data["std"].tolist(), device=device)
        self.central_val = torch.tensor(desi_data["value_at_z"].tolist(), device=device)
        self.z_eff = torch.tensor(desi_data["z"].tolist()[::2], device=device)
        passed_num = torch.tensor(desi_data["num"].tolist()[::2], device=device)
        # use nominal efficiency to calculate the nominal passed ratio
        self.nominal_tot = (passed_num/self.efficiency).sum()
        self.nominal_passed_ratio = passed_num/self.nominal_tot
        print(f"z_eff: {self.z_eff}\n",
            f"sigmas: {self.sigmas}\n",
            f"nominal_passed: {self.nominal_passed_ratio}")

    def calc_passed(self, class_ratio):
        assert class_ratio.shape[-1] == 3, "class_ratio should have 3 columns (LRG, ELG, QSO)"
        obs_ratio = torch.zeros((class_ratio.shape[0], class_ratio.shape[1], 5), device=self.device)

        LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
        LRG_dist = class_ratio[..., 0].unsqueeze(-1) * torch.tensor((LRGs / LRGs.sum()).values, device=self.device).unsqueeze(0)
        obs_ratio[..., 0:2] = LRG_dist[..., 0:2]
        ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
        ELG_dist = class_ratio[..., 1].unsqueeze(-1) * torch.tensor((ELGs / ELGs.sum()).values, device=self.device).unsqueeze(0)
        # add the last value in LRG_dist to the first value in ELG_dist to get LRG3+ELG1
        obs_ratio[..., 2] = (LRG_dist[..., 2] + ELG_dist[..., 0])
        obs_ratio[..., 3] = ELG_dist[..., 1]

        QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
        QSO_dist = class_ratio[..., 2].unsqueeze(-1) * torch.tensor((QSOs / QSOs.sum()).values, device=self.device).unsqueeze(0)
        obs_ratio[..., 4] = QSO_dist[..., 0]

        efficiency = torch.stack([
            torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG1", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 0]),
            torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG2", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 1]),
            (LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[0] + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[0],
            torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG2", "efficiency"].values[0], device=self.device).expand_as(ELG_dist[..., 1]),
            torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "Lya QSO", "efficiency"].values[0], device=self.device).expand_as(QSO_dist[..., 0])
        ], dim=-1)

        # scale obs_ratio by efficiency to get the number of passed objects
        passed_ratio = obs_ratio*efficiency

        return passed_ratio

    def D_H_func(self, z, Om, w0=None, wa=None, hrdrag=None):
        if self.cosmo_params == {'Om'}:
            return (self.c/(100*self.hrdrag)) * 1 / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu())).cpu()

        elif self.cosmo_params == {'Om', 'w0'}:
            return (self.c/(100*self.hrdrag)) * 1 / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu()) * (1+z.cpu())**(3*(1+w0.cpu()))).cpu()

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return (self.c/(100*self.hrdrag)) * 1 / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu()) * (1+z.cpu())**(3*(1+(w0.cpu()+wa.cpu()*(z.cpu()/(1+z.cpu())))))).cpu()
        
        elif self.cosmo_params == {'Om', 'w0', 'wa', 'hrdrag'}:
            return (self.c/(100000*hrdrag.cpu())) * 1 / torch.sqrt(Om.cpu() * (1+z.cpu())**3 + (1-Om.cpu()) * (1+z.cpu())**(3*(1+(w0.cpu()+wa.cpu()*(z.cpu()/(1+z.cpu())))))).cpu()
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def D_M_func(self, z, Om, w0=None, wa=None, hrdrag=None):
        if self.cosmo_params == {'Om'}:
            result = (self.c/(100*self.hrdrag)) * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0'}:
            result = (self.c/(100*self.hrdrag)) * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()) * (1 + z.cpu())**(3 * (1 + w0.unsqueeze(-1).cpu())))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            result = (self.c/(100*self.hrdrag)) * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()) * (1 + z.cpu())**(3 * (1 + (w0.unsqueeze(-1).cpu() + wa.unsqueeze(-1).cpu() * (z.cpu() / (1 + z.cpu()))))))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)
        
        elif self.cosmo_params == {'Om', 'w0', 'wa', 'hrdrag'}:
            result = (self.c/(100000*hrdrag.cpu())) * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1).cpu() * (1 + z.cpu())**3 + (1 - Om.unsqueeze(-1).cpu()) * (1 + z.cpu())**(3 * (1 + (w0.unsqueeze(-1).cpu() + wa.unsqueeze(-1).cpu() * (z.cpu() / (1 + z.cpu()))))))).cpu(), 
                z.cpu(), 
                axis=-1)
            return result.to(self.device)
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def sample_flow(self, tracer_ratio, posterior_flow, num_data_samples=100, num_param_samples=1000):

        rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
        rescaled_sigmas[1::2] = self.sigmas[1::2] * torch.sqrt((self.efficiency[1::2]*tracer_ratio)/self.nominal_passed_ratio)
        if self.include_D_M:
            means = self.central_val
            rescaled_sigmas[::2] = self.sigmas[::2] * torch.sqrt((self.efficiency[::2]*tracer_ratio)/self.nominal_passed_ratio)
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            means = self.central_val[1::2]
            covariance_matrix = self.corr_matrix[1::2, 1::2] * (rescaled_sigmas[1::2].unsqueeze(-1) * rescaled_sigmas[1::2].unsqueeze(-2))

        with pyro.plate("data", num_data_samples):
            data_samples = pyro.sample("y", dist.MultivariateNormal(means, covariance_matrix))

        context = torch.cat([lexpand(tracer_ratio, num_data_samples), data_samples], dim=-1)
        # Sample parameters conditioned on the data batch
        param_samples = posterior_flow(context).sample((num_param_samples,))

        return torch.flatten(param_samples, start_dim=0, end_dim=-2)

    def sample_brute_force(self, tracer_ratio, grid_designs, grid_features, grid_params, designer, num_data_samples=100, num_param_samples=1000):
        
        rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
        rescaled_sigmas[1::2] = self.sigmas[1::2] * torch.sqrt((self.efficiency[1::2]*tracer_ratio)/self.nominal_passed_ratio)
        if self.include_D_M:
            means = self.central_val
            rescaled_sigmas[::2] = self.sigmas[::2] * torch.sqrt((self.efficiency[::2]*tracer_ratio)/self.nominal_passed_ratio)
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            means = self.central_val[1::2]
            covariance_matrix = self.corr_matrix[1::2, 1::2] * (rescaled_sigmas[1::2].unsqueeze(-1) * rescaled_sigmas[1::2].unsqueeze(-2))

        with pyro.plate("data", num_data_samples):
            data_samples = pyro.sample("y", dist.MultivariateNormal(means, covariance_matrix))
            
        post_samples = []
        post_input = {}
        for j in range(num_data_samples):
            for i, k in enumerate(grid_designs.names):
                post_input[k] = tracer_ratio[i].item()
            for i, k in enumerate(grid_features.names):
                post_input[k] = data_samples[j, i].item()
            posterior_pdf = designer.get_posterior(**post_input)
            param_samples = []
            flat_pdf = posterior_pdf.flatten()
            indices = np.array(list(np.ndindex(posterior_pdf.shape)))
            sampled_indices = np.random.choice(len(indices), size=num_param_samples, p=flat_pdf)
            indices = indices[sampled_indices]
            param_samples = []
            param_mesh = np.stack(np.meshgrid(*[getattr(grid_params, grid_params.names[i]).squeeze() for i in range(len(grid_params.names))], indexing='ij'), axis=-1)
            for i in range(num_param_samples):
                param_samples.append(param_mesh[tuple(indices[i])])
            post_samples.append(param_samples)
        param_samples = torch.tensor(np.array(post_samples), device=self.device)
        return param_samples.reshape(-1, len(grid_params.names))

    def brute_force_posterior(self, tracer_ratio, designer, grid_params, num_param_samples=1000):
        
        with pyro.plate("plate", num_param_samples):
            parameters = {}
            for i, (k, v) in enumerate(self.priors.items()):
                if isinstance(v, dist.Distribution):
                    parameters[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    parameters[k] = v

        rescaled_sigmas = torch.zeros(grid_params.shape + (len(self.sigmas),), device=self.device)
        rescaled_sigmas[..., 1::2] = self.sigmas[1::2] * torch.sqrt((self.efficiency[1::2]*tracer_ratio)/self.nominal_passed_ratio)
        if self.include_D_M:
            y = self.central_val
            rescaled_sigmas[..., ::2] = self.sigmas[::2] * torch.sqrt((self.efficiency[::2]*tracer_ratio)/self.nominal_passed_ratio)
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            y = self.central_val[1::2]
            covariance_matrix = self.corr_matrix[1::2, 1::2] * (rescaled_sigmas[..., 1::2].unsqueeze(-1) * rescaled_sigmas[..., 1::2].unsqueeze(-2))
        with GridStack(grid_params):
            parameters = {k: torch.tensor(getattr(grid_params, k), device=self.device).unsqueeze(-1) for k in grid_params.names}
        mean = torch.zeros(grid_params.shape + (len(self.sigmas),), device=self.device)
        z = self.z_eff.reshape((len(self.cosmo_params))*[1] + [-1])
        mean[..., 1::2] = self.D_H_func(z, **parameters)

        if self.include_D_M:
            z_array = self.z_eff.unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
            z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
            mean[..., ::2] = self.D_M_func(z, **parameters)
        else:
            mean = mean[..., 1::2]

        # evaluate the multivariate normal likelihood
        likelihood = dist.MultivariateNormal(mean, covariance_matrix).log_prob(y).exp()

        # normalize the likelihood to get the posterior
        posterior_pdf = likelihood / likelihood.sum()

        flat_pdf = posterior_pdf.cpu().flatten()
        indices = np.array(list(np.ndindex(posterior_pdf.shape)))
        sampled_indices = np.random.choice(len(indices), size=num_param_samples, p=flat_pdf)
        indices = indices[sampled_indices]
        param_samples = []
        param_mesh = np.stack(np.meshgrid(*[getattr(grid_params, grid_params.names[i]).squeeze() for i in range(len(grid_params.names))], indexing='ij'), axis=-1)
        for i in range(num_param_samples):
            param_samples.append(param_mesh[tuple(indices[i])])
        return torch.tensor(np.array(param_samples), device=self.device)

    def pyro_model(self, tracer_ratio):

        passed_ratio = self.calc_passed(tracer_ratio)
        with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
            parameters = {}
            for i, (k, v) in enumerate(self.priors.items()):
                if isinstance(v, dist.Distribution):
                    parameters[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    parameters[k] = v
            means = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)

            z = self.z_eff.reshape(2*[1] + [-1])
            means[:, :, 1::2] = self.D_H_func(z, **parameters)
            rescaled_sigmas[:, :, 1::2] = self.sigmas[1::2] * torch.sqrt(passed_ratio/self.nominal_passed_ratio)
            if self.include_D_M:
                z_array = self.z_eff.unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand(2*[1] + [-1, -1])
                means[:, :, ::2] = self.D_M_func(z, **parameters).cpu()
                rescaled_sigmas[:, :, ::2] = self.sigmas[::2] * torch.sqrt(passed_ratio/self.nominal_passed_ratio)

            # extract correlation matrix from DESI covariance matrix
            if self.include_D_M:
                means = means.to(self.device)
                # convert correlation matrix to covariance matrix using rescaled sigmas
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                # only use D_H values for mean and covariance matrix
                means = means[:, :, 1::2].to(self.device)
                covariance_matrix = self.corr_matrix[1::2, 1::2] * (rescaled_sigmas[:, :, 1::2].unsqueeze(-1) * rescaled_sigmas[:, :, 1::2].unsqueeze(-2))
                
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
            D_H_sigma = self.sigmas[1::2].cpu().numpy()[i] * np.sqrt((self.efficiency[1::2].cpu().numpy()[i]*getattr(designs, designs.names[i]))/self.nominal_passed_ratio[i].cpu().numpy())
            likelihood = np.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood
            
            if self.include_D_M:
                z_array = self.z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
                D_M_mean = self.D_M_func(z, **parameters)
                D_M_diff = getattr(features, features.names[i+len(self.z_eff)]) - D_M_mean.cpu().numpy()
                D_M_sigma = self.sigmas[::2].cpu().numpy()[i] * np.sqrt((self.efficiency[::2].cpu().numpy()[i]*getattr(designs, designs.names[i]))/self.nominal_passed_ratio[i].cpu().numpy())
                likelihood = np.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        return likelihood
