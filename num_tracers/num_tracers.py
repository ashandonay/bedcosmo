class bed_cosmology:

    def __init__(self, priors, true_n_ratios, obs_labels, z_eff, sigma_D_H, sigma_D_M=None, device=device):
        self.priors = priors
        self.cosmo_params = set(priors.keys())
        self.true_n_ratios = true_n_ratios
        self.obs_labels = obs_labels
        self.z_eff = z_eff
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        self.r_drag = 149.77
        self.H0 = Planck18.H0.value
        self.coeff = constants.c.to('km/s').value/(self.H0*self.r_drag)
        self.device = device
    
    def D_H_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om))

        elif self.cosmo_params == {'Om', 'w0'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+w0)))

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+(w0+wa*(z/(1+z))))))
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def D_M_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1) * (1 + z)**3 + (1 - Om.unsqueeze(-1)))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1) * (1 + z)**3 + (1 - Om.unsqueeze(-1)) * (1 + z)**(3 * (1 + w0.unsqueeze(-1))))).cpu(), 
                z.cpu(), 
                axis=-1)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            result = self.coeff * trapezoid(
                (1 / torch.sqrt(Om.unsqueeze(-1) * (1 + z)**3 + (1 - Om.unsqueeze(-1)) * (1 + z)**(3 * (1 + (w0.unsqueeze(-1) + wa.unsqueeze(-1) * (z / (1 + z))))))).cpu(), 
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

            z = self.z_eff.reshape((len(self.cosmo_params)-1)*[1] + [-1])
            means = self.D_H_func(z, **parameters)
            sigmas = self.sigma_D_H * torch.sqrt(self.true_n_ratios/tracers_ratio)

            if self.sigma_D_M is not None:
                z_array = self.z_eff.unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
                D_M_mean = self.D_M_func(z, **parameters)
                D_M_sigmas = self.sigma_D_M * torch.sqrt(self.true_n_ratios/tracers_ratio)
                means = torch.cat((means, D_M_mean), dim=-1)
                sigmas = torch.cat((sigmas, D_M_sigmas), dim=-1)

            #for i,o in enumerate(self.obs_labels):
            #    pyro.sample(o, dist.Normal(means[..., i].unsqueeze(-1), sigmas[..., i].unsqueeze(-1)).to_event(1))
            covariance_matrix = torch.diag_embed(sigmas ** 2)
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
            D_H_sigma = self.sigma_D_H.cpu().numpy()[i] * np.sqrt(self.true_n_ratios[i].cpu().numpy()/getattr(designs, designs.names[i]))
            likelihood = np.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood
            
            if self.sigma_D_M is not None:
                z_array = self.z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
                z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
                D_M_mean = self.D_M_func(z, **parameters)
                D_M_diff = getattr(features, features.names[i+len(self.z_eff)]) - D_M_mean.cpu().numpy()
                D_M_sigma = self.sigma_D_M.cpu().numpy()[i] * np.sqrt(self.true_n_ratios[i].cpu().numpy()/getattr(designs, designs.names[i]))
                likelihood = np.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        return likelihood
