class bed_cosmology:

    def __init__(self, cosmo_params, priors, sigma_D_H=0.2, sigma_D_M=0.2, include_D_M=False, device=device):
        self.cosmo_params = set(cosmo_params.names)
        self.priors = priors
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        self.r_drag = 149.77
        self.H0 = Planck18.H0.value
        self.coeff = constants.c.to('km/s').value/(self.H0*self.r_drag)
        self.include_D_M = include_D_M
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
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0'}:
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + w0)))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + (w0 + wa * (z / (1 + z))))))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def likelihood(self, params, features, designs):
        with GridStack(features, designs, params):
            # create a dictionary of the parameters
            kwargs = { }
            for key in params.names:
                kwargs[key] = getattr(params, key)

            D_H_mean = self.D_H_func(designs.z, **kwargs)
            D_H_diff = features.D_H - D_H_mean
            D_H_likelihood = np.exp(-0.5 * (D_H_diff / self.sigma_D_H) ** 2) 

            if self.include_D_M:
                D_M_mean = self.D_M_func(designs.z, **kwargs)
                D_M_diff = features.D_M - D_M_mean
                D_M_likelihood = np.exp(-0.5 * (D_M_diff / self.sigma_D_M) ** 2)
                likelihood = D_H_likelihood * D_M_likelihood
            else:
                likelihood = D_H_likelihood
            features.normalize(likelihood)
        return likelihood
    
    def pyro_model(self, z):
        with pyro.plate_stack("plate", z.shape[:-1]):
            kwargs = {}
            for i, (k, v) in enumerate(self.priors.items()):
                if isinstance(v, dist.Distribution):
                    kwargs[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    kwargs[k] = v
            D_H_mean = self.D_H_func(z, **kwargs)
            if self.include_D_M:
                D_M_mean = self.D_M_func(z, **kwargs)
                return pyro.sample("D_H", dist.Normal(D_H_mean, self.sigma_D_H).to_event(1)), pyro.sample("D_M", dist.Normal(D_M_mean, self.sigma_D_M).to_event(1))
            else:
                return pyro.sample("D_H", dist.Normal(D_H_mean, self.sigma_D_H).to_event(1))