import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from astropy.cosmology import FLRW, w0waCDM, LambdaCDM, FlatLambdaCDM
from astropy import constants

sys.path.append('/home/ashandonay/bed/bayesdesign/src/bed/')
from grid import Grid, GridStack
from design import ExperimentDesigner

class bed_cosmology:

    def __init__(self, cosmo_params, sigma_D_H=0.2, sigma_D_M=0.2, include_D_M=False):
        self.cosmo_params = set(cosmo_params.names)
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        self.include_D_M = include_D_M
        self.r_drag = 149.77
        self.H0 = Planck18.H0.value
        self.coeff = constants.c.to('km/s').value/(self.H0*self.r_drag)
    
    def D_H_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            return self.coeff/np.sqrt(Om * (1+z)**3 + (1-Om))

        elif self.cosmo_params == {'Om', 'w0'}:
            return self.coeff/np.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+w0)))

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return self.coeff/np.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+(w0+wa*(z/(1+z))))))
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def D_M_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            return self.coeff * cumulative_trapezoid(1 / np.sqrt(Om * (1 + z)**3 + (1 - Om)), z, initial=0, axis=0)

        elif self.cosmo_params == {'Om', 'w0'}:
            return self.coeff * cumulative_trapezoid(1 / np.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + w0))), z, initial=0, axis=0)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return self.coeff * cumulative_trapezoid(1 / np.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + (w0 + wa * (z / (1 + z)))))), z, initial=0, axis=0)
            
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

include_D_M = False
cosmology = 'w'

eigs = []
for i,d in enumerate(np.linspace(0, 5, 100)):
    designs = Grid(z=d)
    if cosmology == 'Om':
        params = Grid(Om=np.linspace(0.2, 0.4, 1000))
    elif cosmology == 'w':
        params = Grid(Om=np.linspace(0.2, 0.4, 1001), w0=np.linspace(-3, 1, 1001))
    elif cosmology == 'w0wa':
        params = Grid(Om=np.linspace(0.2, 0.4, 200), w0=np.linspace(-3, 1, 200), wa=np.linspace(-3, 2, 200))

    if include_D_M:
        features = Grid(D_H=np.linspace(0.1, 40, 120), D_M=np.linspace(0.1, 80, 120))
    else:
        features = Grid(D_H=np.linspace(0.1, 35, 520))

    cosmo = bed_cosmology(
        params,
        sigma_D_H=0.2,
        sigma_D_M=0.2,
        include_D_M=include_D_M
        )

    likelihood = cosmo.likelihood(params, features, designs)

    prior = np.ones(params.shape)
    params.normalize(prior);

    designer = ExperimentDesigner(params, features, designs, likelihood)
    designer.calculateEIG(prior)
    if i == 0:
        designer.describe()
    eigs.append(designer.EIG)

eigs = np.array(eigs)
np.save('grid_w_eigs_3.npy', np.stack((np.linspace(0, 5, 100).reshape(-1, 1), eigs), axis=1))