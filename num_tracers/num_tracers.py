import os
import sys
import mlflow
import pandas as pd
import torch
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
from astropy.cosmology import Planck18
from astropy import constants
from torch import trapezoid
from bed.grid import GridStack
from bed.grid import Grid
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory ('BED_cosmo/') and add it to the Python path
parent_dir_abs = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir_abs)
from util import *

storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(storage_path + "/mlruns")

class NumTracers:
    def __init__(
        self, 
        data_path="/data/tracers_v1/", 
        cosmo_model="base", 
        design_step=0.05, 
        design_lower=0.05, 
        design_upper=None, 
        fixed_design=False, 
        global_rank=0, 
        device="cuda:0", 
        include_D_M=True, 
        include_D_V=True,
        seed=None,
        verbose=False
    ):

        self.name = 'num_tracers'
        self.desi_data = pd.read_csv(home_dir + data_path + 'desi_data.csv')
        self.desi_tracers = pd.read_csv(home_dir + data_path + 'desi_tracers.csv')
        self.nominal_cov = np.load(home_dir + data_path + 'desi_cov.npy')
        self.DH_idx = np.where(self.desi_data["quantity"] == "DH_over_rs")[0]
        self.DM_idx = np.where(self.desi_data["quantity"] == "DM_over_rs")[0]
        self.DV_idx = np.where(self.desi_data["quantity"] == "DV_over_rs")[0]
        self.cosmo_model = cosmo_model
        self.device = device
        self.global_rank = global_rank  
        self.seed = seed
        self.verbose = verbose
        if seed is not None:
            auto_seed(self.seed)
        self.rdrag = 149.77
        self.c = constants.c.to('km/s').value
        self.corr_matrix = torch.tensor(self.nominal_cov/np.sqrt(np.outer(np.diag(self.nominal_cov), np.diag(self.nominal_cov))), device=self.device)
        self.include_D_M = include_D_M
        self.include_D_V = include_D_V
        self.efficiency = torch.tensor(self.desi_data.drop_duplicates(subset=['tracer'])['efficiency'].tolist(), device=self.device)
        self.sigmas = torch.tensor(self.desi_data["std"].tolist(), device=self.device)
        self.central_val = torch.tensor(self.desi_data["value_at_z"].tolist(), device=self.device)
        self.tracer_bins = self.desi_data.drop_duplicates(subset=['tracer'])['tracer'].tolist()
        self.total_obs = int(self.desi_data.drop_duplicates(subset=['tracer'])['observed'].sum())
        self.nominal_passed_ratio = torch.tensor(self.desi_data['passed'].tolist(), device=self.device)/self.total_obs
        # Create dictionary with upper limits and lower limit lists for each class
        self.targets = ["BGS", "LRG", "ELG", "QSO"]
        self.num_targets = self.desi_tracers.groupby('class').sum()['targets'].reindex(self.targets)
        self.context_dim = len(self.targets) + 5
        if include_D_M:
            self.context_dim += 5
        if include_D_V:
            self.context_dim += 2
        self.nominal_design = torch.tensor(self.desi_tracers.groupby('class').sum()['observed'].reindex(self.targets).values, device=self.device)
        self.get_priors() # initialize the priors
        self.cosmo_params = list(self.priors.keys())
        self.observation_labels = ["y"]
        self.init_designs(fixed_design=fixed_design, design_step=design_step, design_lower=design_lower, design_upper=design_upper)
        if self.global_rank == 0 and self.verbose:
            print(f"tracer_bins: {self.tracer_bins}\n",
                f"sigmas: {self.sigmas}\n",
                f"nominal_design: {self.nominal_design}\n",
                f"nominal_passed: {self.nominal_passed_ratio}")


    def init_designs(self, fixed_design=False, design_step=0.05, design_lower=0.05, design_upper=1.0):
        if fixed_design:
            # Create grid with nominal design values using self.targets
            grid_params = {
                f'N_{target}': self.nominal_design[i].cpu().numpy() 
                for i, target in enumerate(self.targets)
            }
            grid_designs = Grid(**grid_params)

            # Convert grid to tensor format
            designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=self.device).unsqueeze(0)
            for name in grid_designs.names[1:]:
                design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=self.device).unsqueeze(0)
                designs = torch.cat((designs, design_tensor), dim=0)
            designs = designs.unsqueeze(0)

        else:
            if type(design_step) == float:
                design_steps = [design_step]*len(self.targets)
            elif type(design_step) == list:
                design_steps = design_step
            else:
                raise ValueError("design_lower must be a float or list")
            
            if type(design_lower) == float:
                lower_limits = [design_lower]*len(self.targets)
            elif type(design_lower) == list:
                lower_limits = design_lower
            else:
                raise ValueError("design_lower must be a float or list")
            
            if design_upper is None:
                upper_limits = [self.num_targets[target] / self.total_obs for target in self.targets]
            elif type(design_upper) == float:
                upper_limits = [design_upper]*len(self.targets)
            elif type(design_upper) == list:
                upper_limits = design_upper
            else:
                raise ValueError("design_upper must be a float or list")
            
            if self.global_rank == 0 and self.verbose:
                print(f"design_lower: {lower_limits}\n",
                    f"design_upper: {upper_limits}\n",
                    f"design_step: {design_step}")
                
            designs_dict = {
                f'N_{target}': np.arange(
                    lower_limits[i],
                    upper_limits[i] + design_steps[i],
                    design_steps[i]
                ) for i, target in enumerate(self.targets)
            }
            
            # Create constrained grid ensuring designs sum to 1
            tol = 1e-3
            grid_designs = Grid(
                **designs_dict, 
                constraint=lambda **kwargs: abs(sum(kwargs.values()) - 1.0) < tol
            )

            # Convert to tensor format
            designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=self.device).unsqueeze(1)
            for name in grid_designs.names[1:]:
                design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=self.device).unsqueeze(1)
                designs = torch.cat((designs, design_tensor), dim=1)
        self.designs = designs.to(self.device)

    def get_priors(self):
        Om_range = torch.tensor([0.01, 0.99], device=self.device)
        Ok_range = torch.tensor([-0.3, 0.3], device=self.device)
        w0_range = torch.tensor([-3.0, 1.0], device=self.device)
        wa_range = torch.tensor([-3.0, 2.0], device=self.device)
        hrdrag_range = torch.tensor([0.01, 1.0], device=self.device)

        if self.cosmo_model == 'base':
            self.priors = {'Om': dist.Uniform(*Om_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
            self.latex_labels = ['\Omega_m', 'H_0r_d']
        elif self.cosmo_model == 'base_omegak':
            self.priors = {'Om': dist.Uniform(*Om_range), 'Ok': dist.Uniform(*Ok_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
            self.latex_labels = ['\Omega_m', '\Omega_k', 'H_0r_d']
        elif self.cosmo_model == 'base_w':
            self.priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
            self.latex_labels = ['\Omega_m', 'w_0', 'H_0r_d']
        elif self.cosmo_model == 'base_w_wa':
            self.priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
            self.latex_labels = ['\Omega_m', 'w_0', 'w_a', 'H_0r_d']
        elif self.cosmo_model == 'base_omegak_w_wa':
            self.priors = {'Om': dist.Uniform(*Om_range), 'Ok': dist.Uniform(*Ok_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
            self.latex_labels = ['\Omega_m', '\Omega_k', 'w_0', 'w_a', 'H_0r_d']
            
    def calc_passed(self, class_ratio):
        if type(class_ratio) == torch.Tensor:
            assert class_ratio.shape[-1] == len(self.targets), f"class_ratio should have {len(self.targets)} columns"
            obs_ratio = torch.zeros((class_ratio.shape[0], class_ratio.shape[1], len(self.desi_data)), device=self.device)

            # multiply each class ratio by the observed fraction in each tracer bin
            BGSs = self.desi_tracers.loc[self.desi_tracers["class"] == "BGS"]["observed"]
            BGS_dist = class_ratio[..., 0].unsqueeze(-1) * torch.tensor((BGSs / BGSs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = BGS_dist[..., 0].unsqueeze(-1)

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_dist = class_ratio[..., 1].unsqueeze(-1) * torch.tensor((LRGs / LRGs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = LRG_dist[..., 0].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = LRG_dist[..., 1].unsqueeze(-1)

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_dist = class_ratio[..., 2].unsqueeze(-1) * torch.tensor((ELGs / ELGs.sum()).values, device=self.device).unsqueeze(0)
            # add the last value in LRG_dist to the first value in ELG_dist to get LRG3+ELG1
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = (LRG_dist[..., 2] + ELG_dist[..., 0]).unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = ELG_dist[..., 1].unsqueeze(-1)

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_dist = class_ratio[..., 3].unsqueeze(-1) * torch.tensor((QSOs / QSOs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = QSO_dist[..., 0].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = QSO_dist[..., 1].unsqueeze(-1)

            efficiency = torch.zeros((class_ratio.shape[0], class_ratio.shape[1], len(self.desi_data)), device=self.device)
            efficiency[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "BGS", "efficiency"].values[0], device=self.device).expand_as(BGS_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG1", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG2", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 1]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = ((LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[0] + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG2", "efficiency"].values[0], device=self.device).expand_as(ELG_dist[..., 1]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "QSO", "efficiency"].values[0], device=self.device).expand_as(QSO_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "Lya QSO", "efficiency"].values[0], device=self.device).expand_as(QSO_dist[..., 1]).unsqueeze(-1)

            # scale obs_ratio by efficiency to get the number of passed objects
            passed_ratio = obs_ratio*efficiency

            return passed_ratio
        elif type(class_ratio) == Grid:
            obs_ratio = np.zeros((5,) + len(class_ratio.shape)*(1,))

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_dist = (class_ratio.N_LRG * np.array((LRGs / LRGs.sum()).values)).squeeze()
            
            obs_ratio[0:2, ...] = LRG_dist[0:2]

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_dist = (class_ratio.N_ELG * np.array((ELGs / ELGs.sum()).values)).squeeze()
            obs_ratio[..., 2] = (LRG_dist[..., 2] + ELG_dist[..., 0])
            obs_ratio[..., 3] = ELG_dist[..., 1]

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_dist = (class_ratio.N_QSO * np.array((QSOs / QSOs.sum()).values)).squeeze()
            obs_ratio[..., 4] = QSO_dist[..., 0]

            efficiency = np.stack([
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG1", "efficiency"].values[0]),
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG2", "efficiency"].values[0]),
                (LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[0] + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[0],
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG2", "efficiency"].values[0]),
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "Lya QSO", "efficiency"].values[0])
            ], axis=-1)

            passed_ratio = obs_ratio*efficiency

            return passed_ratio


    def D_H_func(self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None):
        """
        Hubble distance divided by the sound horizon D_H/r_d
        """
        z = z_eff.reshape((len(self.cosmo_params))*[1] + [-1])
        if self.cosmo_model == 'base':
            return (self.c/(100000*hrdrag)) * 1 / torch.sqrt(
                Om * (1+z)**3 + 
                (1-Om)
                )
        
        elif self.cosmo_model == 'base_omegak':
            return (self.c/(100000*hrdrag)) * 1 / torch.sqrt(
                Om * (1+z)**3 + 
                Ok * (1+z)**2 + 
                (1 - Om - Ok)
                )

        elif self.cosmo_model == 'base_w':
            return (self.c/(100000*hrdrag)) * 1 / torch.sqrt(
                Om * (1+z)**3 + 
                (1-Om) * (1+z)**(3*(1+w0))
                )
        
        elif self.cosmo_model == 'base_w_wa':
            return (self.c/(100000*hrdrag)) * 1 / torch.sqrt(
                Om * (1+z)**3 + 
                (1 - Om) * (1 + z)**(3 * (1 + w0 + wa)) * torch.exp(-3 * wa * (z / (1 + z)))
                )
        
        elif self.cosmo_model == 'base_omegak_w_wa':
            return (self.c/(100000*hrdrag)) * 1 / torch.sqrt(
                Om * (1+z)**3 + Ok * (1+z)**2 + 
                (1 - Om - Ok) * (1 + z)**(3 * (1 + w0 + wa)) * torch.exp(-3 * wa * (z / (1 + z)))
                )

    def D_M_func(self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None):
        """
        Transverse comoving distance divided by the sound horizon D_M/r_d
        """
        z_array = z_eff.unsqueeze(-1) * torch.linspace(0, 1, 100, device=self.device).view(1, -1)
        z = z_array.expand((len(self.cosmo_params)-1)*[1] + [-1, -1])
        if self.cosmo_model == 'base':
            # calculates the transverse comoving distance for a lambdaCDM cosmology 
            result = (self.c/(100000*hrdrag)) * trapezoid(
                (1 / torch.sqrt(
                    Om.unsqueeze(-1) * (1 + z)**3 + 
                    (1 - Om.unsqueeze(-1))
                    )), 
                z, 
                axis=-1)
            return result

        elif self.cosmo_model == 'base_omegak':
            # piecewise function that calculates the transverse comoving distance for a constant dark energy density cosmology 
            # using sinh and sin based on the samples of Ok
            output_shape = Om.shape[:2] + (z.shape[2],)
            result = torch.zeros(output_shape, device=self.device).flatten(0, 1)
            Om = Om.flatten(0, 1)
            Ok = Ok.flatten(0, 1)
            hrdrag = hrdrag.flatten(0, 1)
            z = z.flatten(0, 1)

            neg_mask = Ok.flatten() < 0 # Ok < 0
            if neg_mask.any():
                result[neg_mask, :] = ((self.c/(100000*hrdrag[neg_mask])/torch.sqrt(-Ok[neg_mask])) * torch.sin(
                    torch.sqrt(-Ok[neg_mask]) * trapezoid(
                        (1 / torch.sqrt(
                            Om[neg_mask].unsqueeze(-1) * (1 + z)**3 + 
                            Ok[neg_mask].unsqueeze(-1) * (1 + z)**2 + 
                            (1 - Om[neg_mask].unsqueeze(-1) - Ok[neg_mask].unsqueeze(-1))
                        )),
                        z,
                        axis=-1
                    )
                ))

            pos_mask = Ok.flatten() > 0 # Ok > 0
            if pos_mask.any():
                result[pos_mask, :] = ((self.c/(100000*hrdrag[pos_mask])/torch.sqrt(Ok[pos_mask])) * torch.sinh(
                    torch.sqrt(Ok[pos_mask]) * trapezoid(
                        (1 / torch.sqrt(
                            Om[pos_mask].unsqueeze(-1) * (1 + z)**3 + 
                            Ok[pos_mask].unsqueeze(-1) * (1 + z)**2 + 
                            (1 - Om[pos_mask].unsqueeze(-1) - Ok[pos_mask].unsqueeze(-1))
                        )),
                        z,
                        axis=-1
                    )
                ))

            zero_mask = Ok.flatten() == 0 # Ok = 0
            if zero_mask.any():
                result[zero_mask, :] = ((self.c/(100000*hrdrag[zero_mask])) * trapezoid(
                    (1 / torch.sqrt(
                        Om[zero_mask].unsqueeze(-1) * (1 + z)**3 + 
                        (1 - Om[zero_mask].unsqueeze(-1))
                    )),
                    z,
                    axis=-1
                ))

            return result.reshape(output_shape)

        elif self.cosmo_model == 'base_w':
            result = (self.c/(100000*hrdrag)) * trapezoid(
                (1 / torch.sqrt(
                    Om.unsqueeze(-1) * (1 + z)**3 + 
                    (1 - Om.unsqueeze(-1)) * (1 + z)**(3 * (1 + w0.unsqueeze(-1)))
                )), 
                z, 
                axis=-1)
            return result
        
        elif self.cosmo_model == 'base_w_wa':
            result = (self.c/(100000*hrdrag)) * trapezoid(
                (1 / torch.sqrt(
                    Om.unsqueeze(-1) * (1 + z)**3 + 
                    (1 - Om.unsqueeze(-1)) * (1 + z)**(3 * (1 + w0.unsqueeze(-1) + wa.unsqueeze(-1))) * 
                    torch.exp(-3 * wa.unsqueeze(-1) * (z / (1 + z)))
                    )), 
                z, 
                axis=-1)
            return result
        
        elif self.cosmo_model == 'base_omegak_w_wa':
            # piecewise function that calculates the transverse comoving distance for a w0 and wa cosmology 
            # using sinh and sin based on the samples of Ok 
            output_shape = Om.shape[:2] + (z.shape[2],)
            result = torch.zeros(output_shape, device=self.device).flatten(0, 1)
            Om = Om.flatten(0, 1)
            Ok = Ok.flatten(0, 1)
            w0 = w0.flatten(0, 1)
            wa = wa.flatten(0, 1)
            hrdrag = hrdrag.flatten(0, 1)
            z = z.flatten(0, 1)

            neg_mask = Ok.flatten() < 0 # Ok < 0
            if neg_mask.any():
                result[neg_mask, :] = ((self.c/(100000*hrdrag[neg_mask])/torch.sqrt(-Ok[neg_mask])) * torch.sin(
                    torch.sqrt(-Ok[neg_mask]) * trapezoid(
                        (1 / torch.sqrt(
                            Om[neg_mask].unsqueeze(-1) * (1 + z)**3 + 
                            Ok[neg_mask].unsqueeze(-1) * (1 + z)**2 + 
                            (1 - Om[neg_mask].unsqueeze(-1) - Ok[neg_mask].unsqueeze(-1)) * 
                            (1 + z)**(3 * (1 + w0[neg_mask].unsqueeze(-1) + wa[neg_mask].unsqueeze(-1))) * 
                            torch.exp(-3 * wa[neg_mask].unsqueeze(-1) * (z / (1 + z)))
                        )),
                        z,
                        axis=-1
                    )
                ))

            pos_mask = Ok.flatten() > 0 # Ok > 0
            if pos_mask.any():
                result[pos_mask, :] = ((self.c/(100000*hrdrag[pos_mask])/torch.sqrt(Ok[pos_mask])) * torch.sinh(
                    torch.sqrt(Ok[pos_mask]) * trapezoid(
                        (1 / torch.sqrt(
                            Om[pos_mask].unsqueeze(-1) * (1 + z)**3 + 
                            Ok[pos_mask].unsqueeze(-1) * (1 + z)**2 + 
                            (1 - Om[pos_mask].unsqueeze(-1) - Ok[pos_mask].unsqueeze(-1)) * 
                            (1 + z)**(3 * (1 + w0[pos_mask].unsqueeze(-1) + wa[pos_mask].unsqueeze(-1))) * 
                            torch.exp(-3 * wa[pos_mask].unsqueeze(-1) * (z / (1 + z)))
                        )),
                        z,
                        axis=-1
                    )
                ))

            zero_mask = Ok.flatten() == 0 # Ok = 0
            if zero_mask.any():
                result[zero_mask, :] = ((self.c/(100000*hrdrag[zero_mask])) * trapezoid(
                    (1 / torch.sqrt(
                        Om[zero_mask].unsqueeze(-1) * (1 + z)**3 + 
                        (1 - Om[zero_mask].unsqueeze(-1)) *
                        (1 + z)**(3 * (1 + w0[zero_mask].unsqueeze(-1) + wa[zero_mask].unsqueeze(-1))) * 
                        torch.exp(-3 * wa[zero_mask].unsqueeze(-1) * (z / (1 + z)))
                    )),
                    z,
                    axis=-1
                ))

            return result.reshape(output_shape)

    def D_V_func(self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None):
        """
        The angle-averaged distance D_V/r_d = (z * (D_M/r_d)^2 * (D_H/r_d))^(1/3) divided by the sound horizon D_V/r_d
        """
        return (z_eff.reshape((len(self.cosmo_params))*[1] + [-1]) * self.D_M_func(z_eff, Om, Ok, w0, wa, hrdrag)**2 * self.D_H_func(z_eff, Om, Ok, w0, wa, hrdrag))**(1/3)

    def sample_guide(self, tracer_ratio, guide, num_data_samples=100, num_param_samples=1000, central=True):

        if central:
            rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
            passed_ratio = self.calc_passed(tracer_ratio)
            means = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            means[:, :, self.DH_idx] = lexpand(self.central_val[self.DH_idx].unsqueeze(0), num_data_samples)
            rescaled_sigmas = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt(self.nominal_passed_ratio[self.DH_idx]/passed_ratio[..., self.DH_idx])
            if self.include_D_M:
                means[:, :, self.DM_idx] = lexpand(self.central_val[self.DM_idx].unsqueeze(0), num_data_samples)
                rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt(self.nominal_passed_ratio[self.DM_idx]/passed_ratio[..., self.DM_idx])
            if self.include_D_V:
                means[:, :, self.DV_idx] = lexpand(self.central_val[self.DV_idx].unsqueeze(0), num_data_samples)
                rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[self.DV_idx] * torch.sqrt(self.nominal_passed_ratio[self.DV_idx]/passed_ratio[..., self.DV_idx])

            if self.include_D_V and self.include_D_M:
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1) * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2))

            with pyro.plate("data", num_data_samples):
                data_samples = pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means.squeeze(), covariance_matrix.squeeze())).unsqueeze(1)
        else:
            data_samples = self.pyro_model(tracer_ratio)
        context = torch.cat([tracer_ratio, data_samples], dim=-1)
        # Sample parameters conditioned on the data batch
        param_samples = guide(context.squeeze()).sample((num_param_samples,))
        param_samples[:, :, -1] *= 100000
        return param_samples

    def sample_brute_force(self, tracer_ratio, grid_designs, grid_features, grid_params, designer, num_data_samples=100, num_param_samples=1000):
        
        rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
        rescaled_sigmas[self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt((self.efficiency[self.DH_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DH_idx])
        if self.include_D_M:
            means = self.central_val
            rescaled_sigmas[self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt((self.efficiency[self.DM_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DM_idx])
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            means = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[self.DH_idx].unsqueeze(-1) * rescaled_sigmas[self.DH_idx].unsqueeze(-2))

        with pyro.plate("data", num_data_samples):
            data_samples = pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix))
            
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
        rescaled_sigmas[..., self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt((self.efficiency[self.DH_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DH_idx])
        if self.include_D_M:
            y = self.central_val
            rescaled_sigmas[..., self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt((self.efficiency[self.DM_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DM_idx])
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            y = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[..., self.DH_idx].unsqueeze(-1) * rescaled_sigmas[..., self.DH_idx].unsqueeze(-2))
        with GridStack(grid_params):
            parameters = {k: torch.tensor(getattr(grid_params, k), device=self.device).unsqueeze(-1) for k in grid_params.names}
        mean = torch.zeros(grid_params.shape + (len(self.sigmas),), device=self.device)
        mean[..., self.DH_idx] = self.D_H_func(**parameters)

        if self.include_D_M:
            mean[..., self.DM_idx] = self.D_M_func(**parameters)
        else:
            mean = mean[..., self.DH_idx]

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

    def sample_valid_parameters(self, samples_shape):
        parameters = {}
        # draw samples from priors with the correct shape
        for k, v in self.priors.items():
            if isinstance(v, dist.Distribution):
                parameters[k] = v.sample((samples_shape[0], samples_shape[1]))  # No unsqueeze
            else:
                parameters[k] = v
        # check constraint (w0 + wa) < 0 and re-sample if necessary
        if "w0" in parameters and "wa" in parameters:
            # Compute the constraint for each sample
            constraint_satisfied = (parameters["w0"] + parameters["wa"]) < 0
            invalid_indices = ~constraint_satisfied

            # re-sample until all invalid samples are replaced
            while invalid_indices.any():
                for i in range(invalid_indices.shape[0]):
                    row_invalid = invalid_indices[i]  # shape: (samples_shape[1],)
                    invalid_count = row_invalid.sum().item()
                    if invalid_count > 0:
                        # replace only the invalid samples in row i
                        parameters["w0"][i, row_invalid] = self.priors["w0"].sample((invalid_count,))
                        parameters["wa"][i, row_invalid] = self.priors["wa"].sample((invalid_count,))
                # re-check the constraint after replacement
                constraint_satisfied = (parameters["w0"] + parameters["wa"]) < 0
                invalid_indices = ~constraint_satisfied
        
        if "Ok" in parameters:
            # Compute the constraint for each sample
            constraint_satisfied = ((parameters["Om"] + parameters["Ok"]) < 1) & ((parameters["Om"] + parameters["Ok"]) > 0)
            invalid_indices = ~constraint_satisfied

            # re-sample until all invalid samples are replaced
            while invalid_indices.any():
                for i in range(invalid_indices.shape[0]):
                    row_invalid = invalid_indices[i]
                    invalid_count = row_invalid.sum().item()
                    if invalid_count > 0:
                        parameters["Om"][i, row_invalid] = self.priors["Om"].sample((invalid_count,))
                        parameters["Ok"][i, row_invalid] = self.priors["Ok"].sample((invalid_count,))
                constraint_satisfied = ((parameters["Om"] + parameters["Ok"]) < 1) & ((parameters["Om"] + parameters["Ok"]) > 0)
                invalid_indices = ~constraint_satisfied

        return parameters

    def pyro_model(self, tracer_ratio):
        passed_ratio = self.calc_passed(tracer_ratio)
        constrained_parameters = self.sample_valid_parameters(passed_ratio.shape[:-1])
        with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
            # register samples in the trace using pyro.sample
            parameters = {}
            for k, v in constrained_parameters.items():
                # use dist.Delta to fix the value of each parameter
                parameters[k] = pyro.sample(k, dist.Delta(v)).unsqueeze(-1)
            means = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DH_over_rs"]["z"].to_list(), device=self.device)
            means[:, :, self.DH_idx] = self.D_H_func(z_eff, **parameters)
            rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt(self.nominal_passed_ratio[self.DH_idx]/passed_ratio[..., self.DH_idx])
            if self.include_D_M:
                z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DM_over_rs"]["z"].to_list(), device=self.device)
                means[:, :, self.DM_idx] = self.D_M_func(z_eff, **parameters)
                rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt(self.nominal_passed_ratio[self.DM_idx]/passed_ratio[..., self.DM_idx])

            if self.include_D_V:
                z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DV_over_rs"]["z"].to_list(), device=self.device)
                means[:, :, self.DV_idx] = self.D_V_func(z_eff, **parameters)
                rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[self.DV_idx] * torch.sqrt(self.nominal_passed_ratio[self.DV_idx]/passed_ratio[..., self.DV_idx])

            # extract correlation matrix from DESI covariance matrix
            if self.include_D_M:
                means = means.to(self.device)
                # convert correlation matrix to covariance matrix using rescaled sigmas
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                # only use D_H values for mean and covariance matrix
                means = means[:, :, self.DH_idx].to(self.device)
                covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1) * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2))

            return pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix))

    def unnorm_lfunc(self, params, features, designs):
        parameters = { }
        for key in params.names:
            parameters[key] = torch.tensor(getattr(params, key), device=self.device)
        likelihood = 1
        passed_ratio = self.calc_passed(designs)
        for i in range(len(self.tracer_bins)):
            D_H_mean = self.D_H_func(**parameters)
            D_H_diff = getattr(features, features.names[i]) - D_H_mean.cpu().numpy()
            print(getattr(designs, designs.names[i]).shape)
            D_H_sigma = self.sigmas[self.DH_idx].cpu().numpy()[i] * np.sqrt(self.nominal_passed_ratio[i].cpu().numpy()/passed_ratio[i])
            likelihood = np.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood
            
            if self.include_D_M:
                D_M_mean = self.D_M_func(**parameters)
                D_M_diff = getattr(features, features.names[i+len(self.tracer_bins)]) - D_M_mean.cpu().numpy()
                D_M_sigma = self.sigmas[self.DM_idx].cpu().numpy()[i] * np.sqrt(self.nominal_passed_ratio[i].cpu().numpy()/passed_ratio[i])
                likelihood = np.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        return likelihood
