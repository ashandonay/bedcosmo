import math
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
import matplotlib.pyplot as plt

class ConstrainedUniform2D(TorchDistribution):
    """
    Uniform over the region defined by:
      • x[var] ∈ [priors[var].low, priors[var].high] for each var
      • lower < sum(x[var] for var in vars) < upper
    Currently supports exactly 2 variables.

    Args:
        priors (dict): mapping var-name to pyro.distributions.Uniform(low, high)
        lower (float): lower bound on the sum of the variables
        upper (float): upper bound on the sum of the variables
    """
    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, priors, lower=None, upper=None):
        # check keys
        assert len(priors) == 2, "Only 2 variables supported"
        self.vars = list(priors)
        self.priors = priors
        self.lower = float(lower) if lower is not None else None
        self.upper = float(upper) if upper is not None else None

        # Get device from priors
        self.device = next(iter(priors.values())).low.device

        # extract bounds
        self.bounds = {v: (float(priors[v].low), float(priors[v].high)) for v in self.vars}

        # alias for readability
        self.v0, self.v1 = self.vars
        amin, amax = self.bounds[self.v0]
        bmin, bmax = self.bounds[self.v1]
        a_range = amax - amin
        b_range = bmax - bmin

        # compute all intercepts of the lines a+b=lower and a+b=upper with the square edges
        Uth = [0.0]
        Vth = [0.0]
        # lower bound line intercepts
        if self.lower is not None:
            c_lo = self.lower - (amin + bmin)
            # with v=0 and v=1
            u_vals = torch.tensor([c_lo / a_range, (c_lo - b_range) / a_range], device=self.device)
            u_mask = (u_vals > 0) & (u_vals < 1)
            Uth.extend(u_vals[u_mask].tolist())
            # with u=0 and u=1
            v_vals = torch.tensor([c_lo / b_range, (c_lo - a_range) / b_range], device=self.device)
            v_mask = (v_vals > 0) & (v_vals < 1)
            Vth.extend(v_vals[v_mask].tolist())
        # upper bound line intercepts
        if self.upper is not None:
            c_hi = self.upper - (amin + bmin)
            u_vals = torch.tensor([c_hi / a_range, (c_hi - b_range) / a_range], device=self.device)
            u_mask = (u_vals > 0) & (u_vals < 1)
            Uth.extend(u_vals[u_mask].tolist())
            v_vals = torch.tensor([c_hi / b_range, (c_hi - a_range) / b_range], device=self.device)
            v_mask = (v_vals > 0) & (v_vals < 1)
            Vth.extend(v_vals[v_mask].tolist())
        Uth.append(1.0)
        Vth.append(1.0)
        # merge near-duplicates and sort
        Uth = sorted(set([round(u,8) for u in Uth]))
        Vth = sorted(set([round(v,8) for v in Vth]))

        # build piece boundaries without Python loops using meshgrid
        # thresholds are Uth, Vth already defined lists
        U0 = torch.tensor(Uth[:-1], dtype=torch.float32, device=self.device)
        U1 = torch.tensor(Uth[1:], dtype=torch.float32, device=self.device)
        V0 = torch.tensor(Vth[:-1], dtype=torch.float32, device=self.device)
        V1 = torch.tensor(Vth[1:], dtype=torch.float32, device=self.device)

        # create grid of cell bounds
        # u0g, v0g: lower edges; u1g, v1g: upper edges for each cell
        u0g, v0g = torch.meshgrid(U0, V0, indexing='ij')
        u1g, v1g = torch.meshgrid(U1, V1, indexing='ij')

        # flatten into cell lists
        piece_u = torch.stack((u0g, u1g), dim=-1).reshape(-1, 2)
        piece_v = torch.stack((v0g, v1g), dim=-1).reshape(-1, 2)

        # compute cell areas in (u,v)
        du = piece_u[:, 1] - piece_u[:, 0]
        dv = piece_v[:, 1] - piece_v[:, 0]
        areas_uv = du * dv

        # half-area for corner triangles if bounds exist
        if self.lower is not None:
            areas_uv[0] *= 0.5
        if self.upper is not None:
            areas_uv[-1] *= 0.5

        # store piece boundary tensors
        self.piece_u = piece_u
        self.piece_v = piece_v

        # compute actual areas and mixture weights
        areas = areas_uv * (a_range * b_range)
        self.probs = areas / areas.sum()
        self.cat = dist.Categorical(self.probs)
        self.areas = areas

        # types list for triangle masks
        n_cells = piece_u.shape[0]
        types = ['rect'] * n_cells
        if self.lower is not None:
            types[0] = 'tri_lo'
        if self.upper is not None:
            types[-1] = 'tri_hi'
        self.types = types

        # keep ranges for mapping back
        self.a_range = a_range
        self.b_range = b_range
        self.amin = amin
        self.bmin = bmin

        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([2]))
    
    def rsample(self, sample_shape=torch.Size()):
        piece = self.cat.sample(sample_shape)
        
        # Handle multi-dimensional sampling
        if len(sample_shape) > 1:
            # Flatten the piece tensor for processing
            piece_flat = piece.flatten()
            n_samples = piece_flat.shape[0]
        else:
            piece_flat = piece
            n_samples = piece.shape[0] if sample_shape else 1

        # Expand piece_u and piece_v to sample_shape
        piece_u = self.piece_u.unsqueeze(0).expand(n_samples, -1, -1)  # (n_samples, n_pieces, 2)
        piece_v = self.piece_v.unsqueeze(0).expand(n_samples, -1, -1)  # (n_samples, n_pieces, 2)

        # Sample uniform variables for all pieces
        U = dist.Uniform(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).rsample((n_samples, len(self.probs)))
        V = dist.Uniform(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)).rsample((n_samples, len(self.probs)))

        # Compute sampled points in normalized coordinates
        u0 = piece_u[..., 0]
        u1 = piece_u[..., 1]
        v0 = piece_v[..., 0]
        v1 = piece_v[..., 1]

        # Determine mask for triangles
        n_pieces = len(self.probs)
        mask_lo = torch.zeros(n_pieces, dtype=torch.bool, device=self.device)
        mask_hi = torch.zeros(n_pieces, dtype=torch.bool, device=self.device)
        
        if self.lower is not None:
            mask_lo[0] = True
        if self.upper is not None:
            mask_hi[-1] = True
            
        mask_lo = mask_lo.unsqueeze(0).expand_as(U)
        mask_hi = mask_hi.unsqueeze(0).expand_as(U)

        # For the lower-bound triangle: reflect those in the forbidden lower-left corner (U+V<1)
        U_lo = U.clone()
        V_lo = V.clone()
        mask_lo_reflect = mask_lo & (U + V < 1.0)
        U_lo[mask_lo_reflect] = 1.0 - U[mask_lo_reflect]
        V_lo[mask_lo_reflect] = 1.0 - V[mask_lo_reflect]

        # For the upper-bound triangle: reflect those in the forbidden upper-right corner (U+V>1)
        U_hi = U.clone()
        V_hi = V.clone()
        mask_hi_reflect = mask_hi & (U + V > 1.0)
        U_hi[mask_hi_reflect] = 1.0 - U[mask_hi_reflect]
        V_hi[mask_hi_reflect] = 1.0 - V[mask_hi_reflect]

        # Combine U,V based on type
        U_combined = torch.where(mask_lo, U_lo, U)
        U_combined = torch.where(mask_hi, U_hi, U_combined)
        V_combined = torch.where(mask_lo, V_lo, V)
        V_combined = torch.where(mask_hi, V_hi, V_combined)

        # Compute samples in normalized space
        samples_u = u0 + U_combined * (u1 - u0)
        samples_v = v0 + V_combined * (v1 - v0)

        # Select samples according to piece indices
        idx = piece_flat.long().unsqueeze(-1)
        sx_u = samples_u.gather(-1, idx).squeeze(-1)
        sx_v = samples_v.gather(-1, idx).squeeze(-1)

        # Convert normalized samples to original space
        x0 = self.amin + sx_u * self.a_range
        x1 = self.bmin + sx_v * self.b_range

        # Reshape back to original sample shape if needed
        result = torch.stack([x0, x1], dim=-1)
        if len(sample_shape) > 1:
            result = result.view(*sample_shape, -1)
            
        return result

    def log_prob(self, value):
        x0, x1 = value.unbind(-1)
        v0, v1 = self.vars
        lo0, hi0 = self.bounds[v0]
        lo1, hi1 = self.bounds[v1]
        # apply box & linear constraints
        valid = (
            (x0 >= lo0) & (x0 <= hi0) &
            (x1 >= lo1) & (x1 <= hi1)
        )
        if self.lower is not None:
            valid = valid & ((x0 + x1) > self.lower)
        if self.upper is not None:
            valid = valid & ((x0 + x1) < self.upper)

        total_area = float(self.areas.sum())
        logp = torch.full_like(x0, -float('inf'))
        logp[valid] = - math.log(total_area)
        return logp