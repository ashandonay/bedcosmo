"""Conditional mixed posterior on the same pattern/face/atom measure as the prior."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import zuko
from torch import nn
from torch.distributions import Categorical, Distribution

from .joint_prior_flow import build_flow


def pattern_prior_spec(prior):
    """Architecture and constraint constants needed to rebuild from a state dict."""
    names = ["patterns", "counts", "atomic", "atomic_rows", "lower", "upper", "flow_config"]
    spec = {name: prior.metadata[name] for name in names}
    spec = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in spec.items()
    }
    spec["groups"] = {
        str(m): {name: group[name].tolist() for name in ["mu", "std"]}
        for m, group in prior.groups.items()
    }
    return spec


class PatternPosterior(nn.Module):
    """Learn q(mask|photometry,design) and q(Gaussian coords|mask,photometry,design).

    Frozen prior transforms handle the face geometry and Gaussianization. Their
    weights are checkpoint buffers/parameters, so reconstruction has no external
    prior-file dependency. Public samples and scores remain in physical space.
    """

    def __init__(
        self, spec, context_dim, *, hidden=128, layers=3, transforms=4, bins=16, prior=None
    ):
        super().__init__()
        self.context_dim = context_dim
        self.register_buffer("patterns", torch.tensor(spec["patterns"], dtype=torch.bool))
        self.k = self.patterns.shape[1]
        self.register_buffer("atomic", torch.tensor(spec["atomic"], dtype=torch.bool))
        self.register_buffer("atomic_rows", torch.tensor(spec["atomic_rows"], dtype=torch.float64))
        self.register_buffer("lower", torch.tensor(spec["lower"], dtype=torch.float64))
        self.register_buffer("upper", torch.tensor(spec["upper"], dtype=torch.float64))
        counts = torch.tensor(spec["counts"], dtype=torch.float64)
        self.register_buffer("log_prior_mass", (counts / counts.sum()).log())
        self.register_buffer("code_weights", 2 ** torch.arange(self.k, dtype=torch.int64))
        codes = (self.patterns.long() * self.code_weights).sum(-1)
        order = codes.argsort()
        self.register_buffer("sorted_codes", codes[order])
        self.register_buffer("pattern_order", order)
        if context_dim % 2:
            raise ValueError("Pattern posterior expects visits followed by magnitudes")
        bands = context_dim // 2
        self.register_buffer("context_loc", torch.tensor([0.0] * bands + [20.0] * bands))
        self.register_buffer("context_scale", torch.tensor([200.0] * bands + [5.0] * bands))
        classifier = []
        width = context_dim
        for _ in range(layers):
            classifier.extend([nn.Linear(width, hidden), nn.ELU()])
            width = hidden
        classifier.append(nn.Linear(width, len(counts)))
        self.classifier = nn.Sequential(*classifier).double()
        with torch.no_grad():
            self.classifier[-1].weight.zero_()
            self.classifier[-1].bias.copy_(self.log_prior_mass)
        self.face_flows = nn.ModuleDict()
        self.prior_flows = nn.ModuleDict()
        for key, saved in spec["groups"].items():
            m = int(key)
            self.register_buffer(f"mu_{m}", torch.tensor(saved["mu"], dtype=torch.float64))
            self.register_buffer(f"std_{m}", torch.tensor(saved["std"], dtype=torch.float64))
            frozen = (
                build_flow(m + 1, self.k, spec["flow_config"])
                if prior is None
                else copy.deepcopy(prior.groups[m]["flow"])
            )
            self.prior_flows[key] = frozen.requires_grad_(False)
            self.face_flows[key] = zuko.flows.NSF(
                features=m + 1,
                context=context_dim + self.k,
                transforms=transforms,
                bins=bins,
                hidden_features=(hidden,) * layers,
                activation=nn.ELU,
                randperm=True,
            ).double()

    def forward(self, context):
        normalized = (context - self.context_loc) / self.context_scale
        return PatternPosteriorDistribution(self, normalized, self.classifier(normalized))

    def pattern_ids(self, rows):
        codes = ((rows[..., : self.k] > 0).long() * self.code_weights).sum(-1)
        index = torch.searchsorted(self.sorted_codes, codes).clamp_max(len(self.patterns) - 1)
        return self.pattern_order[index], self.sorted_codes[index] == codes

    def to_gaussian(self, rows, masks, m):
        """Physical face -> saved prior Gaussian base, with its physical Jacobian."""
        active = rows[:, : self.k][masks].reshape(-1, m)
        tails = active.flip(-1).cumsum(-1).flip(-1)
        stick = active[:, :-1].log() - tails[:, 1:].log()
        left, right = rows[:, -2:] - self.lower, self.upper - rows[:, -2:]
        features = torch.cat([stick, left.log() - right.log()], -1)
        std, mu = getattr(self, f"std_{m}"), getattr(self, f"mu_{m}")
        z, ladj = self.prior_flows[str(m)](masks.float()).transform.call_and_ladj(
            ((features - mu) / std).float()
        )
        jac = -active.log().sum(-1) + (
            (self.upper - self.lower).log() - left.log() - right.log()
        ).sum(-1)
        return z.double(), jac - std.log().sum() + ladj.double()

    def from_gaussian(self, z, masks, m):
        x = self.prior_flows[str(m)](masks.float()).transform.inv(z.float()).double()
        features = x * getattr(self, f"std_{m}") + getattr(self, f"mu_{m}")
        stick = features[:, : m - 1]
        log_v = -nn.functional.softplus(-stick)
        log_remain = -nn.functional.softplus(stick)
        prefix = torch.cat([stick.new_zeros((len(stick), 1)), log_remain.cumsum(-1)], -1)
        active = torch.cat([prefix[:, :-1] + log_v, prefix[:, -1:]], -1).exp()
        active = active / active.sum(-1, keepdim=True)
        rows = z.new_zeros((len(z), self.k + 2))
        rows[:, : self.k][masks] = active.flatten()
        rows[:, -2:] = self.lower + (self.upper - self.lower) * features[:, -2:].sigmoid()
        if (
            (active <= 0).any()
            or (rows[:, -2:] <= self.lower).any()
            or (rows[:, -2:] >= self.upper).any()
        ):
            raise ValueError("Posterior tail rounded outside the open continuous face")
        return rows


class PatternPosteriorDistribution(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, model, context, logits):
        self.model, self.context, self.logits = model, context, logits
        super().__init__(context.shape[:-1], (model.k + 2,), validate_args=False)

    def log_prob(self, value):
        model = self.model
        shape = torch.broadcast_shapes(value.shape[:-1], self.batch_shape)
        rows = value.expand(*shape, model.k + 2).reshape(-1, model.k + 2)
        context = self.context.expand(*shape, model.context_dim).reshape(-1, model.context_dim)
        logits = self.logits.expand(*shape, len(model.patterns)).reshape(-1, len(model.patterns))
        ids, known = model.pattern_ids(rows)
        masks = rows[:, : model.k] > 0
        valid = known & torch.isfinite(rows).all(-1) & (rows[:, : model.k] >= 0).all(-1)
        valid &= torch.isclose(rows[:, : model.k].sum(-1), rows.new_tensor(1.0))
        atoms = valid & model.atomic[ids] & (rows == model.atomic_rows[ids]).all(-1)
        continuous = (
            valid
            & ~model.atomic[ids]
            & ((rows[:, -2:] > model.lower) & (rows[:, -2:] < model.upper)).all(-1)
        )
        log_mass = logits.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1)
        scores = rows.new_full((len(rows),), -math.inf)
        scores[atoms] = log_mass[atoms]
        zero = rows.new_zeros(())
        for key, flow in model.face_flows.items():
            m = int(key)
            selected = continuous & (masks.sum(-1) == m)
            if selected.any():
                z, jac = model.to_gaussian(rows[selected], masks[selected], m)
                ctx = torch.cat([context[selected], masks[selected].double()], -1)
                scores[selected] = log_mass[selected] + flow(ctx).log_prob(z) + jac
            else:
                # DDP needs every trainable branch in the graph, including a
                # rare active-count group absent from this rank's batch.
                dummy = context.new_zeros((1, model.context_dim + model.k))
                zero = zero + 0 * flow(dummy).log_prob(context.new_zeros((1, m + 1))).sum()
        return (scores + zero).reshape(shape)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        model = self.model
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        ids = Categorical(logits=self.logits).sample(sample_shape).reshape(-1)
        context = self.context.expand(*shape, model.context_dim).reshape(-1, model.context_dim)
        masks = model.patterns[ids]
        atomic = model.atomic[ids]
        rows = context.new_empty((len(ids), model.k + 2))
        rows[atomic] = model.atomic_rows[ids[atomic]]
        for key, flow in model.face_flows.items():
            m = int(key)
            selected = (~atomic) & (masks.sum(-1) == m)
            if not selected.any():
                continue
            ctx = torch.cat([context[selected], masks[selected].double()], -1)
            z = flow(ctx).sample()
            rows[selected] = model.from_gaussian(z, masks[selected], m)
        return rows.reshape(*shape, model.k + 2)
