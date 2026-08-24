"""
Quick diagnostic to check what's happening during training with prior_flow.
"""
import os
import torch
import yaml
from bedcosmo.util import init_experiment, auto_seed, get_experiment_config_path

auto_seed(1)

# Load args
with open(get_experiment_config_path('num_tracers', 'prior_args_posterior_base_dr2.yaml'), 'r') as f:
    prior_args = yaml.safe_load(f)
with open(get_experiment_config_path('num_tracers', 'design_args_dr2.yaml'), 'r') as f:
    design_args = yaml.safe_load(f)

print("Initializing experiment...")
experiment = init_experiment(
    cosmo_exp='num_tracers',
    prior_args=prior_args,
    design_args=design_args,
    cosmo_model='base_omegak_w_wa',
    dataset='dr2',
    device='cuda:0',
    global_rank=0,
    transform_input=True,
)

print("\n=== Testing Training Path ===\n")

# Simulate what happens during training
from bedcosmo.pyro_oed_src import _create_condition_input
import pyro.poutine as poutine
from pyro.contrib.util import lexpand

# Get a batch of designs - use ALL designs like training does
designs = experiment.designs  # All 209 designs
n_particles = 10  # Match training

print(f"Designs shape: {designs.shape}")
print(f"N particles: {n_particles}")

# Expand designs for sampling
expanded_design = lexpand(designs, n_particles)
print(f"Expanded design shape: {expanded_design.shape}")

# Sample from pyro_model (this uses prior_flow)
print("\nSampling from pyro_model (uses prior_flow internally)...")
with torch.no_grad():
    trace = poutine.trace(experiment.pyro_model).get_trace(expanded_design)
    y_dict = {l: trace.nodes[l]["value"] for l in experiment.observation_labels}
    theta_dict = {l: trace.nodes[l]["value"] for l in experiment.cosmo_params}

# Check samples
print("\n=== Parameter Samples (Physical Space) ===")
samples_list = [theta_dict[p] for p in experiment.cosmo_params]
samples = torch.stack(samples_list, dim=-1).squeeze(-2)
print(f"Samples shape: {samples.shape}")
for i, p in enumerate(experiment.cosmo_params):
    vals = samples[..., i]
    print(f"  {p}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
    nan_count = torch.isnan(vals).sum().item()
    inf_count = torch.isinf(vals).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"    WARNING: {nan_count} NaN, {inf_count} inf")

# Check observations
print("\n=== Observations (from likelihood) ===")
for l in experiment.observation_labels:
    obs = y_dict[l]
    print(f"  {l} shape: {obs.shape}")
    nan_count = torch.isnan(obs).sum().item()
    inf_count = torch.isinf(obs).sum().item()
    if torch.all(torch.isnan(obs)):
        print(f"    All values are NaN!")
    else:
        valid = obs[~torch.isnan(obs)]
        print(f"    min={valid.min().item():.4f}, max={valid.max().item():.4f}")
    if nan_count > 0:
        print(f"    WARNING: {nan_count} NaN values!")
    if inf_count > 0:
        print(f"    WARNING: {inf_count} inf values!")

# Create condition input
condition_input = _create_condition_input(
    expanded_design, y_dict, experiment.observation_labels, condition_design=True
)
print(f"\nCondition input shape: {condition_input.shape}")
nan_count = torch.isnan(condition_input).sum().item()
inf_count = torch.isinf(condition_input).sum().item()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN in condition_input!")
if inf_count > 0:
    print(f"  WARNING: {inf_count} inf in condition_input!")

# Transform samples for flow
print("\n=== Transformed Samples (for flow input) ===")
if experiment.transform_input:
    y_flat = experiment.params_to_unconstrained(samples.view(-1, samples.shape[-1]))
    print(f"Transformed shape: {y_flat.shape}")
    for i, p in enumerate(experiment.cosmo_params):
        vals = y_flat[..., i]
        print(f"  {p}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
        nan_count = torch.isnan(vals).sum().item()
        inf_count = torch.isinf(vals).sum().item()
        if nan_count > 0:
            print(f"    WARNING: {nan_count} NaN")
        if inf_count > 0:
            print(f"    WARNING: {inf_count} inf")
        # Check for extreme values
        extreme_count = ((vals.abs() > 10).sum()).item()
        if extreme_count > 0:
            print(f"    WARNING: {extreme_count} values with |x| > 10")

print("\n=== Testing nf_loss ===")

# Initialize a flow (like in training)
from util import init_nf
run_args = {
    'flow_type': 'NAF',
    'activation': 'elu',
    'n_transforms': 6,
    'cond_hidden_size': 512,
    'cond_n_layers': 8,
    'mnn_hidden_size': 512,
    'mnn_n_layers': 6,
    'mnn_signal': 64,
    'nf_transform': 'affine',
}
input_dim = len(experiment.cosmo_params)
context_dim = experiment.context_dim

guide = init_nf(run_args, input_dim, context_dim, device='cuda:0', seed=1)
guide.eval()

# Compute nf_loss
from pyro_oed_src import nf_loss
agg_loss, loss = nf_loss(
    samples=samples,
    context=condition_input,
    guide=guide,
    experiment=experiment,
    rank=0,
    verbose_shapes=True
)

print(f"\nAggregate loss: {agg_loss.item()}")
print(f"Loss shape: {loss.shape}")
print(f"Loss stats: min={loss.min().item():.4f}, max={loss.max().item():.4f}, mean={loss.mean().item():.4f}")

# Check for extreme values in loss
extreme_loss = (loss.abs() > 1e10).sum().item()
if extreme_loss > 0:
    print(f"  WARNING: {extreme_loss} loss values with |loss| > 1e10!")
    
print("\n=== Done ===")
