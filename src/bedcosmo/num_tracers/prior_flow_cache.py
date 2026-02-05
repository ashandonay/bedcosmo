#!/usr/bin/env python
"""
Diagnostic script to compare cached vs non-cached prior_flow sampling.

Visualizes sample distributions over multiple steps and reports timing.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir)

from bedcosmo.util import init_experiment
import yaml


def sample_over_steps(experiment, n_steps, sample_shape, use_cache):
    """
    Collect samples over multiple steps, simulating training loop behavior.
    
    Returns:
        all_samples: List of sample tensors per step
        step_times: List of time per step
        total_time: Total time for all steps
    """
    # Set cache mode
    experiment.prior_flow_use_cache = use_cache
    
    # Reset cache if using cache mode
    if use_cache and hasattr(experiment, '_prior_flow_cache'):
        experiment._prior_flow_cache = None
        experiment._prior_flow_cache_idx = 0
    
    all_samples = []
    step_times = []
    
    start_total = time.time()
    
    for step in range(n_steps):
        start_step = time.time()
        
        # Sample like in training (raw samples, no pyro registration for speed)
        samples = experiment._sample_from_prior_flow(sample_shape)
        
        step_time = time.time() - start_step
        step_times.append(step_time)
        all_samples.append(samples.cpu().numpy())
        
        if step % 10 == 0:
            print(f"  Step {step}: {step_time:.4f}s")
    
    total_time = time.time() - start_total
    
    return all_samples, step_times, total_time


def plot_comparison(cached_samples, nocache_samples, param_names, output_path):
    """
    Create comparison plots of cached vs non-cached samples.
    """
    n_params = len(param_names)
    n_steps = len(cached_samples)
    
    # Flatten samples for overall distribution comparison
    cached_flat = np.concatenate([s.reshape(-1, n_params) for s in cached_samples], axis=0)
    nocache_flat = np.concatenate([s.reshape(-1, n_params) for s in nocache_samples], axis=0)
    
    fig, axes = plt.subplots(3, n_params, figsize=(4*n_params, 12))
    
    # Row 1: Overall distribution comparison (histograms)
    for i, param in enumerate(param_names):
        ax = axes[0, i]
        ax.hist(cached_flat[:, i], bins=50, alpha=0.5, label='Cached', density=True)
        ax.hist(nocache_flat[:, i], bins=50, alpha=0.5, label='No Cache', density=True)
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{param} Distribution')
    
    # Row 2: Sample mean over steps (to check for drift/artifacts)
    step_indices = np.arange(n_steps)
    for i, param in enumerate(param_names):
        ax = axes[1, i]
        cached_means = [s[..., i].mean() for s in cached_samples]
        nocache_means = [s[..., i].mean() for s in nocache_samples]
        ax.plot(step_indices, cached_means, 'b-', alpha=0.7, label='Cached')
        ax.plot(step_indices, nocache_means, 'r-', alpha=0.7, label='No Cache')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean')
        ax.legend()
        ax.set_title(f'{param} Mean Over Steps')
    
    # Row 3: Sample std over steps
    for i, param in enumerate(param_names):
        ax = axes[2, i]
        cached_stds = [s[..., i].std() for s in cached_samples]
        nocache_stds = [s[..., i].std() for s in nocache_samples]
        ax.plot(step_indices, cached_stds, 'b-', alpha=0.7, label='Cached')
        ax.plot(step_indices, nocache_stds, 'r-', alpha=0.7, label='No Cache')
        ax.set_xlabel('Step')
        ax.set_ylabel('Std')
        ax.legend()
        ax.set_title(f'{param} Std Over Steps')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_sample_evolution(cached_samples, nocache_samples, param_names, output_path):
    """
    Plot how individual sample values evolve over steps (first few samples).
    """
    n_params = len(param_names)
    n_steps = len(cached_samples)
    n_traces = 5  # Number of sample traces to show
    
    fig, axes = plt.subplots(2, n_params, figsize=(4*n_params, 8))
    
    step_indices = np.arange(n_steps)
    
    for i, param in enumerate(param_names):
        # Cached samples - track specific indices across shuffles
        ax = axes[0, i]
        for trace in range(n_traces):
            values = [s.reshape(-1, n_params)[trace, i] for s in cached_samples]
            ax.plot(step_indices, values, alpha=0.7, label=f'Sample {trace}')
        ax.set_xlabel('Step')
        ax.set_ylabel(param)
        ax.set_title(f'{param} - Cached (trace {n_traces} samples)')
        
        # No cache samples
        ax = axes[1, i]
        for trace in range(n_traces):
            values = [s.reshape(-1, n_params)[trace, i] for s in nocache_samples]
            ax.plot(step_indices, values, alpha=0.7, label=f'Sample {trace}')
        ax.set_xlabel('Step')
        ax.set_ylabel(param)
        ax.set_title(f'{param} - No Cache (trace {n_traces} samples)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved evolution plot to {output_path}")
    plt.close()


def print_timing_report(cached_times, nocache_times, cached_total, nocache_total, cache_size):
    """Print timing comparison report."""
    print("\n" + "="*60)
    print("TIMING REPORT")
    print("="*60)
    
    print(f"\nCache size: {cache_size:,}")
    
    print(f"\n{'Metric':<30} {'Cached':>12} {'No Cache':>12} {'Speedup':>10}")
    print("-"*60)
    
    # Skip first step for cached (includes cache init)
    cached_after_init = cached_times[1:] if len(cached_times) > 1 else cached_times
    
    print(f"{'Total time (s)':<30} {cached_total:>12.2f} {nocache_total:>12.2f} {nocache_total/cached_total:>10.2f}x")
    print(f"{'First step (s) [incl. init]':<30} {cached_times[0]:>12.4f} {nocache_times[0]:>12.4f} {'-':>10}")
    print(f"{'Mean step time after init (s)':<30} {np.mean(cached_after_init):>12.4f} {np.mean(nocache_times):>12.4f} {np.mean(nocache_times)/np.mean(cached_after_init):>10.0f}x")
    print(f"{'Median step time (s)':<30} {np.median(cached_after_init):>12.4f} {np.median(nocache_times):>12.4f} {np.median(nocache_times)/np.median(cached_after_init):>10.0f}x")
    print(f"{'Min step time (s)':<30} {np.min(cached_after_init):>12.4f} {np.min(nocache_times):>12.4f} {'-':>10}")
    print(f"{'Max step time (s)':<30} {np.max(cached_after_init):>12.4f} {np.max(nocache_times):>12.4f} {'-':>10}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Diagnose prior_flow cache behavior')
    parser.add_argument('--prior_args_path', type=str, 
                        default='num_tracers/prior_args_posterior.yaml',
                        help='Path to prior_args.yaml with prior_flow')
    parser.add_argument('--dataset', type=str, default='dr1',
                        help='Dataset to use (dr1 or dr2)')
    parser.add_argument('--design_args_path', type=str,
                        default=None,
                        help='Path to design_args.yaml (defaults to design_args_{dataset}.yaml)')
    parser.add_argument('--cosmo_model', type=str, default='base_omegak_w_wa',
                        help='Cosmology model')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of steps to simulate')
    parser.add_argument('--n_particles', type=int, default=30,
                        help='Number of particles per step')
    parser.add_argument('--n_designs', type=int, default=287,
                        help='Number of designs (set to 1 for faster test)')
    parser.add_argument('--cache_size', type=int, default=100000,
                        help='Cache size for cached sampling')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set design_args_path based on dataset if not provided
    if args.design_args_path is None:
        args.design_args_path = f'num_tracers/design_args_{args.dataset}.yaml'
    
    print("="*60)
    print("Prior Flow Cache Diagnostic")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Cosmo model: {args.cosmo_model}")
    print(f"N steps: {args.n_steps}")
    print(f"Sample shape: ({args.n_particles}, {args.n_designs})")
    print(f"Samples per step: {args.n_particles * args.n_designs:,}")
    print(f"Cache size: {args.cache_size:,}")
    print(f"Steps before shuffle: ~{args.cache_size // (args.n_particles * args.n_designs)}")
    print("="*60)
    
    # Load prior_args
    prior_args_full_path = os.path.join(parent_dir, args.prior_args_path)
    with open(prior_args_full_path, 'r') as f:
        prior_args = yaml.safe_load(f)
    
    # Load design_args
    design_args_full_path = os.path.join(parent_dir, args.design_args_path)
    with open(design_args_full_path, 'r') as f:
        design_args = yaml.safe_load(f)
    
    # Initialize experiment
    print("\nInitializing experiment...")
    experiment = init_experiment(
        cosmo_exp='num_tracers',
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model=args.cosmo_model,
        dataset=args.dataset,
        device=args.device,
        prior_flow_cache_size=args.cache_size,
        prior_flow_use_cache=True,
        prior_flow_cache_verbose=True,
    )
    
    sample_shape = (args.n_particles, args.n_designs)
    param_names = experiment.cosmo_params
    
    print(f"\nParameters: {param_names}")
    
    # Run cached sampling
    print(f"\n--- Cached Sampling (cache_size={args.cache_size:,}) ---")
    cached_samples, cached_times, cached_total = sample_over_steps(
        experiment, args.n_steps, sample_shape, use_cache=True
    )
    
    # Run non-cached sampling
    print(f"\n--- Non-Cached Sampling (fresh each step) ---")
    nocache_samples, nocache_times, nocache_total = sample_over_steps(
        experiment, args.n_steps, sample_shape, use_cache=False
    )
    
    # Print timing report
    print_timing_report(cached_times, nocache_times, cached_total, nocache_total, args.cache_size)
    
    # Generate plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    comparison_path = os.path.join(args.output_dir, 'prior_flow_cache_comparison.png')
    plot_comparison(cached_samples, nocache_samples, param_names, comparison_path)
    
    evolution_path = os.path.join(args.output_dir, 'prior_flow_cache_evolution.png')
    plot_sample_evolution(cached_samples, nocache_samples, param_names, evolution_path)
    
    print(f"\nDone! Check {args.output_dir} for plots.")


if __name__ == '__main__':
    main()
