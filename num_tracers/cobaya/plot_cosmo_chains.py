#!/usr/bin/env python3
"""
Script to plot cosmological chain samples using GetDist.
Works with both PolyChord and MCMC output directories.
"""

import numpy as np
import os
import glob
import sys

def detect_sampler_type(chain_dir):
    """
    Detect whether this is a PolyChord or MCMC output directory.
    """
    # Check for MCMC files
    if os.path.exists(os.path.join(chain_dir, 'chain.1.txt')):
        return 'mcmc'
    
    # Check for PolyChord files
    polychord_raw_dir = os.path.join(chain_dir, 'chain_polychord_raw')
    if os.path.exists(polychord_raw_dir):
        return 'polychord'
    
    # Check for any chain files
    chain_files = glob.glob(os.path.join(chain_dir, 'chain*.txt'))
    if chain_files:
        return 'mcmc'  # Assume MCMC if we find chain files
    
    return 'unknown'

def get_chain_file(chain_dir, sampler_type):
    """
    Get the appropriate chain file based on sampler type.
    """
    if sampler_type == 'mcmc':
        # Look for chain files (chain.1.txt, chain.2.txt, etc.)
        chain_files = glob.glob(os.path.join(chain_dir, 'chain.*.txt'))
        if chain_files:
            # Use the first chain file
            return chain_files[0]
        else:
            raise FileNotFoundError("No MCMC chain files found in {}".format(chain_dir))
    
    elif sampler_type == 'polychord':
        # Use the dead chain file
        dead_file = os.path.join(chain_dir, 'chain_polychord_raw', 'chain_dead.txt')
        if os.path.exists(dead_file):
            return dead_file
        else:
            raise FileNotFoundError("No PolyChord dead chain file found in {}".format(dead_file))
    
    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))

def parse_mcmc_header(chain_file):
    """
    Parse the header from MCMC chain file to get parameter names.
    """
    with open(chain_file, 'r') as f:
        header_line = f.readline().strip()
    
    # Skip the comment character and split
    if header_line.startswith('#'):
        header_line = header_line[1:].strip()
    
    # Split by whitespace and clean up
    all_columns = [name.strip() for name in header_line.split()]
    
    # Remove common non-parameter columns (but keep all actual parameters)
    non_param_cols = ['weight', 'minuslogpost', 'chi2', 'minuslogprior']
    param_names = [name for name in all_columns if not any(non in name.lower() for non in non_param_cols)]
    
    # Debug: print the mapping
    print("MCMC header analysis:")
    print("All columns:", all_columns)
    print("Parameter columns:", param_names)
    
    return param_names

def get_polychord_param_names():
    """
    Get parameter names for PolyChord chains (hardcoded based on analysis).
    """
    return [
        'loglike',  # Column 0
        'w',        # Column 1
        'wa',       # Column 2
        'omk',      # Column 3
        'hrdrag',   # Column 4
        'omm',      # Column 5
        'H0rdrag',  # Column 7 (skip column 6 which is rdrag_zero)
        'H0',       # Column 8
        'omegabh2', 'omegach2', 'tau', 'mnu', 'nnu', 
        'zrei', 'YHe', 'DHBBN', 'A', 'clamp', 'age', 'zdrag', 
        'Y_p', 'omegal', 'omegamh2'
    ]

def load_chain_data(chain_file, sampler_type):
    """
    Load chain data and return samples, parameter names, and all column names.
    """
    if sampler_type == 'mcmc':
        # MCMC chains have headers
        with open(chain_file, 'r') as f:
            header_line = f.readline().strip()
        
        # Skip the comment character and split
        if header_line.startswith('#'):
            header_line = header_line[1:].strip()
        
        # Get all column names
        all_columns = [name.strip() for name in header_line.split()]
        
        # Remove common non-parameter columns (but keep all actual parameters)
        non_param_cols = ['weight', 'minuslogpost', 'chi2', 'minuslogprior']
        param_names = [name for name in all_columns if not any(non in name.lower() for non in non_param_cols)]
        
        print("MCMC header analysis:")
        print("All columns:", all_columns)
        print("Parameter columns:", param_names)
        
        # Load data, skipping the header
        samples = np.loadtxt(chain_file)
        print("MCMC chain loaded with {} parameters: {}".format(len(param_names), param_names))
        
        return samples, param_names, all_columns
        
    elif sampler_type == 'polychord':
        # PolyChord chains don't have headers
        param_names = get_polychord_param_names()
        samples = np.loadtxt(chain_file)
        print("PolyChord chain loaded with {} columns".format(samples.shape[1]))
        return samples, param_names, param_names
        
    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))

def plot_with_getdist(chain_dir, model='base_omegak_w_wa', param_names=None, sampler_type=None):
    """
    Plot cosmological chain samples using GetDist.
    
    Parameters:
    -----------
    chain_dir : str
        Path to the chain directory
    param_names : list, optional
        List of parameter names to plot. If None, uses default parameters.
    sampler_type : str, optional
        Sampler type ('mcmc' or 'polychord'). If None, auto-detects.
    """
    try:
        from getdist import MCSamples, plots
    except ImportError:
        print("GetDist not available. Please install with: pip install getdist")
        return None
    
    # Detect sampler type if not provided
    if sampler_type is None:
        sampler_type = detect_sampler_type(chain_dir)
        print("Detected sampler type: {}".format(sampler_type))
    
    if sampler_type == 'unknown':
        print("Could not detect sampler type. Please specify manually.")
        return None
    
    # Get the chain file
    chain_file = get_chain_file(chain_dir, sampler_type)
    print("Using chain file: {}".format(chain_file))
    
    # Load the data
    samples, param_names_only, all_column_names = load_chain_data(chain_file, sampler_type)
    
    # Select parameters to plot
    if param_names is None:
        # Default parameters to plot
        if sampler_type == 'mcmc':
            # For MCMC, use the main cosmological parameters
            if model == 'base':
                preferred_params = ['omm', 'H0rdrag']
            elif model == 'base_omegak':
                preferred_params = ['omm', 'omk', 'H0rdrag']
            elif model == 'base_w':
                preferred_params = ['omm', 'w', 'H0rdrag']
            elif model == 'base_w_wa':
                preferred_params = ['omm', 'w', 'wa', 'H0rdrag']
            elif model == 'base_omegak_w_wa':
                preferred_params = ['omm', 'omk', 'w', 'wa', 'H0rdrag']
            param_names = [p for p in preferred_params if p in all_column_names]
            # If some are missing, add other available params
            if len(param_names) < len(preferred_params):
                remaining = [p for p in all_column_names if p not in param_names and not any(non in p.lower() for non in ['weight', 'minuslogpost', 'chi2', 'minuslogprior'])]
                param_names.extend(remaining[:5-len(param_names)])
        else:
            # For PolyChord, use the sampled parameters
            param_names = ['omm', 'omk', 'w', 'wa', 'hrdrag']
    
    print("Plotting parameters: {}".format(param_names))
    
    # Find parameter indices in the original column order
    param_indices = []
    for param in param_names:
        if param in all_column_names:
            param_indices.append(all_column_names.index(param))
            print("Parameter {} is at column {} (index {})".format(param, all_column_names.index(param), all_column_names.index(param)))
        else:
            print("Warning: Parameter {} not found in chain".format(param))
    
    if not param_indices:
        print("No valid parameters found!")
        return None
    
    # Extract the selected parameters
    param_samples = samples[:, param_indices]

    np.save(f'{model}.npy', param_samples)
    
    # Remove any samples with invalid values
    valid_mask = np.all(np.isfinite(param_samples), axis=1)
    param_samples = param_samples[valid_mask]
    
    print("Valid samples: {}".format(len(param_samples)))
    
    # Define LaTeX labels
    label_map = {
        'omm': '\Omega_m', 'omk': '\Omega_k', 'w': 'w_0', 'wa': 'w_a', 
        'H0rdrag': 'H_0 r_d', 'hrdrag': 'h r_d', 'H0': 'H_0',
        'rdrag': 'r_d', 'omegabh2': '\Omega_b h^2', 'omegach2': '\Omega_c h^2',
        'omegamh2': '\Omega_m h^2', 'omegal': '\Omega_\Lambda',
        'As': 'A_s', 'omch2': '\Omega_c h^2', 'omegam': '\Omega_m'
    }
    
    param_labels_latex = [label_map.get(param, param) for param in param_names]
    
    # Create MCSamples object
    mc_samples = MCSamples(
        samples=param_samples,
        names=param_names,
        labels=param_labels_latex
    )
    
    # Create triangle plot
    print("Creating GetDist triangle plot...")
    g = plots.getSubplotPlotter(width_inch=10)
    g.triangle_plot([mc_samples], filled=True, alpha=0.8)
    
    # Save the plot
    output_file = os.path.join(chain_dir, 'triangle_plot.png')
    g.export(output_file)
    print("GetDist plot saved to: {}".format(output_file))
    
    # Print statistics
    print("\n" + "="*60)
    print("GETDIST PARAMETER STATISTICS")
    print("="*60)
    print(mc_samples.getTable(limit=1).tableTex())
    
    return mc_samples

def main():
    """Main function."""
    # Default to PolyChord directory
    model = 'base_omegak'
    chain_dir = "/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/chains/mcmc/base_omegak/camb/run20251003_162414"
    #chain_dir = "/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/chains/bao/polychord/camb/run20251001_022505/base_omegak_w_wa/desi-bao-all"
    
    # Allow custom chain directory from command line
    if len(sys.argv) > 1:
        chain_dir = sys.argv[1]
    
    # Allow custom parameters from command line
    param_names = None
    if len(sys.argv) > 2:
        param_names = sys.argv[2].split(',')
        print("Using custom parameters: {}".format(param_names))
    
    try:
        mc_samples = plot_with_getdist(chain_dir, model, param_names)
        if mc_samples is not None:
            print("\nSuccessfully created GetDist plot: {}".format(os.path.join(chain_dir, 'triangle_plot.png')))
        
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
