import sys
import os
import io
import contextlib
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

home_dir = os.environ["HOME"]
if home_dir + "/bed/BED_cosmo" not in sys.path:
    sys.path.insert(0, home_dir + "/bed/BED_cosmo")
sys.path.insert(0, home_dir + "/bed/BED_cosmo/num_tracers")

import mlflow
from mlflow.tracking import MlflowClient
import getdist
import numpy as np
from getdist import plots
from util import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.colors
import warnings
from datetime import datetime
from pyro_oed_src import posterior_loss
import json
from IPython.display import display
import glob

home_dir = os.environ["HOME"]
sys.path.insert(0, home_dir + '/desi-y1-kp/')
from desi_y1_plotting import KP7StylePaper, utils

style = KP7StylePaper()

# Disable LaTeX rendering globally to avoid "latex could not be found" errors
# This overrides any LaTeX settings that might be set by the style object
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times', 'serif']

def plot_posterior(
    samples, 
    colors, 
    legend_labels=None, 
    show_scatter=False, 
    line_style="-", 
    alpha=1.0, 
    levels=[0.68, 0.95], 
    width_inch=7, 
    ranges=None,
    scatter_alpha=0.6,
    contour_alpha_factor=0.8,
    style=style
    ):
    """
    Plots posterior distributions using GetDist triangle plots.

    Args:
        samples (list): List of GetDist MCSamples objects.
        colors (list): List of colors for each sample.
        legend_labels (list, optional): List of legend labels for each sample.
        show_scatter (bool or list): If True, show scatter/histograms on the 1D/2D plots for all samples.
            If a list, specifies whether to show scatter for each sample individually.
        line_style (str or list): Line style for contours. Can be a single string or a list of strings corresponding to each sample.
        alpha (float or list): Alpha value for the contours. Can be a single float or a list of floats corresponding to each sample.
        levels (float or list, optional): Contour levels to use (e.g., 0.68 or [0.68, 0.95]).
            If a single float is provided, it is converted to a list.
            If None, the default GetDist settings are used.
        width_inch (float): Width of the plot in inches. Higher values increase resolution.
        ranges (dict, optional): Dictionary specifying fixed ranges for parameters. 
            Keys should be parameter names, values should be tuples of (min, max).
        scatter_alpha (float): Alpha value for scatter points. Default 0.6 for better distinguishability.
        contour_alpha_factor (float): Factor to adjust contour alpha for distinguishability. Default 0.8.
        style (object, optional): Style object (like KP7StylePaper) to apply to the plotter settings.
    Returns:
        g: GetDist plotter object with the generated triangle plot.
    """
    # Use get_single_plotter with proper ratio and scaling like in the notebook
    getdist_2D_ratio = 1 / 1.2
    g = plots.get_single_plotter(width_inch=width_inch, ratio=getdist_2D_ratio, scaling=True)
    
    # Apply style settings if provided (like KP7StylePaper)
    if style is not None:
        g.settings.__dict__.update(style.settings.__dict__)
    
    # Calculate dynamic font sizes for axis labels based on plot width and number of parameters
    # This ensures text stays readable as plot size changes
    if isinstance(samples, list) and len(samples) > 0:
        n_params = len(samples[0].paramNames.names)
        # Scale axis labels with plot width AND number of parameters
        # More parameters = larger triangle plot = need larger fonts
        # Use additive sqrt scaling with reduced coefficients for better balance
        axis_label_fontsize = max(9, min(22, width_inch * (0.3 + 0.5 * np.sqrt(n_params))))
        # Apply to GetDist settings
        g.settings.axes_fontsize = axis_label_fontsize
        g.settings.axes_labelsize = axis_label_fontsize
        g.settings.lab_fontsize = axis_label_fontsize

    if type(samples) != list:
        samples = [samples]
    if type(colors) != list:
        colors = [colors]
    if type(legend_labels) != list and legend_labels is not None:
        legend_labels = [legend_labels]

    colors = [convert_color(c) for c in colors]

    # Create adjusted colors for contours and scatter points
    def adjust_color_brightness(color, factor):
        """Adjust color brightness by a factor (0-1 makes darker, >1 makes lighter)"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        # Make darker by multiplying by factor
        adjusted_rgb = tuple(min(1.0, c * factor) for c in rgb)
        return mcolors.to_hex(adjusted_rgb)
    
    # Create slightly darker colors for contours and lighter for scatter
    contour_colors = [adjust_color_brightness(c, 0.8) for c in colors]  # Darker contours
    scatter_colors = [adjust_color_brightness(c, 1.2) for c in colors]  # Lighter scatter

    # Handle line_style - convert to list if it's a string
    if isinstance(line_style, str):
        line_style = [line_style] * len(samples)
    elif isinstance(line_style, list):
        # If it's already a list, ensure it has enough elements
        if len(line_style) < len(samples):
            # Extend the list by repeating the last element
            line_style = line_style + [line_style[-1]] * (len(samples) - len(line_style))
        elif len(line_style) > len(samples):
            # Truncate if too many elements
            line_style = line_style[:len(samples)]

    # Handle show_scatter - convert to list if it's a boolean
    if isinstance(show_scatter, bool):
        show_scatter = [show_scatter] * len(samples)
    elif isinstance(show_scatter, list):
        # If it's already a list, ensure it has enough elements
        if len(show_scatter) < len(samples):
            # Extend the list by repeating the last element
            show_scatter = show_scatter + [show_scatter[-1]] * (len(samples) - len(show_scatter))
        elif len(show_scatter) > len(samples):
            # Truncate if too many elements
            show_scatter = show_scatter[:len(samples)]

    # Set line styles in GetDist settings
    g.settings.line_styles = line_style
    # Apply contour alpha factor for better distinguishability
    # Note: We'll handle per-sample alphas after plotting
    if isinstance(alpha, (int, float)):
        adjusted_alpha = alpha * contour_alpha_factor
        g.settings.plot_args = {'alpha': adjusted_alpha}
    else:
        # Convert list to per-sample adjusted alphas
        adjusted_alpha = [a * contour_alpha_factor for a in alpha]
        # Don't set in plot_args yet - will handle after triangle_plot
    
    # Additional settings for higher resolution
    g.settings.solid_contour_palefactor = 0.6
    g.settings.alpha_filled_add = 0.85
    g.settings.alpha_factor_contour_lines = 1.0
    g.settings.linewidth_contour = 1.5
    g.settings.linewidth_meanlikes = 1.5

    # Prepare contour_args with custom levels if provided
    # For GetDist, we don't pass line styles in contour_args when using multiple styles

    # Set contour levels if provided
    if levels is not None:
        if isinstance(levels, float):
            levels = [levels]
        for sample in samples:
            sample.updateSettings({'contours': levels})

    # Create triangle plot
    g.triangle_plot(
        samples,
        colors=contour_colors,
        legend_labels=legend_labels,
        filled=False,
        normalized=True,
        diag1d_kwargs={
            'colors': contour_colors,
            'normalized': True
        },
        show=False
    )
    
    # If alpha is a list, manually set alpha for each sample's lines and contours
    if isinstance(alpha, list):
        # Iterate through all subplots and set alpha for lines and collections
        n_params = len(samples[0].paramNames.names)
        for i in range(n_params):
            for j in range(i + 1):
                ax = g.subplots[i, j]
                if ax is not None:
                    # Handle 1D plots (diagonal, i == j) - set alpha for lines
                    if i == j:
                        lines = ax.get_lines()
                        n_lines_per_sample = len(lines) // len(samples) if len(samples) > 0 else 0
                        for sample_idx in range(len(samples)):
                            start_idx = sample_idx * n_lines_per_sample
                            end_idx = (sample_idx + 1) * n_lines_per_sample
                            for line in lines[start_idx:end_idx]:
                                line.set_alpha(adjusted_alpha[sample_idx])
                    # Handle 2D plots (off-diagonal, i != j) - set alpha for contour collections
                    else:
                        collections = ax.collections
                        n_collections_per_sample = len(collections) // len(samples) if len(samples) > 0 else 0
                        for sample_idx in range(len(samples)):
                            start_idx = sample_idx * n_collections_per_sample
                            end_idx = (sample_idx + 1) * n_collections_per_sample
                            for collection in collections[start_idx:end_idx]:
                                collection.set_alpha(adjusted_alpha[sample_idx])

    param_names = g.param_names_for_root(samples[0])
    param_name_list = [p.name for p in param_names.names]
    
    # Manual axis limits if ranges is provided and didn't work
    if ranges is not None:
        # Set axis limits manually for each parameter
        for i, param in enumerate(param_name_list):
            if param in ranges:
                min_val, max_val = ranges[param]
                # Set limits for diagonal (1D) plots
                if hasattr(g, 'subplots') and g.subplots is not None:
                    g.subplots[i, i].set_xlim(min_val, max_val)
                    # Set limits for off-diagonal (2D) plots
                    for j in range(i):
                        if param_name_list[j] in ranges:
                            g.subplots[i, j].set_xlim(ranges[param_name_list[j]][0], ranges[param_name_list[j]][1])
                            g.subplots[i, j].set_ylim(min_val, max_val)

    if any(show_scatter):
        for i, param in enumerate(param_name_list):
            if i < len(g.subplots) and i < len(g.subplots[i]):
                ax = g.subplots[i][i]
                current_ylim = ax.get_ylim()
                for k, sample in enumerate(samples):
                    if show_scatter[k]:  # Only show scatter for this sample if enabled
                        param_index = sample.paramNames.list().index(param)
                        if param_index is not None:
                            values = sample.samples[:, param_index]
                            ax.hist(values, bins=30, alpha=scatter_alpha, color=scatter_colors[k],
                                    density=True, histtype='stepfilled', zorder=1)
                ax.set_ylim(current_ylim)

        param_combinations = [
            (param_name_list[i], param_name_list[j])
            for i in range(len(param_name_list))
            for j in range(i+1, len(param_name_list))
        ]

        for param_x, param_y in param_combinations:
            for i in range(len(g.subplots)):
                for j in range(i):
                    if param_name_list[i] == param_y and param_name_list[j] == param_x:
                        ax = g.subplots[i][j]
                        for k, sample in enumerate(samples):
                            if show_scatter[k]:  # Only show scatter for this sample if enabled
                                g.add_2d_scatter(
                                    sample,
                                    param_x,
                                    param_y,
                                    color=scatter_colors[k],
                                    ax=ax,
                                    scatter_size=4,
                                    alpha=scatter_alpha,
                                )

    return g

def plot_run(run_id, eval_args, show_scatter=False, cosmo_exp='num_tracers'):
    run_data_list, _, _ = get_runs_data(run_ids=run_id, parse_params=True, cosmo_exp=cosmo_exp)
    if not run_data_list:
        print(f"Run {run_id} not found.")
        return
    run_data = run_data_list[0]
    samples, _ = get_nominal_samples(run_data['run_obj'], run_data['params'], eval_args, cosmo_exp=cosmo_exp, global_rank=0)
    g = plot_posterior(samples, ["tab:blue"], show_scatter=show_scatter)
    plt.show()

def plot_training(
        run_id,
        var=None,
        cosmo_exp='num_tracers',
        log_scale=True,
        loss_step_freq=10,
        start_step=0,
        area_step_freq=100,
        lr_step_freq=1,
        show_area=True,
        area_limits=[0.5, 2.0],
        show_lr=True,
        dpi=300,
        step_range=None
        ):
    """
    Plots training loss, learning rate, and posterior contour area evolution
    for a single MLflow run using three vertically stacked subplots sharing the x-axis.
    Top plot: Loss.
    Middle plot: Learning Rate.
    Bottom plot: Posterior Contour Area (only if log_nominal_area is True in run args).

    Args:
        run_id (str): Specific MLflow run ID to plot.
        var (str or list): Parameter(s) from MLflow run params to include in the label.
        cosmo_exp (str): Experiment name or ID (if needed for path).
        log_scale (bool): If True, use log scale for the y-axes (Loss, LR, Area). Loss values <= 0 will be omitted in log scale.
        loss_step_freq (int): Sampling frequency for plotting loss points.
        start_step (int): Starting step offset for x-axis.
        area_step_freq (int): Sampling frequency for plotting nominal area points (must be multiple of 100).
        lr_step_freq (int): Sampling frequency for plotting learning rate points.
        show_area (bool): If True, show the area subplot. Default is True.
        show_lr (bool): If True, show the learning rate subplot. Default is True.
        step_range (tuple, optional): Tuple of (min_step, max_step) to limit the x-axis range. If None, plots all available steps.
    """
    # Set MLflow tracking URI before creating client
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()

    run_data_list, experiment_id_for_save_path, _ = get_runs_data(
        run_ids=[run_id],
        parse_params=False,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        return

    run_data_item = run_data_list[0]
    run_params = run_data_item['params']

    # Check if we should show area plot
    # Only show area if user requests it AND the run has area data
    if show_area:
        has_area_data = run_params.get('log_nominal_area', 'False').lower() == 'true'
        if not has_area_data:
            print(f"Warning: Run {run_id} does not have area data (log_nominal_area=False). Area plot will not be shown.")

    # --- Data Fetching (Metrics) ---
    try:
        # Fetch metrics
        loss_hist_raw = client.get_metric_history(run_id, 'loss')
        lr_hist_raw = client.get_metric_history(run_id, 'lr')
        
        # Process and filter NaNs/Infs
        loss = [(m.step + start_step, m.value) for m in loss_hist_raw if np.isfinite(m.value)]
        lr = [(m.step + start_step, m.value) for m in lr_hist_raw if np.isfinite(m.value)]

        if not loss:
            print(f"Warning: No valid loss points found for run {run_id}.")
            return

        # Apply step range filtering if specified
        if step_range is not None:
            min_step, max_step = step_range
            loss = [(step, value) for step, value in loss if min_step <= step <= max_step]
            lr = [(step, value) for step, value in lr if min_step <= step <= max_step]
            
            if not loss:
                print(f"Warning: No loss points found in step range {step_range} for run {run_id}.")
                return

        # Get area data if needed
        nom_area = {}
        if show_area:
            # Get all metrics for this run to find area pair metrics
            all_metrics = client.get_run(run_id).data.metrics
            area_metrics = {k: v for k, v in all_metrics.items() if k.startswith('nominal_area_avg')}
            if area_metrics:
                for metric_name in area_metrics.keys():
                    metric_hist = client.get_metric_history(run_id, metric_name)
                    area_data = [(m.step + start_step, m.value) for m in metric_hist if np.isfinite(m.value)]
                    
                    # Apply step range filtering if specified
                    if step_range is not None:
                        min_step, max_step = step_range
                        area_data = [(step, value) for step, value in area_data if min_step <= step <= max_step]
                    
                    nom_area[metric_name] = area_data

    except Exception as e:
        print(f"Error processing metrics for run {run_id}: {e}.")
        return

    # --- Plotting Setup ---
    # Calculate number of subplots based on what we want to show
    num_subplots = 1  # Always show loss
    if show_area and len(nom_area) > 0:
        num_subplots += 1
    if show_lr:
        num_subplots += 1
    
    fig_height = 4 * num_subplots  # Adjust height based on number of subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(14, fig_height), sharex=True)
    
    # If only one subplot, axes won't be an array
    if num_subplots == 1:
        axes = np.array([axes])
    
    # Assign axes based on what we're showing
    current_ax = 0
    ax1 = axes[current_ax]  # Loss is always first
    current_ax += 1
    
    ax_area = None
    if show_area and len(nom_area) > 0:
        ax_area = axes[current_ax]
        current_ax += 1
    
    ax_lr = None
    if show_lr:
        ax_lr = axes[current_ax]

    # Create label from parameters
    label_parts = []
    if var:
        vars_list = var if isinstance(var, list) else [var]
        for v_key in vars_list:
            if v_key in run_params:
                label_parts.append(f"{v_key}={run_params[v_key]}")
    base_label = ", ".join(label_parts) if label_parts else run_id[:8]

    # --- Plot Loss (ax1) ---
    if loss:
        loss_steps, loss_values = zip(*loss)
        sampled_indices = np.arange(0, len(loss_steps), loss_step_freq)
        plot_loss_steps = np.array(loss_steps)[sampled_indices]
        plot_loss_values = np.array(loss_values)[sampled_indices]

        if log_scale:
            min_loss = np.min(plot_loss_values)
            ax1.plot(plot_loss_steps, plot_loss_values - min_loss, color='tab:gray', label=base_label)
        else:
            ax1.plot(plot_loss_steps, plot_loss_values, color='tab:gray', label=base_label)

    # --- Plot Nominal Area (ax2) ---
    if show_area and len(nom_area) > 0:
        if area_step_freq % 100 != 0:
            print("Warning: area_step_freq should ideally be a multiple of 100 as nominal_area is logged every 100 steps.")
        sampling_rate = max(1, area_step_freq // 100)

        # Define line styles for different area pairs
        area_line_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for area_idx, (metric_name, area_data) in enumerate(nom_area.items()):
            if area_data:
                # Extract parameter pair name from metric name (e.g., 'avg_nominal_area_Om_hrdrag' -> 'Om, hrdrag')
                pair_name = metric_name.replace('nominal_area_avg_', '')
                param1, param2 = pair_name.split('_')[:2]

                area_steps, area_values = zip(*area_data)
                sampled_indices = np.arange(0, len(area_steps), sampling_rate)
                plot_area_steps = np.array(area_steps)[sampled_indices]
                plot_area_values = np.array(area_values)[sampled_indices]

                # Use different line style for each area pair
                line_color = area_line_colors[area_idx % len(area_line_colors)]

                try:
                    nominal_samples, target_labels, latex_labels = load_nominal_samples(run_params['cosmo_exp'], run_params['cosmo_model'], dataset=run_params['dataset'])
                    with contextlib.redirect_stdout(io.StringIO()):
                        nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
                    # Get all area pairs from DESI samples
                    nominal_area = get_contour_area([nominal_samples_gd], 0.68, param1, param2)[0]["nominal_area_"+pair_name]
                    ax_area.plot(plot_area_steps, plot_area_values/nominal_area, 
                                color=line_color, label=pair_name.replace('_', ', '))
                    ax_area.axhline(1, color='black', linestyle='--', lw=1.5)

                except NotImplementedError:
                    ax_area.plot(plot_area_steps, plot_area_values, 
                                color=line_color, label=pair_name.replace('_', ', '))

        # Configure ax2 (Contour Area)
        ax_area.set_ylabel("Nominal Design Area Ratio to DESI")
        ax_area.set_ylim(area_limits)
        ax_area.tick_params(axis='y')
        ax_area.legend(loc='best', title="Parameter Pair")
        ax_area.grid(True, axis='y', linestyle='--', alpha=0.6)

    # --- Plot Learning Rate (ax3) ---
    if show_lr and lr:
        lr_steps, lr_values = zip(*lr)
        sampled_indices = np.arange(0, len(lr_steps), lr_step_freq)
        plot_lr_steps = np.array(lr_steps)[sampled_indices]
        plot_lr_values = np.array(lr_values)[sampled_indices]

        ax_lr.plot(plot_lr_steps, plot_lr_values, color='tab:gray', label=base_label)

    # --- Final Plot Configuration ---
    # Configure ax1 (Loss)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Configure ax3 (Learning Rate)
    if show_lr:
        ax_lr.set_xlabel("Training Step")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.tick_params(axis='y')
        ax_lr.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Apply log scale if requested
    if log_scale:
        ax1.set_yscale('log')

    # Set x-axis limits if step_range is specified
    if step_range is not None:
        min_step, max_step = step_range
        ax1.set_xlim(min_step, max_step)

    # Adjust layout
    fig.set_constrained_layout(True)
    fig.suptitle(f"Training History - Run: {run_id[:8]}", fontsize=16)

    # Determine save path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = None
    save_path = None

    if experiment_id_for_save_path:
        save_dir = f"{storage_path}/mlruns/{experiment_id_for_save_path}/{run_id}/artifacts/plots"
    else:
        save_dir = f"{storage_path}/plots"
    
    save_path = f"{save_dir}/training_{timestamp}.png"

    os.makedirs(save_dir, exist_ok=True)
    save_figure(save_path, fig=fig, dpi=dpi)
    plt.close(fig)

def compare_posterior(
        mlflow_exp=None,
        run_ids=None,
        var=None,
        filter_string=None,
        guide_samples=10000,
        show_scatter=False,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        step='loss_best',
        seed=1,
        device="cuda:0",
        global_rank=0,
        dpi=300,
        levels=[0.68, 0.95],
        width_inch=10,
        colors=None
        ):
    """
    Compares posterior distributions across multiple runs in a triangle plot.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment.
        run_ids (list, optional): List of specific run IDs to compare.
        var (str or list, optional): Parameter(s) to group runs by. If None, each run is a separate group.
        filter_string (str, optional): MLflow filter string.
        guide_samples (int, optional): Number of samples to draw from the posterior (guide).
        seed (int, optional): Random seed.
        device (str, optional): Device to use.
        global_rank (int or list, optional): Global rank(s) to evaluate. If list, plots all ranks with same color per run.
        show_scatter (bool): Whether to show scatter points.
        excluded_runs (list): List of run IDs to exclude.
        levels (list): List of contour levels to plot (e.g., [0.68, 0.95]).
        cosmo_exp (str): Cosmology experiment folder name.
        step (str or int): Checkpoint to evaluate.
        width_inch (float): Width of the triangle plot in inches.
        colors (list, optional): List of colors to use for each group. If None, uses default matplotlib colors.
    """
    
    # Convert global_rank to list if it's a single value
    global_ranks = global_rank if isinstance(global_rank, list) else [global_rank]
    
    # Convert levels to list if it's a single value
    if not isinstance(levels, list):
        levels = [levels]

    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        filter_string=filter_string,
        parse_params=True,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        return
    
    if 'params' in run_data_list[0] and 'cosmo_model' in run_data_list[0]['params']:
        cosmo_model_for_desi = run_data_list[0]['params']['cosmo_model']
    else:
        raise ValueError("Could not determine cosmo_model from the first run for DESI plot.")

    # Create groups based on var parameter
    vars_list = var if isinstance(var, list) else [var] if var is not None else []
    
    # Group runs by variable values (or by run_id if no var specified)
    grouped_runs = {}
    for run_data_item in run_data_list:
        if var:
            # Group by variable values
            current_params = run_data_item['params']
            group_values = []
            is_valid_for_grouping = True
            for v_key in vars_list:
                if v_key in current_params:
                    group_values.append(current_params[v_key])
                else:
                    is_valid_for_grouping = False
                    break
            
            if is_valid_for_grouping:
                group_key = tuple(group_values)
            else:
                continue  # Skip runs that don't have all required parameters
        else:
            # Each run is its own group
            group_key = run_data_item['run_id']
        
        if group_key not in grouped_runs:
            grouped_runs[group_key] = []
        grouped_runs[group_key].append(run_data_item)
    
    if not grouped_runs:
        print("No valid groups found. Cannot plot.")
        return
    
    # Sort groups for consistent ordering - descending for numerical, alphabetical for text
    if var:
        # Use existing sort_key_for_group_tuple function but modify for descending numerical order
        def custom_sort_key(group_key):
            # First get the sort key using your existing function
            base_sort_key = sort_key_for_group_tuple(group_key)
            
            # Then modify numerical values to be negative for descending order
            if isinstance(base_sort_key, tuple):
                modified_values = []
                for val in base_sort_key:
                    if isinstance(val, (int, float)):
                        # Use negative value for descending order
                        modified_values.append(-val)
                    else:
                        # Keep string values as-is for alphabetical ordering
                        modified_values.append(val)
                return tuple(modified_values)
            else:
                # Single value case
                if isinstance(base_sort_key, (int, float)):
                    return -base_sort_key
                else:
                    return base_sort_key
        
        sorted_group_keys = sorted(grouped_runs.keys(), key=custom_sort_key)
    else:
        # For run_id grouping, use alphabetical order
        sorted_group_keys = sorted(grouped_runs.keys())
    
    # Collect samples and calculate areas for each group
    all_samples = []
    all_colors = []
    legend_handles = []
    
    # Use provided colors or default matplotlib colors
    if colors is not None:
        if len(colors) < len(sorted_group_keys):
            print(f"Warning: Only {len(colors)} colors provided for {len(sorted_group_keys)} groups. Repeating colors.")
        group_colors = {group_key: colors[i % len(colors)] for i, group_key in enumerate(sorted_group_keys)}
    else:
        prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        group_colors = {group_key: prop_cycle_colors[i % len(prop_cycle_colors)] for i, group_key in enumerate(sorted_group_keys)}
    
    for group_key in sorted_group_keys:
        group_runs = grouped_runs[group_key]
        group_color = group_colors[group_key]
        
        # Collect samples for all runs in this group across all ranks
        group_samples = []
        
        for run_data_item in group_runs:  
            # Get samples for all ranks
            for rank in global_ranks:
                samples_obj, selected_step = get_nominal_samples(
                    run_data_item['run_obj'], 
                    run_data_item['params'], 
                    guide_samples=guide_samples,
                    seed=seed,
                    device=device,
                    step=step, 
                    cosmo_exp=cosmo_exp, 
                    global_rank=rank
                    )
                if samples_obj is not None:
                    group_samples.append(samples_obj)
        
        if not group_samples:
            print(f"Warning: No valid samples for group {group_key}. Skipping.")
            continue
        
        # Add samples to plotting lists (all ranks get same color)
        all_samples.extend(group_samples)
        all_colors.extend([group_color] * len(group_samples))
        
        # Create legend entry
        if var:
            group_label = ', '.join([f'{vars_list[j]}={val}' for j, val in enumerate(group_key)])
        else:
            group_label = group_key[:8]  # First 8 chars of run_id
        
        # Create legend entry with just the group label (no area information)
        legend_handles.append(
            Line2D([0], [0], color=group_color, label=group_label)
        )
    
    if not all_samples:
        print("No samples generated for any group. Cannot plot.")
        return
    
    # Add DESI samples
    nominal_samples, target_labels, latex_labels = load_nominal_samples(cosmo_exp, cosmo_model_for_desi, dataset=run_data_list[0]['params']['dataset'])
    with contextlib.redirect_stdout(io.StringIO()):
        nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
    nominal_label = f'Nominal ({cosmo_model_for_desi})'
    
    all_samples.append(nominal_samples_gd)
    all_colors.append('black')
    
    legend_handles.append(
        Line2D([0], [0], color='black', label=nominal_label)
    )
    
    # Create the triangle plot with multiple contour levels
    g = plot_posterior(all_samples, all_colors, show_scatter=show_scatter, levels=levels, width_inch=width_inch)
    
    # Remove default legends and add custom one
    if g.fig.legends:
        for legend in g.fig.legends:
            legend.remove()
    
    # Set title
    title = f'Posterior Comparison, Step: {step}'
    if filter_string:
        title += f' (filter: {filter_string})'

    g.fig.set_constrained_layout(True)
    leg = g.fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.99, 0.96))
    leg.set_in_layout(False)
    g.fig.suptitle(title)

    storage_path_base = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    save_dir = f"{storage_path_base}/plots"
    filename_prefix = "posterior_comparison"

    if mlflow_exp and experiment_id_for_save_path:
        save_dir = f"{storage_path_base}/mlruns/{experiment_id_for_save_path}/plots"
    elif not mlflow_exp and run_ids and len(run_data_list) == 1 and experiment_id_for_save_path:
        single_run_id_for_path = run_data_list[0]['run_id']
        save_dir = f"{storage_path_base}/mlruns/{experiment_id_for_save_path}/{single_run_id_for_path}/artifacts/plots"
        filename_prefix = f"posterior_step_{step}"

    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    last_save_path = os.path.join(save_dir, save_filename)
    
    fig_to_close = None
    if hasattr(g, 'fig'):
        fig_to_close = g.fig
    elif plt.get_fignums():
        fig_to_close = plt.gcf()

    save_figure(last_save_path, fig=g.fig, dpi=dpi)

    if fig_to_close and fig_to_close in plt.get_fignums():
         plt.close(fig_to_close)

def load_eig_data_file(artifacts_dir, step_key=None):
    """
    Load the most recent completed eig_data JSON file from the artifacts directory.
    
    Args:
        artifacts_dir (str): Path to the artifacts directory containing eig_data files
        step_key (str or int, optional): If provided, only load files matching this step
    
    Returns:
        tuple: (json_path, data) where json_path is the path to the file and data is the loaded JSON.
               Returns (None, None) if no valid file is found.
    
    Raises:
        ValueError: If no completed eig_data files are found or if file cannot be loaded.
    """
    if not os.path.exists(artifacts_dir):
        raise ValueError(f"Artifacts directory not found: {artifacts_dir}")
    
    # Find all eig_data JSON files
    eig_files = glob.glob(f"{artifacts_dir}/eig_data_*.json")
    
    if len(eig_files) == 0:
        raise ValueError(f"No eig_data JSON files found in {artifacts_dir}")
    
    # Filter by step_key if provided
    if step_key is not None:
        step_str = str(step_key)
        eig_files = [f for f in eig_files if f"eig_data_{step_str}_" in os.path.basename(f)]
        if len(eig_files) == 0:
            raise ValueError(f"No eig_data files for step {step_key} found in {artifacts_dir}")
    
    # Sort by filename (most recent first)
    eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
    
    # Check each file for completion status (most recent first)
    for json_path in eig_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if evaluation is completed using status field
            status = data.get('status')
            if status == 'complete':
                return json_path, data
            # Skip incomplete files or files with missing/unexpected status
            continue
                
        except Exception as e:
            # Skip files that can't be loaded and continue to next
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    # No completed files found
    raise ValueError(f"No completed eig_data files found in {artifacts_dir}")

def compare_eigs(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=None,
        cosmo_exp='num_tracers',
        run_labels=None,
        var=None,
        step_key=None,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        colors=None,
        x_lim=None,
        y_lim=None,
        show_optimal=False,
        show_nominal=False,
        title=None,
        sort=True,
        sort_reference=None,
        normalize=False,
        show_errorbars=True,
        plot_input_design=False,
        design_labels=None
    ):
    """
    Compare EIG values across multiple runs using the same layout as `evaluate.py::eig_designs`.

    Args:
        mlflow_exp (str, optional): MLflow experiment name. Ignored when `run_ids` is provided.
        run_ids (list, optional): Explicit run IDs to load.
        excluded_runs (list, optional): Run IDs to skip.
        cosmo_exp (str): Cosmological experiment directory name.
        run_labels (list, optional): Custom legend labels per run.
        var (str or list, optional): Parameter(s) from MLflow run params to include in the label.
                                    If provided and run_labels is None, labels will be generated from these parameters.
        step_key (str or int, optional): Step identifier (if omitted the most recent step is used).
        save_path (str, optional): File path to persist the figure.
        figsize (tuple): Matplotlib figure size.
        dpi (int): Save resolution.
        colors (list, optional): Explicit colors for each run. Must be the same length as `run_ids`.
        x_lim (tuple, optional): X-axis limits.
        y_lim (tuple, optional): Y-axis limits.
        show_optimal (bool): If True, highlight each run's optimal EIG point when available.
        show_nominal (bool): If True, draw horizontal lines for nominal EIGs when available.
        title (str, optional): Custom figure title.
        sort (bool): Whether to reorder designs by descending EIG using the reference run.
        sort_reference (str, optional): Run ID or label that defines the sorting EIG. Required when `sort=True`.
        normalize (bool): If True, subtract each run's nominal EIG so curves are relative to nominal.
        show_errorbars (bool): If True, draw the filled std bands for each run.
        plot_input_design (bool): If True, plot scatter points for input_designs from MLflow params on the heatmap.
                                Only plots if they match the evaluation designs. Default False.
        design_labels (list, optional): Custom labels for each design dimension. Must be the same length as the number of design dimensions.
    """

    excluded_runs = excluded_runs or []

    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"

    # Parse params if we need them for var labels or plot_input_design
    need_params = plot_input_design or (var is not None and run_labels is None)
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=need_params,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")

    run_ids = [run_data['run_id'] for run_data in run_data_list]
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}

    all_data = []
    found_run_ids = []

    for run_id in run_ids:
        exp_id = run_id_to_exp_id.get(run_id)
        if exp_id is None:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue

        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        try:
            json_path, data = load_eig_data_file(artifacts_dir, step_key=step_key)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue

        all_data.append(data)
        found_run_ids.append(run_id)

    if not all_data:
        raise ValueError("No valid EIG data files found to compare")

    # Generate run_labels from var if provided and run_labels is None
    if run_labels is None:
        if var is not None:
            # Generate labels from var parameters
            vars_list = var if isinstance(var, list) else [var]
            run_labels = []
            for run_id in found_run_ids:
                # Find the corresponding run_data to get params
                run_data_item = next((r for r in run_data_list if r['run_id'] == run_id), None)
                if run_data_item and 'params' in run_data_item:
                    run_params = run_data_item['params']
                    label_parts = []
                    for v_key in vars_list:
                        if v_key in run_params:
                            label_parts.append(f"{v_key}={run_params[v_key]}")
                    if label_parts:
                        run_labels.append(", ".join(label_parts))
                    else:
                        run_labels.append(run_id[:8])
                else:
                    run_labels.append(run_id[:8])
        else:
            run_labels = [run_id[:8] for run_id in found_run_ids]
    if len(run_labels) != len(found_run_ids):
        raise ValueError("run_labels must match the number of discovered runs")

    step_numbers = []
    for data, run_label in zip(all_data, run_labels):
        if step_key is not None:
            selected_step = int(step_key) if isinstance(step_key, str) else step_key
        else:
            step_keys = [k for k in data.keys() if k.startswith('step_')]
            if not step_keys:
                raise ValueError(f"No step_* entries found in eig_data for {run_label}")
            step_numbers_in_file = []
            for key in step_keys:
                try:
                    step_num = int(key.split('_')[1])
                    step_numbers_in_file.append((step_num, key))
                except (ValueError, IndexError):
                    continue
            if step_numbers_in_file:
                selected_step = max(step_numbers_in_file, key=lambda x: x[0])[0]
            else:
                selected_step = int(step_keys[0].split('_')[1])
        step_numbers.append(selected_step)

    if colors is not None:
        if len(colors) != len(found_run_ids):
            raise ValueError("When provided, colors must match the number of run_ids.")
        color_list = [convert_color(c) for c in colors]
    else:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = [convert_color(c) for c in prop_cycle]

    design_shape = None
    run_records = []

    for data, run_id, run_label, color, step_num in zip(all_data, found_run_ids, run_labels, color_list, step_numbers):
        step_key_name = f"step_{int(step_num)}"
        if step_key_name not in data:
            print(f"Warning: Step {step_num} not found for run {run_id}, skipping...")
            continue
        step_payload = data[step_key_name]
        
        # Access nested structure: step_payload['variable'] and step_payload['nominal']
        variable_data = step_payload.get('variable', {})
        nominal_data = step_payload.get('nominal', {})

        # Get designs from top-level input_designs
        designs_root = data.get('input_designs')
        if designs_root is None:
            raise ValueError(f"`input_designs` missing in eig_data for run {run_id}")

        # input_designs is now always an array, not a dict with sorted/unsorted keys
        designs_arr = np.array(designs_root)

        if designs_arr.ndim == 1:
            designs_arr = designs_arr.reshape(-1, 1)
        if design_shape is None:
            design_shape = designs_arr.shape
        elif designs_arr.shape != design_shape:
            raise ValueError("Design arrays must have identical shapes across runs to enable comparison.")

        if design_labels is None:
            metadata = data.get('metadata', {})
            design_labels = metadata.get('design_labels', None)
            if design_labels is None or len(design_labels) != designs_arr.shape[1]:
                design_labels = [f'$d_{i}$' for i in range(designs_arr.shape[1])]
        else:
            if isinstance(design_labels, str):
                design_labels = [design_labels]

        eig_values_raw = variable_data.get('eigs_avg')
        if eig_values_raw is None:
            raise ValueError(f"`eigs_avg` missing in eig_data for run {run_id}")

        eig_values = np.array(eig_values_raw, dtype=float)
        eig_std_raw = variable_data.get('eigs_std')
        eig_std_values = np.array(eig_std_raw, dtype=float) if eig_std_raw is not None else np.zeros_like(eig_values)

        if eig_values.size == 0:
            print(f"Warning: No EIG data found for {run_id}, skipping...")
            continue

        # Get nominal_eig from nominal subdict (scalar value from eigs_avg)
        nominal_eig = nominal_data.get('eigs_avg')
        if nominal_eig is not None and isinstance(nominal_eig, list):
            nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
        optimal_eig = variable_data.get('optimal_eig')
        optimal_design = variable_data.get('optimal_design')

        # Get input_designs from MLflow params if requested
        input_designs_from_params = None
        if plot_input_design:
            # Find the corresponding run_data to get run_obj
            run_data_item = next((r for r in run_data_list if r['run_id'] == run_id), None)
            if run_data_item and run_data_item.get('run_obj') is not None:
                try:
                    input_designs_param = run_data_item['run_obj'].data.params.get('input_designs')
                    if input_designs_param is not None:
                        input_designs_list = json.loads(input_designs_param)
                        input_designs_from_params = np.array(input_designs_list)
                        if input_designs_from_params.ndim == 1:
                            input_designs_from_params = input_designs_from_params.reshape(-1, 1)
                        # Check if design dimensionality matches (allow different number of designs)
                        if input_designs_from_params.shape[1] != designs_arr.shape[1]:
                            print(f"Warning: input_designs from params for run {run_id} has {input_designs_from_params.shape[1]} dimensions, but evaluation designs have {designs_arr.shape[1]} dimensions. Skipping.")
                            input_designs_from_params = None
                except Exception as e:
                    print(f"Warning: Error loading input_designs from params for run {run_id}: {e}")

        run_records.append({
            'run_id': run_id,
            'run_label': run_label,
            'color': color,
            'designs': designs_arr,
            'eigs_avg': eig_values,
            'eigs_std': eig_std_values if eig_std_values.shape == eig_values.shape else np.zeros_like(eig_values),
            'nominal_eig': nominal_eig,
            'optimal_eig': optimal_eig,
            'optimal_design': np.array(optimal_design) if optimal_design is not None else None,
            'input_designs_from_params': input_designs_from_params
        })

    if not run_records:
        raise ValueError("No runs with valid design/eig data to plot.")

    num_designs, num_dims = design_shape

    # Check if designs are 1D or multi-dimensional (matching eig_designs logic)
    is_1d_design = (num_dims == 1)

    reference_record = None
    global_sort_idx = None

    if sort:
        if sort_reference is not None:
            reference_record = next(
                (rec for rec in run_records if rec['run_id'] == sort_reference or rec['run_label'] == sort_reference),
                None
            )
            if reference_record is None:
                raise ValueError(f"sort_reference '{sort_reference}' not found among provided run_ids or labels.")
            global_sort_idx = np.argsort(reference_record['eigs_avg'])[::-1]
        else:
            for rec in run_records:
                rec['sort_idx'] = np.argsort(rec['eigs_avg'])[::-1]
    else:
        global_sort_idx = np.arange(num_designs)

    # Get sorted designs for heatmap (use reference run or first run)
    if global_sort_idx is not None:
        idx_for_heatmap = global_sort_idx
    else:
        idx_for_heatmap = np.arange(num_designs)
    sorted_designs = run_records[0]['designs'][idx_for_heatmap]

    if is_1d_design and not sort:
        # For 1D designs without sorting, plot EIG directly against design variable (no second subplot needed)
        fig, ax_line = plt.subplots(figsize=figsize)
        ax_heat = None
        cbar_ax = None
        x_vals = sorted_designs[:, 0]
        x_label = design_labels[0]
    else:
        # For sorted 1D or multi-dimensional designs, align heatmap beneath line plot and add a vertical colorbar
        if is_1d_design and sort:
            height_ratios = [0.65, 0.15]
        else:
            height_ratios = [0.6, 0.2]
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=height_ratios,
            hspace=0.0
        )
        ax_line = fig.add_subplot(gs[0, 0])  # Top plot for EIG
        ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_line)  # Bottom plot for heatmap shares x-axis
        # Reserve a dedicated colorbar axis on the right that spans both subplots
        fig.subplots_adjust(left=0.06, right=0.92)
        bbox = ax_heat.get_position()
        cbar_width = 0.015
        cbar_gap = 0.02  # maintain a small gap between plots and colorbar
        cbar_left = min(0.985 - cbar_width, bbox.x1 + cbar_gap)
        cbar_height = 0.6
        cbar_bottom = 0.5 - cbar_height / 2  # vertically center the colorbar
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        plt.setp(ax_line.get_xticklabels(), visible=False)
        ax_line.tick_params(axis='x', which='both', length=0)
        
        # Set x_vals and x_label for sorted/multi-dimensional case
        x_vals = np.arange(num_designs)
        if sort_reference is not None:
            x_label_suffix = f" (sorted by {reference_record['run_label']})"
        elif sort:
            x_label_suffix = " (sorted per run)"
        else:
            x_label_suffix = ""
        x_label = f"Design Index{x_label_suffix}"

    nominal_line_drawn = False

    ax_line.margins(x=0)

    handles_for_legend = []
    nominal_handle = None

    for record in run_records:
        if global_sort_idx is not None:
            sort_idx = global_sort_idx
        else:
            sort_idx = record.get('sort_idx', np.arange(num_designs))

        eig_vals = record['eigs_avg'][sort_idx]
        eig_std_vals = record['eigs_std'][sort_idx]
        color = record['color']

        baseline = 0.0
        if normalize:
            if record['nominal_eig'] is None:
                raise ValueError(f"normalize=True requires nominal_eig for run {record['run_id']}")
            baseline = record['nominal_eig']

        eig_vals_plot = eig_vals - baseline

        if sorted_designs.shape[0] == 1:
            line = ax_line.scatter(x_vals, eig_vals_plot, color=color, zorder=5, label=record['run_label'])
        else:
            line = ax_line.plot(x_vals, eig_vals_plot, color=color, linewidth=2, label=record['run_label'])[0]
            
        handles_for_legend.append(line)
        if show_errorbars and np.any(eig_std_vals > 0):
            if sorted_designs.shape[0] == 1:
                ax_line.errorbar(x_vals, eig_vals_plot, yerr=eig_std_vals, color=color, zorder=5, fmt='o', capsize=5, label=record['run_label'])
            else:
                ax_line.fill_between(
                    x_vals,
                    eig_vals_plot - eig_std_vals,
                    eig_vals_plot + eig_std_vals,
                    color=color,
                    alpha=0.2
                )

        if show_nominal:
            if normalize:
                if not nominal_line_drawn:
                    nominal_handle = ax_line.axhline(0, linestyle='--', color='gray', alpha=0.6, linewidth=1.2, label='Nominal (all runs)')
                    nominal_line_drawn = True
            elif record['nominal_eig'] is not None:
                ax_line.axhline(record['nominal_eig'], linestyle='--', color=color, alpha=0.6, linewidth=1.2)

        if ax_heat is not None and show_optimal:
            point_y = None
            point_x = None
            is_reference_run = bool(sort_reference) and reference_record and record['run_id'] == reference_record['run_id']

            if record['optimal_design'] is not None and record['optimal_eig'] is not None and not is_reference_run:
                optimal_idx = None
                for idx, design_row in enumerate(record['designs']):
                    if np.allclose(design_row, record['optimal_design'], rtol=1e-5, atol=1e-8):
                        optimal_idx = idx
                        break
                if optimal_idx is not None:
                    sorted_pos = np.where(sort_idx == optimal_idx)[0][0]
                    point_x = x_vals[sorted_pos]
                    point_y = eig_vals_plot[sorted_pos]

            if (point_x is None or point_y is None) and is_reference_run:
                extremum_idx = np.argmax(eig_vals_plot)
                point_x = x_vals[extremum_idx]
                point_y = eig_vals_plot[extremum_idx]

            if point_x is not None and point_y is not None:
                ax_line.scatter(
                    point_x,
                    point_y,
                    color=color,
                    edgecolor='black',
                    zorder=5,
                    s=60
                )
        
        # Plot input_designs from MLflow params if requested
        if plot_input_design and record.get('input_designs_from_params') is not None:
            input_designs = record['input_designs_from_params']
            # Match each input_design to the corresponding evaluation design
            scatter_x = []
            scatter_y = []
            
            for input_design in input_designs:
                # Find the index in the original (unsorted) designs array
                match_idx = None
                for idx, eval_design in enumerate(record['designs']):
                    if np.allclose(eval_design, input_design, rtol=1e-5, atol=1e-8):
                        match_idx = idx
                        break
                
                if match_idx is not None:
                    # Find the position in the sorted array
                    sorted_pos = np.where(sort_idx == match_idx)[0][0]
                    scatter_x.append(x_vals[sorted_pos])
                    scatter_y.append(eig_vals_plot[sorted_pos])
            
            if scatter_x:
                # Plot scatter points for this run with its color (no label to exclude from legend)
                ax_line.scatter(
                    scatter_x,
                    scatter_y,
                    marker='x',
                    s=100,
                    color=color,
                    linewidths=2.5,
                    zorder=6,
                    alpha=0.9
                )

    y_label = 'Expected Information Gain [bits]'
    if normalize:
        y_label = 'EIG Relative to Nominal Design [bits]'
    ax_line.set_ylabel(y_label, fontsize=12, weight='bold')
    ax_line.grid(True, alpha=0.3)
    legend_handles = []
    legend_labels = []
    if nominal_handle is not None:
        legend_handles.append(nominal_handle)
        legend_labels.append(nominal_handle.get_label())
    legend_handles.extend(handles_for_legend)
    legend_labels.extend([h.get_label() for h in handles_for_legend])
    ax_line.legend(legend_handles, legend_labels, loc='best', framealpha=0.9)
    
    # Plot sorted designs (heatmap) if ax_heat exists
    if ax_heat is not None:
        if is_1d_design and not sort:
            extent = [x_vals.min(), x_vals.max(), -0.5, 0.5]
            im = ax_heat.imshow(
                sorted_designs.T,
                aspect='auto',
                cmap='viridis',
                origin='lower',
                extent=extent
            )
            ax_heat.set_xlim(x_vals.min(), x_vals.max())
            ax_line.set_xlim(x_vals.min(), x_vals.max())
            ax_heat.set_yticks([0])
            ax_heat.set_yticklabels([design_labels[0]])
        else:
            im = ax_heat.imshow(
                sorted_designs.T,
                aspect='auto',
                cmap='viridis',
                origin='upper'
            )
            ax_heat.set_xlim(-0.5, num_designs - 0.5)
            ax_line.set_xlim(-0.5, num_designs - 0.5)
            ax_heat.set_yticks(np.arange(num_dims))
            ax_heat.set_yticklabels(design_labels)

        ax_heat.set_xlabel(x_label, fontsize=12, weight='bold')
        ax_heat.set_ylabel('')
        ax_heat.tick_params(axis='x', which='both', labelsize=10)
        ax_heat.margins(x=0)

        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Design Value', fontsize=11, weight='bold')
        ax_line.spines['bottom'].set_visible(False)
        ax_heat.spines['bottom'].set_visible(True)
    else:
        # Single plot case: show x-label on ax_line
        ax_line.set_xlabel(x_label, fontsize=12, weight='bold')
        if x_lim is not None:
            ax_line.set_xlim(x_lim)
        elif is_1d_design and not sort:
            ax_line.set_xlim(x_vals.min(), x_vals.max())
        else:
            ax_line.set_xlim(-0.5, num_designs - 0.5)

    if y_lim is not None:
        ax_line.set_ylim(y_lim)

    if title is None:
        if mlflow_exp and actual_mlflow_exp_for_title:
            title = f'EIG per Design - {actual_mlflow_exp_for_title}'
        elif mlflow_exp:
            title = f'EIG per Design - {mlflow_exp}'
        else:
            title = 'EIG per Design Comparison'
    fig.suptitle(title, fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved comparison plot to {save_path}")

    return fig, (ax_line, ax_heat)

def compare_increasing_design(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        labels=None,
        step_key=None,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        colors=None,
        title=None,
        log_scale=True,
        use_fractional=False,
        include_nominal=False,
        ref_id=None
    ):
    """
    Compare increasing quantities for fixed design.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): List of MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Cosmological experiment name (default: 'num_tracers')
        labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
        step_key (str or int, optional): Which step to use. If None, finds most recent eig_data file
        save_path (str, optional): Path to save the comparison plot. If None, doesn't save
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
        colors (list, optional): Colors for each run. If None, uses a continuous colormap for increasing designs
        title (str, optional): Custom title for the plot. If None, generates default title
        log_scale (bool): Whether to use log scale for y-axis (default: True)
        use_fractional (bool): Whether to plot fractional values or absolute quantities (default: False)
        include_nominal (bool): Whether to include nominal design bars on the left (default: False)
        ref_id (str, optional): Reference run ID to use for nominal design, nominal EIG, and optimal EIG
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Set MLflow tracking URI
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Load reference run data if ref_id is provided
    ref_nominal_design = None
    ref_nominal_eig = None
    ref_nominal_eig_std = None
    ref_optimal_eig = None
    ref_optimal_eig_std = None
    ref_optimal_design = None
    ref_run_data_list = []
    if ref_id is not None:
        try:
            # Get reference run's experiment ID
            ref_run_data_list, _, _ = get_runs_data(
                run_ids=[ref_id],
                parse_params=True,
                cosmo_exp=cosmo_exp
            )
            if not ref_run_data_list:
                print(f"Warning: Reference run {ref_id} not found, skipping reference data...")
            else:
                ref_run_data = ref_run_data_list[0]
                ref_run_id = ref_run_data['run_id']
                ref_exp_id = ref_run_data['exp_id']
                ref_artifacts_dir = f"{storage_path}/mlruns/{ref_exp_id}/{ref_run_id}/artifacts"
                
                if os.path.exists(ref_artifacts_dir):
                    # Load completed reference eig_data file
                    try:
                        ref_json_path, ref_data = load_eig_data_file(ref_artifacts_dir, step_key=None)
                        if ref_data is not None:
                            print(f"Loaded reference EIG data from {ref_json_path}")
                            
                            # Determine which step to use
                            if step_key is not None:
                                step_str = f"step_{step_key}"
                            else:
                                # Find the highest available step key
                                step_keys = [k for k in ref_data.keys() if k.startswith('step_')]
                                if step_keys:
                                    # Extract step numbers and find the highest
                                    step_numbers = []
                                    for k in step_keys:
                                        try:
                                            step_num = int(k.split('_')[1])
                                            step_numbers.append((step_num, k))
                                        except (ValueError, IndexError):
                                            continue
                                    if step_numbers:
                                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                                    else:
                                        step_str = step_keys[0]  # Fallback to first if parsing fails
                                else:
                                    print(f"Warning: No step data found in reference file {ref_json_path}")
                                    step_str = None
                            
                            if step_str and step_str in ref_data:
                                step_data = ref_data[step_str]
                                
                                # Access nested structure
                                variable_data = step_data.get('variable', {})
                                nominal_data = step_data.get('nominal', {})
                                
                                # Extract nominal_eig from nominal subdict (scalar value from eigs_avg)
                                nominal_eig_val = nominal_data.get('eigs_avg')
                                if nominal_eig_val is not None:
                                    if isinstance(nominal_eig_val, list) and len(nominal_eig_val) > 0:
                                        nominal_eig_val = nominal_eig_val[0]
                                    if isinstance(nominal_eig_val, (int, float)):
                                        ref_nominal_eig = nominal_eig_val
                                
                                nominal_eig_std_val = nominal_data.get('eigs_std')
                                if nominal_eig_std_val is not None:
                                    if isinstance(nominal_eig_std_val, list) and len(nominal_eig_std_val) > 0:
                                        nominal_eig_std_val = nominal_eig_std_val[0]
                                    if isinstance(nominal_eig_std_val, (int, float)):
                                        ref_nominal_eig_std = nominal_eig_std_val
                                
                                if 'optimal_eig' in variable_data:
                                    ref_optimal_eig = variable_data['optimal_eig']
                                
                                if 'optimal_eig_std' in variable_data and isinstance(variable_data['optimal_eig_std'], (int, float)):
                                    ref_optimal_eig_std = variable_data['optimal_eig_std']
                                
                                if 'optimal_design' in variable_data:
                                    ref_optimal_design = np.array(variable_data['optimal_design'])
                                
                                # Use nominal_design stored in the reference eig_data without initializing an experiment
                                if include_nominal:
                                    if 'nominal_design' in ref_data:
                                        ref_nominal_design = np.array(ref_data['nominal_design'])
                        else:
                            print(f"Warning: No completed reference eig_data file found, skipping reference data...")
                            ref_data = None
                    except ValueError as e:
                        print(f"Warning: {e}, skipping reference data...")
                        ref_data = None
                else:
                    print(f"Warning: Artifacts directory not found for reference run {ref_id}")
        except Exception as e:
            print(f"Warning: Error loading reference run {ref_id}: {e}")
    
    # Get run data - can use either mlflow_exp or run_ids
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=True,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")
    
    # Extract run_ids from run_data_list
    run_ids = [run_data['run_id'] for run_data in run_data_list]
    
    # Build a map from run_id to exp_id
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    # Find and load JSON files for each run
    optimal_designs = []
    optimal_eigs = []
    optimal_eig_stds = []
    found_run_ids = []
    design_labels = None
    nominal_eig = None  # Store nominal_eig from first run's data (for scatter plot)
    nominal_eig_std = None
    
    for run_id in run_ids:
        if run_id not in run_id_to_exp_id:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue
        
        exp_id = run_id_to_exp_id[run_id]
        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        # Load completed eig_data file
        try:
            json_path, data = load_eig_data_file(artifacts_dir, step_key=None)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
            print(f"Loaded EIG data from {json_path}")
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue
        
        try:
            
            # Determine which step to use
            if step_key is not None:
                step_str = f"step_{step_key}"
            else:
                # Find the highest available step key
                step_keys = [k for k in data.keys() if k.startswith('step_')]
                if step_keys:
                    # Extract step numbers and find the highest
                    step_numbers = []
                    for k in step_keys:
                        try:
                            step_num = int(k.split('_')[1])
                            step_numbers.append((step_num, k))
                        except (ValueError, IndexError):
                            continue
                    if step_numbers:
                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                    else:
                        step_str = step_keys[0]  # Fallback to first if parsing fails
                else:
                    print(f"Warning: No step data found in {json_path}, skipping...")
                    continue
            
            if step_str not in data:
                print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                continue
            
            step_data = data[step_str]
            
            # Access nested structure
            variable_data = step_data.get('variable', {})
            nominal_data = step_data.get('nominal', {})
                
            # Extract optimal design and EIG from variable subdict
            if 'optimal_design' in variable_data and 'optimal_eig' in variable_data:
                optimal_design = np.array(variable_data['optimal_design'])
                optimal_eig_value = variable_data.get('optimal_eig', None)
                optimal_eig_std_value = variable_data.get('optimal_eig_std', None)
                if not isinstance(optimal_eig_std_value, (int, float)):
                    optimal_eig_std_value = None
                optimal_designs.append(optimal_design)
                optimal_eigs.append(optimal_eig_value)
                optimal_eig_stds.append(optimal_eig_std_value)
                found_run_ids.append(run_id)
                
                # Get design labels from experiment (will be initialized later if needed)
                # For now, we'll get them from the experiment initialization below
                
                # Extract nominal_eig from first run's data (for scatter plot in compare_increasing_design)
                # Only do this in compare_increasing_design - check if include_nominal is in scope
                if 'include_nominal' in locals() and include_nominal:
                    nominal_eig_val = nominal_data.get('eigs_avg')
                    if nominal_eig_val is not None:
                        if isinstance(nominal_eig_val, list) and len(nominal_eig_val) > 0:
                            nominal_eig_val = nominal_eig_val[0]
                        if isinstance(nominal_eig_val, (int, float)):
                            if nominal_eig is None:
                                nominal_eig = nominal_eig_val
                    nominal_eig_std_val = nominal_data.get('eigs_std')
                    if nominal_eig_std_val is not None:
                        if isinstance(nominal_eig_std_val, list) and len(nominal_eig_std_val) > 0:
                            nominal_eig_std_val = nominal_eig_std_val[0]
                        if isinstance(nominal_eig_std_val, (int, float)):
                            if nominal_eig_std is None:
                                nominal_eig_std = nominal_eig_std_val
            else:
                print(f"Warning: No optimal_design data found in {json_path} for {step_str}, skipping...")
                continue
                
        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    if len(optimal_designs) == 0:
        raise ValueError("No valid optimal design data found to compare")
    
    # Generate labels if not provided
    if labels is None:
        labels = [run_id[:8] for run_id in found_run_ids]
    
    if len(labels) != len(optimal_designs):
        labels = labels[:len(optimal_designs)]
    
    # Generate colors if not provided
    # Use a continuous colormap for increasing designs (excluding nominal which will be tab:blue)
    if colors is None:
        n_designs = len(optimal_designs)
        if n_designs > 0:
            cmap = plt.get_cmap('viridis')
            # Normalize indices to [0.2, 0.9] range to avoid very dark/light ends
            color_indices = np.linspace(0.2, 0.9, n_designs)
            colors = [cmap(idx) for idx in color_indices]
        else:
            colors = []
    else:
        if len(colors) < len(optimal_designs):
            colors = [colors[i % len(colors)] for i in range(len(optimal_designs))]
        else:
            colors = colors[:len(optimal_designs)]
    
    # Initialize experiment from first run to get nominal_total_obs, nominal_design, and design labels if needed
    nominal_total_obs = None
    nominal_design = None
    experiment = None
    
    # Use reference nominal design if ref_id is provided
    if ref_id is not None and ref_nominal_design is not None:
        nominal_design = ref_nominal_design
    
    # Always initialize if use_fractional=False (need nominal_total_obs for conversion), design_labels missing, or include_nominal=True (and we don't have ref data)
    if not use_fractional or design_labels is None or (include_nominal and nominal_design is None):
        try:
            # Use ref run data if available, otherwise use first run
            if ref_id is not None and len(ref_run_data_list) > 0:
                run_data_for_init = ref_run_data_list[0]
            else:
                run_data_for_init = run_data_list[0]
            
            device = "cuda:0"
            # We don't need to initialize designs, just need the experiment object
            experiment = init_experiment(
                run_data_for_init['run_obj'], 
                run_data_for_init['params'], 
                device, 
                design_args={}, 
                global_rank=0
            )
            # Always get nominal_total_obs when use_fractional=False (needed for conversion)
            # Also get it if design_labels is missing or include_nominal=True (needed for nominal design conversion)
            if not use_fractional or design_labels is None or include_nominal:
                nominal_total_obs = experiment.nominal_total_obs
            # Get nominal_design if include_nominal is True and we don't have ref data
            if include_nominal and nominal_design is None and hasattr(experiment, 'nominal_design'):
                nominal_design = experiment.nominal_design.cpu().numpy()
            if design_labels is None and hasattr(experiment, 'design_labels'):
                design_labels = experiment.design_labels
        except Exception as e:
            print(f"Warning: Could not initialize experiment to get nominal_total_obs: {e}")
            if not use_fractional:
                print(f"Warning: Cannot convert to absolute quantities without nominal_total_obs. Designs will remain fractional.")
            if include_nominal:
                print(f"Warning: Cannot get nominal design. Skipping nominal design bars.")
    
    # Use generic labels if design_labels not found
    if design_labels is None:
        # Infer from first design's dimensionality
        if len(optimal_designs) > 0:
            n_dims = len(optimal_designs[0])
            design_labels = [f'$f_{i}$' for i in range(n_dims)]
        else:
            design_labels = []
    
    # Check if designs are 1D or multi-dimensional
    if len(optimal_designs) > 0:
        n_dims = len(optimal_designs[0])
        is_1d = (n_dims == 1)
    else:
        is_1d = False
    
    # Store fractional designs before conversion (needed for scatter plot % increase calculation)
    fractional_optimal_designs = [np.array(design).copy() for design in optimal_designs]
    fractional_nominal_design = np.array(nominal_design).copy() if nominal_design is not None else None
    
    # Convert fractional values to absolute quantities if needed
    # Note: nominal_design is stored as fractional (like optimal_designs), so convert it too
    if not use_fractional and nominal_total_obs is not None:
        optimal_designs = np.array(optimal_designs) * nominal_total_obs
        if nominal_design is not None:
            nominal_design = nominal_design * nominal_total_obs
    
    # Determine y-axis label
    if is_1d:
        ylabel = design_labels[0] if design_labels else 'Design Value'
    else:
        if use_fractional:
            ylabel = 'Fraction of Total Tracers'
        else:
            ylabel = 'Number of Tracers'
    
    # Create figure
    if is_1d:
        # For 1D designs, use a simple bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        design_values = [design[0] for design in optimal_designs]
        plot_labels = list(labels) if labels else [run_id[:8] for run_id in found_run_ids]
        plot_colors = list(colors) if colors else [plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])] for i in range(len(optimal_designs))]
        
        # Add nominal design if requested - prepend to lists
        if include_nominal and nominal_design is not None:
            design_values = [nominal_design[0]] + design_values
            plot_labels = ['Nominal'] + plot_labels
            plot_colors = ['tab:blue'] + plot_colors
        
        x_pos = np.arange(len(design_values))
        # Plot bars individually to conditionally apply edgecolor
        # Nominal (first bar if include_nominal) has no edge, increasing designs have black edge
        bars = []
        for i, (x, val, color) in enumerate(zip(x_pos, design_values, plot_colors)):
            is_nominal = (include_nominal and nominal_design is not None and i == 0)
            edgecolor = 'none' if is_nominal else 'black'
            bar = ax.bar(x, val, color=color, alpha=0.7, edgecolor=edgecolor, linewidth=1.5)
            bars.append(bar)
        
        ax.set_xlabel('Run', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        if log_scale:
            ax.set_yscale('log')
        
    else:
        # For multi-dimensional designs, use grouped bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(design_labels))  # Design dimension positions
        
        # Prepare data for plotting - prepend nominal design if requested
        plot_designs = list(optimal_designs)
        plot_labels = list(labels) if labels else [run_id[:8] for run_id in found_run_ids]
        plot_colors = list(colors) if colors else [plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])] for i in range(len(optimal_designs))]
        
        if include_nominal and nominal_design is not None:
            plot_designs = [nominal_design] + plot_designs
            plot_labels = ['Nominal'] + plot_labels
            plot_colors = ['tab:blue'] + plot_colors
        
        n_designs_to_plot = len(plot_designs)
        width = 0.8 / n_designs_to_plot  # Width of bars
        
        # Plot bars for each design
        # Nominal (first design if include_nominal) has no edge, increasing designs have black edge
        for i, (design, label, color) in enumerate(zip(plot_designs, plot_labels, plot_colors)):
            offset = (i - n_designs_to_plot/2 + 0.5) * width
            is_nominal = (include_nominal and nominal_design is not None and i == 0)
            edgecolor = 'none' if is_nominal else 'black'
            bars = ax.bar(x + offset, design, width, label=label, color=color, alpha=0.7, edgecolor=edgecolor, linewidth=1.5)
            
        
        ax.set_xlabel('Tracer Class', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(design_labels, fontsize=11)
        ax.legend(loc='best', fontsize=16, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        if log_scale:
            ax.set_yscale('log')
    
    # Set title
    if title is None:
        title = f'Uniformly Increasing Tracer Observations'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved comparison plot to {save_path}")
    
    # Create second scatter plot: EIG vs % increase
    # Calculate % increase from fractional designs (sum of fractions - 1)
    scatter_x = []  # % increase (sum of fractions - 1)
    scatter_y = []  # EIG values
    scatter_yerr = []  # EIG std
    scatter_colors = []
    scatter_edgecolors = []
    scatter_labels = []
    
    # Use reference EIG values if ref_id is provided, otherwise use values from first run
    eig_to_use_for_nominal = ref_nominal_eig if (ref_id is not None and ref_nominal_eig is not None) else nominal_eig
    std_to_use_for_nominal = ref_nominal_eig_std if (ref_id is not None and ref_nominal_eig_std is not None) else nominal_eig_std
    
    # Track which points are from bar chart (nominal + increasing designs) for fitted line
    bar_chart_x = []  # % increase for bar chart points
    bar_chart_y = []  # EIG for bar chart points
    
    # Add nominal design point if requested
    if include_nominal and fractional_nominal_design is not None and eig_to_use_for_nominal is not None:
        nominal_sum = np.sum(fractional_nominal_design)
        nominal_pct_increase = (nominal_sum - 1.0) * 100.0
        scatter_x.append(nominal_pct_increase)
        scatter_y.append(eig_to_use_for_nominal)
        scatter_yerr.append(std_to_use_for_nominal if isinstance(std_to_use_for_nominal, (int, float)) else None)
        scatter_colors.append('tab:blue')
        scatter_edgecolors.append('none')
        scatter_labels.append('Nominal Design')
        # Track for fitted line
        bar_chart_x.append(nominal_pct_increase)
        bar_chart_y.append(eig_to_use_for_nominal)
    
    # Add increasing design points (these are the bar chart points)
    for i, (design, eig, eig_std, color) in enumerate(zip(fractional_optimal_designs, optimal_eigs, optimal_eig_stds, colors)):
        if eig is not None:  # Only plot if EIG is available
            design_sum = np.sum(design)
            pct_increase = (design_sum - 1.0) * 100.0
            scatter_x.append(pct_increase)
            scatter_y.append(eig)
            scatter_yerr.append(eig_std if isinstance(eig_std, (int, float)) else None)
            scatter_colors.append(color)
            scatter_edgecolors.append('black')
            scatter_labels.append(labels[i] if labels else found_run_ids[i][:8])
            # Track for fitted line
            bar_chart_x.append(pct_increase)
            bar_chart_y.append(eig)
    
    # Add reference optimal EIG point if ref_id is provided
    if ref_id is not None and ref_optimal_eig is not None and ref_optimal_design is not None:
        # Calculate % increase for reference optimal design
        ref_design_sum = np.sum(ref_optimal_design)
        ref_pct_increase = (ref_design_sum - 1.0) * 100.0
        scatter_x.append(ref_pct_increase)
        scatter_y.append(ref_optimal_eig)
        scatter_yerr.append(ref_optimal_eig_std if isinstance(ref_optimal_eig_std, (int, float)) else None)
        scatter_colors.append('tab:orange')  # Different color to distinguish from nominal
        scatter_edgecolors.append('none')  # No outline for ref optimal
        scatter_labels.append(f'Optimal Design')
    
    # Create scatter plot figure
    if len(scatter_x) > 0 and any(y is not None for y in scatter_y):
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Filter out None values for plotting
        valid_indices = [i for i, y in enumerate(scatter_y) if y is not None]
        plot_x = [scatter_x[i] for i in valid_indices]
        plot_y = [scatter_y[i] for i in valid_indices]
        plot_yerr = [scatter_yerr[i] for i in valid_indices]
        plot_colors = [scatter_colors[i] for i in valid_indices]
        plot_edgecolors = [scatter_edgecolors[i] for i in valid_indices]
        plot_labels = [scatter_labels[i] for i in valid_indices]
        
        # Plot scatter points with same color coding and edge styles as bars
        legend_handles = {}
        legend_order = []
        for x, y, yerr, color, edgecolor, label in zip(plot_x, plot_y, plot_yerr, plot_colors, plot_edgecolors, plot_labels):
            # Only add label to legend if we haven't seen it before
            label_to_use = label if label not in legend_handles else ""
            handle = ax2.scatter(x, y, c=[color], s=100, alpha=0.7, edgecolors=edgecolor, 
                                 linewidths=1.5, label=label_to_use)
            if label_to_use:
                legend_handles[label_to_use] = handle
                legend_order.append(label_to_use)
            if yerr is not None and isinstance(yerr, (int, float)) and yerr > 0:
                ax2.errorbar(x, y, yerr=yerr, fmt='none', ecolor=color, elinewidth=1.2, capsize=4, alpha=0.8)
        
        # Add horizontal orange dotted line for ref optimal eig
        if ref_id is not None and ref_optimal_eig is not None:
            ax2.axhline(y=ref_optimal_eig, color='tab:orange', linestyle='--', 
                       linewidth=1.5, alpha=0.7)
        
        # Add black dotted fitted line for bar chart points (nominal + increasing designs)
        coeffs = None
        if len(bar_chart_x) >= 2 and len(bar_chart_y) >= 2:
            # Fit a line to the bar chart points
            bar_chart_x_arr = np.array(bar_chart_x)
            bar_chart_y_arr = np.array(bar_chart_y)
            # Fit polynomial (degree 1 = linear fit)
            coeffs = np.polyfit(bar_chart_x_arr, bar_chart_y_arr, 1)
            poly = np.poly1d(coeffs)
            # Generate x values for the line
            x_line = np.linspace(min(bar_chart_x_arr), max(bar_chart_x_arr), 100)
            y_line = poly(x_line)
            ax2.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7)
        elif len(bar_chart_x) == 1:
            # If only one point, just draw a horizontal line
            ax2.axhline(y=bar_chart_y[0], color='black', linestyle='--', 
                       linewidth=1.5, alpha=0.7)
        
        # Add vertical intersection line between ref optimal eig and fitted line when available
        if (
            ref_id is not None
            and ref_optimal_eig is not None
            and coeffs is not None
            and len(coeffs) >= 2
            and coeffs[0] != 0
        ):
            x_intersection = (ref_optimal_eig - coeffs[1]) / coeffs[0]
            if np.isfinite(x_intersection):
                ax2.axvline(x_intersection, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
                x_pct = x_intersection
                # Place text slightly above the intersection point
                ax2.text(
                    x_intersection,
                    ref_optimal_eig,
                    f'{x_pct:+.1f}%',
                    color='red',
                    fontsize=17,
                    fontweight='bold',
                    ha='left',
                    va='bottom',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.3)
                )
        
        # Build ordered legend handles (Nominal first, Ref Optimal second)
        ordered_labels = []
        if 'Nominal Design' in legend_handles:
            ordered_labels.append('Nominal Design')
        for label in legend_order:
            if label.startswith('Optimal Design') and label not in ordered_labels:
                ordered_labels.append(label)
        for label in legend_order:
            if label not in ordered_labels:
                ordered_labels.append(label)
        if ordered_labels:
            ordered_handles = [legend_handles[label] for label in ordered_labels]
            ax2.legend(ordered_handles, ordered_labels, loc='best', fontsize=12, framealpha=0.9)
        
        ax2.set_xlabel('Percentage Increase in Total Observations', fontsize=12, weight='bold')
        ax2.set_ylabel('Expected Information Gain [bits]', fontsize=12, weight='bold')
        ax2.set_title('EIG of Uniformly Increasing Observations', fontsize=16, weight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save scatter plot if save_path provided
        if save_path is not None:
            # Modify save path to add _scatter suffix
            scatter_save_path = save_path.replace('.png', '_scatter.png').replace('.pdf', '_scatter.pdf')
            if scatter_save_path == save_path:  # If no extension found, just append
                scatter_save_path = save_path + '_scatter'
            os.makedirs(os.path.dirname(scatter_save_path) if os.path.dirname(scatter_save_path) else '.', exist_ok=True)
            save_figure(scatter_save_path, fig=fig2, dpi=dpi)
            print(f"Saved scatter plot to {scatter_save_path}")
    else:
        fig2 = None
        ax2 = None
    
    return fig, ax
    

def compare_optimal_design(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        labels=None,
        step_key=None,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        colors=None,
        title=None,
        log_scale=True,
        display_mode='absolute',
        include_nominal=False
    ):
    """
    Compare optimal designs across multiple runs by loading eig_data JSON files.
    Can take either an experiment name or a list of run IDs.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): List of MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Cosmological experiment name (default: 'num_tracers')
        labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
        step_key (str or int, optional): Which step to use. If None, finds most recent eig_data file
        save_path (str, optional): Path to save the comparison plot. If None, doesn't save
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
        colors (list, optional): Colors for each run. If None, uses default color cycle
        title (str, optional): Custom title for the plot. If None, generates default title
        log_scale (bool): Whether to use log scale for y-axis (default: True)
        display_mode (str): How to display design values. Options: 'absolute' (total number of tracers per class),
            'fractional' (relative to total number), or 'ratio' (relative to nominal design values) (default: 'absolute')
        include_nominal (bool): Whether to include nominal design bars on the left (default: False)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Validate display_mode
    valid_modes = ['absolute', 'fractional', 'ratio']
    if display_mode not in valid_modes:
        raise ValueError(f"display_mode must be one of {valid_modes}, got '{display_mode}'")
    
    # Set MLflow tracking URI
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Get run data - can use either mlflow_exp or run_ids
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=True,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")
    
    # Extract run_ids from run_data_list
    run_ids = [run_data['run_id'] for run_data in run_data_list]
    
    # Build a map from run_id to exp_id
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    # Find and load JSON files for each run
    optimal_designs = []
    optimal_eigs = []
    found_run_ids = []
    design_labels = None
    nominal_eig = None  # Store nominal_eig from first run's data (for scatter plot)
    
    for run_id in run_ids:
        if run_id not in run_id_to_exp_id:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue
        
        exp_id = run_id_to_exp_id[run_id]
        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        # Load completed eig_data file
        try:
            json_path, data = load_eig_data_file(artifacts_dir, step_key=None)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue
        
        try:
            
            # Determine which step to use
            if step_key is not None:
                step_str = f"step_{step_key}"
            else:
                # Find the highest available step key
                step_keys = [k for k in data.keys() if k.startswith('step_')]
                if step_keys:
                    # Extract step numbers and find the highest
                    step_numbers = []
                    for k in step_keys:
                        try:
                            step_num = int(k.split('_')[1])
                            step_numbers.append((step_num, k))
                        except (ValueError, IndexError):
                            continue
                    if step_numbers:
                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                    else:
                        step_str = step_keys[0]  # Fallback to first if parsing fails
                else:
                    print(f"Warning: No step data found in {json_path}, skipping...")
                    continue
            
            if step_str not in data:
                print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                continue
            
            step_data = data[step_str]
            
            # Access nested structure
            variable_data = step_data.get('variable', {})
                
            # Extract optimal design and EIG from variable subdict
            if 'optimal_design' in variable_data and 'optimal_eig' in variable_data:
                optimal_design = np.array(variable_data['optimal_design'])
                optimal_eig_value = variable_data.get('optimal_eig', None)
                optimal_designs.append(optimal_design)
                optimal_eigs.append(optimal_eig_value)
                found_run_ids.append(run_id)
            else:
                print(f"Warning: No optimal_design data found in {json_path} for {step_str}, skipping...")
                continue
                
            # Get design labels from experiment (will be initialized later if needed)
            # For now, we'll get them from the experiment initialization below
                
        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    if len(optimal_designs) == 0:
        raise ValueError("No valid optimal design data found to compare")
    
    # Generate labels if not provided
    if labels is None:
        labels = [run_id[:8] for run_id in found_run_ids]
    
    if len(labels) != len(optimal_designs):
        labels = labels[:len(optimal_designs)]
    
    # Generate colors if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(len(optimal_designs))]
    else:
        if len(colors) < len(optimal_designs):
            colors = [colors[i % len(colors)] for i in range(len(optimal_designs))]
        else:
            colors = colors[:len(optimal_designs)]
    
    # Convert all colors to proper format (handles hex codes, numpy arrays, etc.)
    colors = [convert_color(c) for c in colors]
    
    # Initialize experiments for each run to get per-run nominal designs and other metadata
    nominal_total_obs_list = []
    nominal_designs = []
    design_labels = None
    needs_experiment = (display_mode in ['absolute', 'ratio'] or design_labels is None or include_nominal)
    
    if needs_experiment:
        device = "cuda:0"
        # Build a map from run_id to run_data for easy lookup
        run_id_to_run_data = {run_data['run_id']: run_data for run_data in run_data_list}
        
        # Initialize experiment for each found run to get its nominal design
        for run_id in found_run_ids:
            if run_id not in run_id_to_run_data:
                print(f"Warning: Run {run_id} not found in run_data_list, skipping experiment initialization...")
                nominal_total_obs_list.append(None)
                nominal_designs.append(None)
                continue
            
            run_data = run_id_to_run_data[run_id]
            try:
                experiment = init_experiment(
                    run_data['run_obj'], 
                    run_data['params'], 
                    device, 
                    design_args={}, 
                    global_rank=0
                )
                
                # Get nominal_total_obs for this run
                if display_mode == 'absolute' or display_mode == 'ratio':
                    nominal_total_obs_list.append(experiment.nominal_total_obs)
                else:
                    nominal_total_obs_list.append(None)
                
                # Get nominal_design for this run if needed
                if (display_mode == 'ratio' or include_nominal) and hasattr(experiment, 'nominal_design'):
                    nominal_designs.append(experiment.nominal_design.cpu().numpy())
                else:
                    nominal_designs.append(None)
                
                # Get design_labels from first successful experiment initialization
                if design_labels is None and hasattr(experiment, 'design_labels'):
                    design_labels = experiment.design_labels
                    
            except Exception as e:
                print(f"Warning: Could not initialize experiment for run {run_id}: {e}")
                nominal_total_obs_list.append(None)
                nominal_designs.append(None)
        
        # Check if we got the required data
        if display_mode == 'absolute' and all(nto is None for nto in nominal_total_obs_list):
            print(f"Warning: Could not get nominal_total_obs for any run. Cannot convert to absolute quantities.")
        if display_mode == 'ratio' and all(nd is None for nd in nominal_designs):
            print(f"Warning: Could not get nominal_design for any run. Cannot compute ratios.")
        if include_nominal and all(nd is None for nd in nominal_designs):
            print(f"Warning: Could not get nominal designs. Skipping nominal design bars.")
    
    # Use generic labels if design_labels not found
    if design_labels is None:
        # Infer from first design's dimensionality
        if len(optimal_designs) > 0:
            n_dims = len(optimal_designs[0])
            design_labels = [f'$f_{i}$' for i in range(n_dims)]
        else:
            design_labels = []
    
    # Check if designs are 1D or multi-dimensional
    if len(optimal_designs) > 0:
        n_dims = len(optimal_designs[0])
        is_1d = (n_dims == 1)
    else:
        is_1d = False
    
    # Convert design values based on display_mode
    # Note: optimal_designs and nominal_designs are stored as fractional values
    if display_mode == 'absolute':
        # Convert fractional to absolute quantities using per-run nominal_total_obs
        converted_designs = []
        for i, design in enumerate(optimal_designs):
            if i < len(nominal_total_obs_list) and nominal_total_obs_list[i] is not None:
                converted_designs.append(design * nominal_total_obs_list[i])
            else:
                print(f"Warning: nominal_total_obs not available for run {found_run_ids[i]}, keeping fractional values.")
                converted_designs.append(design)
        optimal_designs = converted_designs
        
        # Convert nominal designs to absolute if including them
        if include_nominal:
            for i, nominal_design in enumerate(nominal_designs):
                if nominal_design is not None and i < len(nominal_total_obs_list) and nominal_total_obs_list[i] is not None:
                    nominal_designs[i] = nominal_design * nominal_total_obs_list[i]
                    
    elif display_mode == 'ratio':
        # Convert to ratio relative to each run's nominal design
        converted_designs = []
        for i, design in enumerate(optimal_designs):
            if i < len(nominal_designs) and nominal_designs[i] is not None:
                # Avoid division by zero
                nominal_design_safe = np.where(nominal_designs[i] == 0, np.nan, nominal_designs[i])
                converted_designs.append(design / nominal_design_safe)
            else:
                print(f"Warning: nominal_design not available for run {found_run_ids[i]}, cannot compute ratio. Keeping fractional values.")
                converted_designs.append(design)
        optimal_designs = converted_designs
        
        # For nominal designs themselves in ratio mode, ratio is 1.0
        if include_nominal:
            for i, nominal_design in enumerate(nominal_designs):
                if nominal_design is not None:
                    nominal_designs[i] = np.ones_like(nominal_design)
    
    # For 'fractional' mode, no conversion needed
    
    # Determine y-axis label
    if is_1d:
        ylabel = design_labels[0] if design_labels else 'Design Value'
    else:
        if display_mode == 'absolute':
            ylabel = 'Number of Tracers'
        elif display_mode == 'fractional':
            ylabel = 'Fraction of Total Tracers'
        elif display_mode == 'ratio':
            ylabel = 'Ratio to Nominal Design'
        else:
            ylabel = 'Design Value'
    
    # Create figure
    if is_1d:
        # For 1D designs, use a simple bar plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        design_values = [design[0] for design in optimal_designs]
        plot_labels = list(labels) if labels else [run_id[:8] for run_id in found_run_ids]
        plot_colors = list(colors) if colors else [plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])] for i in range(len(optimal_designs))]
        
        # Add nominal designs if requested - prepend to lists
        if include_nominal:
            nominal_values = []
            nominal_labels = []
            for i, nominal_design in enumerate(nominal_designs):
                if nominal_design is not None:
                    nominal_values.append(nominal_design[0])
                    nominal_labels.append(f'Nominal ({found_run_ids[i][:8]})')
            if nominal_values:
                design_values = nominal_values + design_values
                plot_labels = nominal_labels + plot_labels
                # Use a distinct color for nominal designs (light blue/gray)
                nominal_colors = ['tab:blue'] * len(nominal_values)
                plot_colors = nominal_colors + plot_colors
        
        x_pos = np.arange(len(design_values))
        bars = ax.bar(x_pos, design_values, color=plot_colors, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Run', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        if log_scale:
            ax.set_yscale('log')
        
    else:
        # For multi-dimensional designs, use grouped bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(design_labels))  # Design dimension positions
        
        # Prepare data for plotting - prepend nominal designs if requested
        plot_designs = list(optimal_designs)
        plot_labels = list(labels) if labels else [run_id[:8] for run_id in found_run_ids]
        plot_colors = list(colors) if colors else [plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])] for i in range(len(optimal_designs))]
        
        if include_nominal:
            nominal_plot_designs = []
            nominal_plot_labels = []
            for i, nominal_design in enumerate(nominal_designs):
                if nominal_design is not None:
                    nominal_plot_designs.append(nominal_design)
                    nominal_plot_labels.append(f'Nominal ({found_run_ids[i][:8]})')
            if nominal_plot_designs:
                plot_designs = nominal_plot_designs + plot_designs
                plot_labels = nominal_plot_labels + plot_labels
                # Use a distinct color for nominal designs
                nominal_colors = ['tab:blue'] * len(nominal_plot_designs)
                plot_colors = nominal_colors + plot_colors
        
        n_designs_to_plot = len(plot_designs)
        width = 0.8 / n_designs_to_plot  # Width of bars
        
        '''
        # Get max possible tracers if experiment is available
        max_tracers = None
        if experiment is not None and hasattr(experiment, 'num_targets') and hasattr(experiment, 'design_labels'):
            try:
                # Use design_labels (which should match experiment.design_labels when available)
                # to ensure correct ordering
                if design_labels == experiment.design_labels:
                    # design_labels matches experiment.design_labels, use them directly
                    target_labels = design_labels
                else:
                    # Fall back to experiment.design_labels
                    target_labels = experiment.design_labels
                
                if use_fractional:
                    # Get maximum possible tracers as fractions
                    max_tracers = np.array([experiment.num_targets[target] for target in target_labels])
                    max_tracers = max_tracers / nominal_total_obs if nominal_total_obs is not None else max_tracers
                else:
                    # Get maximum possible tracers for each class
                    max_tracers = np.array([experiment.num_targets[target] for target in target_labels])
            except (KeyError, AttributeError) as e:
                print(f"Warning: Could not get max tracers from experiment: {e}")
                max_tracers = None
        
        # Add gray bars with dotted edges for maximum possible tracers (if available)
        if max_tracers is not None and len(max_tracers) == len(design_labels):
            # Use a reference width for max tracers bars (use the full width available)
            max_width = 0.8
            for i, max_val in enumerate(max_tracers):
                # Draw filled gray rectangle
                rect = Rectangle((x[i] - max_width/2, 0), max_width, max_val,
                               facecolor='gray', alpha=0.3, edgecolor='none', zorder=0)
                ax.add_patch(rect)
                # Manually draw dotted edges
                x_left = x[i] - max_width/2
                x_right = x[i] + max_width/2
                y_top = max_val
                # Top edge
                ax.plot([x_left, x_right], [y_top, y_top], 'k:', linewidth=1.5, alpha=0.7, zorder=1)
                # Left edge
                ax.plot([x_left, x_left], [0, y_top], 'k:', linewidth=1.5, alpha=0.7, zorder=1)
                # Right edge
                ax.plot([x_right, x_right], [0, y_top], 'k:', linewidth=1.5, alpha=0.7, zorder=1)
        '''
        # Plot bars for each design
        for i, (design, label, color) in enumerate(zip(plot_designs, plot_labels, plot_colors)):
            offset = (i - n_designs_to_plot/2 + 0.5) * width
            bars = ax.bar(x + offset, design, width, label=label, color=color, alpha=0.7, linewidth=1.5, zorder=2)
        
        '''
        # Add label for max possible tracers (manually since we used patches)
        if max_tracers is not None:
            max_label = 'Max Possible Tracers' if not use_fractional else 'Max Possible Fraction'
            ax.plot([], [], 'k:', linewidth=1.5, alpha=0.7, label=max_label)
        '''
        ax.set_xlabel('Tracer Class', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(design_labels, fontsize=11)
        ax.legend(loc='best', fontsize=16, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        if log_scale:
            ax.set_yscale('log')
    
    # Set title
    if title is None:
        if mlflow_exp and actual_mlflow_exp_for_title:
            title = f'Optimal Design Comparison - {actual_mlflow_exp_for_title}'
        elif mlflow_exp:
            title = f'Optimal Design Comparison - {mlflow_exp}'
        else:
            title = f'Optimal Design Comparison Across Runs'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved comparison plot to {save_path}")
    
    return fig, ax

def compare_best_designs(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        run_labels=None,
        step_key=None,
        top_n=5,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        design_labels=None,
        title=None,
        cmap='viridis',
        nominal_design=None
    ):
    """
    Compare top N best designs across multiple runs by loading eig_data JSON files.
    Designs are sorted by EIG and displayed as a heatmap using imshow, with visual
    separation between runs.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): List of MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Cosmological experiment name (default: 'num_tracers')
        run_labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
        step_key (str or int, optional): Which step to use. If None, finds most recent eig_data file
        top_n (int): Number of top designs to plot from each run (default: 5)
        save_path (str, optional): Path to save the comparison plot. If None, doesn't save
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
        design_labels (list, optional): Labels for each design dimension. If None, tries to infer from experiment or uses generic labels
        title (str, optional): Custom title for the plot. If None, generates default title
        cmap (str): Colormap to use for the heatmap (default: 'viridis'). If nominal_design is provided, a diverging colormap will be used.
        nominal_design (list, optional): Nominal design as a list of floats. If provided, colors will be plotted as ratios relative to this nominal design in each class/dimension (ratio = design_value / nominal_value). A ratio of 1.0 means equal to nominal.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    # Set MLflow tracking URI
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Get run data - can use either mlflow_exp or run_ids
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=True,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")
    
    # Extract run_ids from run_data_list
    run_ids = [run_data['run_id'] for run_data in run_data_list]
    
    # Build a map from run_id to exp_id
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    # Find and load JSON files for each run
    all_run_top_designs = []  # List of lists: each inner list contains top N designs for that run
    found_run_ids = []
    run_boundaries = []  # Track where each run's designs start/end for visual separation
    
    for run_id in run_ids:
        if run_id not in run_id_to_exp_id:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue
        
        exp_id = run_id_to_exp_id[run_id]
        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        # Load completed eig_data file
        try:
            json_path, data = load_eig_data_file(artifacts_dir, step_key=None)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue
        
        try:
            
            # Determine which step to use
            if step_key is not None:
                step_str = f"step_{step_key}"
            else:
                # Find the highest available step key
                step_keys = [k for k in data.keys() if k.startswith('step_')]
                if step_keys:
                    # Extract step numbers and find the highest
                    step_numbers = []
                    for k in step_keys:
                        try:
                            step_num = int(k.split('_')[1])
                            step_numbers.append((step_num, k))
                        except (ValueError, IndexError):
                            continue
                    if step_numbers:
                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                    else:
                        step_str = step_keys[0]  # Fallback to first if parsing fails
                else:
                    print(f"Warning: No step data found in {json_path}, skipping...")
                    continue
            
            if step_str not in data:
                print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                continue
            
            step_data = data[step_str]
            
            # Access nested structure
            variable_data = step_data.get('variable', {})
            
            # Extract designs from top-level input_designs and EIGs from variable subdict
            if 'input_designs' not in data:
                print(f"Warning: No input_designs found in {json_path}, skipping...")
                continue
            
            if 'eigs_avg' not in variable_data:
                print(f"Warning: No eigs_avg data found in {json_path} for {step_str}, skipping...")
                continue
            
            designs = np.array(data['input_designs'])
            eigs = np.array(variable_data['eigs_avg'])
            
            # Check if designs and eigs have matching lengths
            if len(designs) != len(eigs):
                print(f"Warning: Mismatch - {len(designs)} designs but {len(eigs)} EIG values for run {run_id}")
            
            # Sort designs by EIG (descending) and take top N
            sorted_indices = np.argsort(eigs)[::-1]
            top_n_indices = sorted_indices[:min(top_n, len(designs))]
            top_designs = designs[top_n_indices]
            
            # Ensure designs are 2D (n_designs, n_dims)
            if top_designs.ndim == 1:
                top_designs = top_designs.reshape(1, -1)
            
            all_run_top_designs.append(top_designs)
            found_run_ids.append(run_id)
                
        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    if len(all_run_top_designs) == 0:
        raise ValueError("No valid design data found to compare")
    
    # Combine all designs into a single array
    combined_designs = np.concatenate(all_run_top_designs, axis=0)
    
    # Calculate run boundaries for visual separation
    current_idx = 0
    for i, run_designs in enumerate(all_run_top_designs):
        run_boundaries.append((current_idx, current_idx + len(run_designs)))
        current_idx += len(run_designs)
    
    # Generate labels if not provided
    if run_labels is None:
        run_labels = [run_id[:8] for run_id in found_run_ids]
    
    if len(run_labels) != len(all_run_top_designs):
        run_labels = run_labels[:len(all_run_top_designs)]
    
    # Determine design labels
    if design_labels is None:
        # Try to infer from first run's experiment
        try:
            first_run_data = run_data_list[0]
            device = "cuda:0"
            experiment = init_experiment(
                first_run_data['run_obj'], 
                first_run_data['params'], 
                device, 
                design_args={}, 
                global_rank=0
            )
            if hasattr(experiment, 'design_labels'):
                design_labels = experiment.design_labels
        except Exception as e:
            print(f"Warning: Could not initialize experiment to get design labels: {e}")
    
    # Use generic labels if still None
    if design_labels is None:
        n_dims = combined_designs.shape[1]
        design_labels = [f'$f_{i}$' for i in range(n_dims)]
    
    if len(design_labels) != combined_designs.shape[1]:
        raise ValueError(f"design_labels length ({len(design_labels)}) must match number of design dimensions ({combined_designs.shape[1]})")
    
    # Handle nominal design if provided
    plot_data = combined_designs.copy()
    use_relative_colors = False
    if nominal_design is not None:
        nominal_design = np.array(nominal_design)
        if len(nominal_design) != combined_designs.shape[1]:
            raise ValueError(f"nominal_design length ({len(nominal_design)}) must match number of design dimensions ({combined_designs.shape[1]})")
        
        # Check for zeros in nominal_design to avoid division by zero
        if np.any(nominal_design == 0):
            raise ValueError("nominal_design cannot contain zero values (division by zero)")
        
        # Compute ratios relative to nominal for each dimension
        # Ensure proper broadcasting: combined_designs is (n_designs, n_dims), nominal_design is (n_dims,)
        # This will broadcast correctly: each row of combined_designs is divided element-wise by nominal_design
        # Ratio = design_value / nominal_value
        # Ratio > 1.0 means design is larger than nominal, Ratio < 1.0 means design is smaller than nominal
        plot_data = combined_designs / nominal_design[np.newaxis, :]
        use_relative_colors = True
        
        # Use a diverging colormap if not explicitly overridden
        if cmap == 'viridis':  # Only change default, allow user override
            cmap = 'RdBu'  # Red for larger ratios (>1.0), Blue for smaller ratios (<1.0)
    
    # Create figure with space for colorbar
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.06, right=0.99)
    
    ax = fig.add_subplot(111)
    
    # Plot heatmap using imshow (transpose so each column is a design, rows are dimensions)
    if use_relative_colors:
        # Simple ratio: designs / nominal_design
        # Ratio > 1.0 means larger than nominal, Ratio < 1.0 means smaller than nominal
        # Use TwoSlopeNorm to center the colormap at 1.0
        from matplotlib.colors import TwoSlopeNorm
        vmin = plot_data.min()
        vmax = plot_data.max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        
        im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap, norm=norm)
    else:
        im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap)
    
    # Add vertical lines to separate runs
    for start_idx, end_idx in run_boundaries:
        if start_idx > 0:  # Don't draw line at the very beginning
            ax.axvline(start_idx - 0.5, color='white', linewidth=2, linestyle='-', alpha=0.8)
    
    # Set y-axis labels (use design_labels as-is - user can format with $ for LaTeX if desired)
    ax.set_yticks(np.arange(len(design_labels)))
    ax.set_yticklabels(design_labels, fontsize=14)
    ax.set_ylabel('Design Dimension', fontsize=12, weight='bold')
    
    # Set x-axis ticks and labels for run groups
    x_tick_positions = []
    x_tick_labels = []
    for i, (start_idx, end_idx) in enumerate(run_boundaries):
        # Position tick at the middle of each run's chunk
        tick_pos = (start_idx + end_idx - 1) / 2
        x_tick_positions.append(tick_pos)
        x_tick_labels.append(run_labels[i] if i < len(run_labels) else found_run_ids[i][:8])
    
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    
    # Set x-axis limits
    ax.set_xlim(-0.5, len(combined_designs) - 0.5)
    
    # Add colorbar (let matplotlib manage placement)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if use_relative_colors:
        cbar.set_label('Ratio to Nominal Design', labelpad=10, fontsize=12, weight='bold')
    else:
        cbar.set_label('Design Value', labelpad=10, fontsize=12, weight='bold')
    
    # Configure axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', length=0)
    ax.margins(x=0)
    
    # Set title
    if title is None:
        if mlflow_exp and actual_mlflow_exp_for_title:
            title = f'Top {top_n} Designs Comparison - {actual_mlflow_exp_for_title}'
        elif mlflow_exp:
            title = f'Top {top_n} Designs Comparison - {mlflow_exp}'
        else:
            title = f'Top {top_n} Designs Comparison Across Runs'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved comparison plot to {save_path}")
    
    return fig, ax

def plot_design_dim_by_eig(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        run_labels=None,
        step_key=None,
        save_path=None,
        figsize=(16, 10),
        dpi=400,
        design_labels=None,
        title=None,
        nominal_design=None,
        colors=None,
        moving_avg_window=50,
        top_n=None,
        ratio=False,
        show_nominal=False
    ):
    """
    Plot eig values for each design dimension across multiple runs, sorted by EIG.
    Creates one subplot per dimension showing the eig values.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): List of MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Cosmological experiment name (default: 'num_tracers')
        run_labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
        step_key (str or int, optional): Which step to use. If None, finds most recent eig_data file
        save_path (str, optional): Path to save the plot. If None, doesn't save
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
        design_labels (list, optional): Labels for each design dimension. If None, tries to infer from experiment or uses generic labels
        title (str, optional): Custom title for the plot. If None, generates default title
        nominal_design (list, optional): Nominal design as a list of floats. Required if ratio=True.
        colors (list, optional): Colors for each run. If None, uses default color cycle
        moving_avg_window (int): Window size for moving average smoothing. If None or 0, no smoothing is applied. Default: 50
        top_n (int, optional): If specified, only plot the top N designs (by EIG) from each run. If None, plots all designs. Default: None
        ratio (bool): If True, plot ratios to nominal design. If False, plot actual design fractions. Default: False
        show_nominal (bool): If True, show a horizontal line at the nominal value (1.0 for ratios, or actual nominal value for fractions). Default: False
    
    Returns:
        fig, axes: Matplotlib figure and axes objects (one axis per tracer)
    """
    if ratio and nominal_design is None:
        raise ValueError("nominal_design is required when ratio=True")
    
    # Set MLflow tracking URI
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Get run data - can use either mlflow_exp or run_ids
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=True,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")
    
    # Extract run_ids from run_data_list
    run_ids = [run_data['run_id'] for run_data in run_data_list]
    
    # Build a map from run_id to exp_id
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    # Find and load JSON files for each run
    all_run_data = []  # List of dicts: each dict contains designs, eigs, and run_id for that run
    found_run_ids = []
    
    for run_id in run_ids:
        if run_id not in run_id_to_exp_id:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue
        
        exp_id = run_id_to_exp_id[run_id]
        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        # Load completed eig_data file
        try:
            json_path, data = load_eig_data_file(artifacts_dir, step_key=None)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue
        
        try:
            
            # Determine which step to use
            if step_key is not None:
                step_str = f"step_{step_key}"
            else:
                # Find the highest available step key
                step_keys = [k for k in data.keys() if k.startswith('step_')]
                if step_keys:
                    # Extract step numbers and find the highest
                    step_numbers = []
                    for k in step_keys:
                        try:
                            step_num = int(k.split('_')[1])
                            step_numbers.append((step_num, k))
                        except (ValueError, IndexError):
                            continue
                    if step_numbers:
                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                    else:
                        step_str = step_keys[0]  # Fallback to first if parsing fails
                else:
                    print(f"Warning: No step data found in {json_path}, skipping...")
                    continue
            
            if step_str not in data:
                print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                continue
            
            step_data = data[step_str]
            
            # Access nested structure
            variable_data = step_data.get('variable', {})
            
            # Extract designs from top-level input_designs and EIGs from variable subdict
            if 'input_designs' not in data:
                print(f"Warning: No input_designs found in {json_path}, skipping...")
                continue
            
            if 'eigs_avg' not in variable_data:
                print(f"Warning: No eigs_avg data found in {json_path} for {step_str}, skipping...")
                continue
            
            designs = np.array(data['input_designs'])
            eigs = np.array(variable_data['eigs_avg'])
            
            # Check if designs and eigs have matching lengths
            if len(designs) != len(eigs):
                print(f"Warning: Mismatch - {len(designs)} designs but {len(eigs)} EIG values for run {run_id}")
                continue
            
            # Ensure designs are 2D (n_designs, n_dims)
            if designs.ndim == 1:
                designs = designs.reshape(-1, 1)
            
            all_run_data.append({
                'run_id': run_id,
                'designs': designs,
                'eigs': eigs
            })
            found_run_ids.append(run_id)
                
        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    if len(all_run_data) == 0:
        raise ValueError("No valid design data found to compare")
    
    # Determine design labels
    if design_labels is None:
        # Try to infer from first run's experiment
        try:
            first_run_data = run_data_list[0]
            device = "cuda:0"
            experiment = init_experiment(
                first_run_data['run_obj'], 
                first_run_data['params'], 
                device, 
                design_args={}, 
                global_rank=0
            )
            if hasattr(experiment, 'design_labels'):
                design_labels = experiment.design_labels
        except Exception as e:
            print(f"Warning: Could not initialize experiment to get design labels: {e}")
    
    # Use generic labels if still None
    if design_labels is None:
        n_dims = all_run_data[0]['designs'].shape[1]
        design_labels = [f'$f_{i}$' for i in range(n_dims)]
    
    # Verify all runs have the same number of dimensions
    n_dims = all_run_data[0]['designs'].shape[1]
    for run_data in all_run_data:
        if run_data['designs'].shape[1] != n_dims:
            raise ValueError(f"Design dimension mismatch: run {run_data['run_id']} has {run_data['designs'].shape[1]} dimensions, expected {n_dims}")
    
    if len(design_labels) != n_dims:
        raise ValueError(f"design_labels length ({len(design_labels)}) must match number of design dimensions ({n_dims})")
    
    # Validate nominal_design if ratio is True
    if ratio:
        nominal_design = np.array(nominal_design)
        if len(nominal_design) != n_dims:
            raise ValueError(f"nominal_design length ({len(nominal_design)}) must match number of design dimensions ({n_dims})")
        
        if np.any(nominal_design == 0):
            raise ValueError("nominal_design cannot contain zero values (division by zero)")
    
    # Generate labels if not provided
    if run_labels is None:
        run_labels = [run_id[:8] for run_id in found_run_ids]
    
    if len(run_labels) != len(all_run_data):
        run_labels = run_labels[:len(all_run_data)]
    
    # Generate colors if not provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Cycle colors if we have more runs than colors
        colors = [colors[i % len(colors)] for i in range(len(all_run_data))]
    elif len(colors) < len(all_run_data):
        colors = [colors[i % len(colors)] for i in range(len(all_run_data))]
    else:
        colors = colors[:len(all_run_data)]
    
    # Sort designs by EIG within each run and compute ratios
    sorted_run_data = []
    for run_data in all_run_data:
        designs = run_data['designs']
        eigs = run_data['eigs']
        
        # Sort by EIG (descending) within this run
        sorted_indices = np.argsort(eigs)[::-1]
        
        # If top_n is specified, only take top N designs
        if top_n is not None and top_n > 0:
            sorted_indices = sorted_indices[:min(top_n, len(sorted_indices))]
        
        sorted_designs = designs[sorted_indices]
        
        # Compute ratios or use actual design values
        if ratio:
            plot_values = sorted_designs / nominal_design[np.newaxis, :]
        else:
            plot_values = sorted_designs
        
        sorted_run_data.append({
            'plot_values': plot_values,
            'eigs': eigs[sorted_indices]
        })
    
    # Create figure with subplots - one per tracer/dimension
    n_cols = min(2, n_dims)  # 2 columns
    n_rows = (n_dims + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Plot each tracer/dimension
    for dim_idx in range(n_dims):
        ax = axes[dim_idx]
        
        # Plot each run as a line
        for run_idx, run_sorted_data in enumerate(sorted_run_data):
            plot_values = run_sorted_data['plot_values'][:, dim_idx]
            n_designs = len(plot_values)
            design_indices = np.arange(n_designs)
            
            # Apply moving average if window size is specified
            if moving_avg_window is not None and moving_avg_window > 0 and n_designs > moving_avg_window:
                # Use manual moving average that handles edges properly
                # At each point, use the available window (may be smaller at edges)
                values_smoothed = np.zeros_like(plot_values)
                half_window = moving_avg_window // 2
                
                for i in range(n_designs):
                    # Determine the actual window bounds for this point
                    start_idx = max(0, i - half_window)
                    end_idx = min(n_designs, i + half_window + 1)
                    # Compute average over the available window
                    values_smoothed[i] = np.mean(plot_values[start_idx:end_idx])
            else:
                values_smoothed = plot_values
            
            ax.plot(
                design_indices,
                values_smoothed,
                label=run_labels[run_idx],
                color=colors[run_idx],
                linewidth=2,
                alpha=0.8
            )
        
        # Add horizontal line at ratio = 1.0 (nominal) or nominal value if ratio mode
        if show_nominal:
            if ratio:
                ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal' if dim_idx == 0 else '')
            elif nominal_design is not None:
                nominal_value = np.array(nominal_design)[dim_idx]
                ax.axhline(y=nominal_value, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal' if dim_idx == 0 else '')
        
        ax.set_xlabel('Design Index (sorted by EIG per run)', fontsize=12, weight='bold')
        if ratio:
            ax.set_ylabel(f'Ratio to Nominal', fontsize=12, weight='bold')
        else:
            ax.set_ylabel(f'Design Fraction', fontsize=12, weight='bold')
        ax.set_title(f'{design_labels[dim_idx]}', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Hide unused subplots
    for dim_idx in range(n_dims, len(axes)):
        axes[dim_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved plot to {save_path}")
    
    return fig, axes
    
def compare_training(
        mlflow_exp=None,
        run_ids=None,
        var=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        log_scale=False,
        loss_step_freq=10,
        start_step=0,
        area_step_freq=100,
        lr_step_freq=1,
        loss_limits=None,
        show_area=True,
        param_pair=None,
        area_limits=None,
        show_lr=True,
        dpi=300,
        colors=None,
        step_range=None
        ):
    """
    Compares training loss, learning rate, and posterior contour area evolution
    for multiple MLflow runs using three vertically stacked subplots sharing the x-axis.
    Top plot: Loss (using plot_training).
    Middle plot: Learning Rate (using plot_training).
    Bottom plot: Parameter pair area comparison (if param_pair is specified).
    Can take either an experiment name or a list of run IDs.

    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): Individual or list of specific MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        var (str or list): Parameter(s) from MLflow run params to include in the label.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Experiment name or ID (if needed for path).
        log_scale (bool): If True, use log scale for the y-axes (Loss, LR, Area). Loss values <= 0 will be omitted in log scale.
        loss_step_freq (int): Sampling frequency for plotting loss points.
        start_step (int or list): Starting step offset for x-axis. If list, must be same length as run_id.
        area_step_freq (int): Sampling frequency for plotting nominal area points (must be multiple of 100).
        lr_step_freq (int): Sampling frequency for plotting learning rate points.
        show_area (bool): If True, show the area subplot. Default is True.
        show_lr (bool): If True, show the learning rate subplot. Default is True.
        param_pair (str, optional): Parameter pair to compare in format "param1,param2" (e.g., "Om,hrdrag").
        dpi (int): DPI for saving the figure.
        colors (list, optional): List of colors to use for each group. If None, uses default matplotlib colors.
        step_range (tuple, optional): Tuple of (min_step, max_step) to limit the x-axis range. If None, plots all available steps.
    """
    # Set MLflow tracking URI before creating client
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()

    run_data_list, experiment_id_for_save_path, _ = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=False,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        return

    # Convert start_step to list if it's a single value
    if isinstance(start_step, (int, float)):
        start_step_list = [start_step] * len(run_data_list)
    elif isinstance(start_step, list):
        if len(start_step) != len(run_data_list):
            raise ValueError(f"start_step list length ({len(start_step)}) must match number of runs ({len(run_data_list)})")
        start_step_list = start_step
    else:
        raise ValueError("start_step must be an int, float, or list")

    # Convert var to list if it's a single variable for sorting
    vars_list = var if isinstance(var, list) else [var] if var is not None else []
    
    # Group runs by their variable values to assign colors
    grouped_runs = {}
    for run_data_item in run_data_list:
        current_params = run_data_item['params']
        group_key = []
        is_valid_for_grouping = True
        
        # If var is None, treat each run as its own group using run_id
        if not vars_list:
            group_key_tuple = (run_data_item['run_id'],)
            if group_key_tuple not in grouped_runs:
                grouped_runs[group_key_tuple] = []
            grouped_runs[group_key_tuple].append(run_data_item)
        else:
            # Original logic for when var is specified
            for v_key in vars_list:
                if v_key in current_params:
                    value = current_params[v_key]
                    group_key.append(value)
                else:
                    is_valid_for_grouping = False
                    break
            
            if is_valid_for_grouping:
                group_key_tuple = tuple(group_key)
                if group_key_tuple not in grouped_runs:
                    grouped_runs[group_key_tuple] = []
                grouped_runs[group_key_tuple].append(run_data_item)

    # Sort groups for consistent ordering - descending for numerical, alphabetical for text
    def custom_sort_key(group_key):
        # First get the sort key using your existing function
        base_sort_key = sort_key_for_group_tuple(group_key)
        
        # Then modify numerical values to be negative for descending order
        if isinstance(base_sort_key, tuple):
            modified_values = []
            for val in base_sort_key:
                if isinstance(val, (int, float)):
                    # Use negative value for descending order
                    modified_values.append(-val)
                else:
                    # Keep string values as-is for alphabetical ordering
                    modified_values.append(val)
            return tuple(modified_values)
        else:
            # Single value case
            if isinstance(base_sort_key, (int, float)):
                return -base_sort_key
            else:
                return base_sort_key
    
    sorted_group_keys = sorted(grouped_runs.keys(), key=custom_sort_key)
    
    # Create a mapping of run_id to its group key for color assignment
    run_to_group = {}
    for run_data_item in run_data_list:
        run_id_iter = run_data_item['run_id']
        current_params = run_data_item['params']
        group_key = []
        is_valid_for_grouping = True
        
        # If var is None, each run is its own group
        if not vars_list:
            run_to_group[run_id_iter] = (run_id_iter,)
        else:
            # Original logic for when var is specified
            for v_key in vars_list:
                if v_key in current_params:
                    group_key.append(current_params[v_key])
                else:
                    is_valid_for_grouping = False
                    break
            
            if is_valid_for_grouping:
                run_to_group[run_id_iter] = tuple(group_key)
            else:
                run_to_group[run_id_iter] = None

    # --- Data Fetching (Metrics) ---
    all_metrics_for_runs = {}  # Keyed by run_id
    min_loss_overall = float('inf')
    max_loss_overall = float('-inf')
    min_lr_overall = float('inf')
    max_lr_overall = float('-inf')
    min_area_overall = float('inf')
    max_area_overall = float('-inf')

    valid_runs_processed_for_metrics = []

    for i, run_data_item in enumerate(run_data_list):
        run_id_iter = run_data_item['run_id']
        run_params = run_data_item['params']
        current_start_step = start_step_list[i]

        try:
            # Fetch metrics
            loss_hist_raw = client.get_metric_history(run_id_iter, 'loss')
            lr_hist_raw = client.get_metric_history(run_id_iter, 'lr')
            
            # Process and filter NaNs/Infs
            loss = [(m.step + current_start_step, m.value) for m in loss_hist_raw if np.isfinite(m.value)]
            lr = [(m.step + current_start_step, m.value) for m in lr_hist_raw if np.isfinite(m.value)]

            if not loss:
                print(f"Warning: No valid loss points found for run {run_id_iter}. Skipping.")
                continue

            # Apply step range filtering if specified
            if step_range is not None:
                min_step, max_step = step_range
                loss = [(step, value) for step, value in loss if min_step <= step <= max_step]
                lr = [(step, value) for step, value in lr if min_step <= step <= max_step]
                
                if not loss:
                    print(f"Warning: No loss points found in step range {step_range} for run {run_id_iter}. Skipping.")
                    continue

            all_metrics_for_runs[run_id_iter] = {
                'loss': loss,
                'lr': lr,
                'params': run_params
            }
            valid_runs_processed_for_metrics.append(run_data_item)

            # Update overall min/max for axis scaling
            run_losses = [v for s, v in loss]
            run_lrs = [v for s, v in lr]

            if run_losses:
                min_loss_overall = min(min_loss_overall, np.min(run_losses))
                max_loss_overall = max(max_loss_overall, np.max(run_losses))
            if run_lrs:
                min_lr_overall = min(min_lr_overall, np.min(run_lrs))
                max_lr_overall = max(max_lr_overall, np.max(run_lrs))

        except Exception as e:
            print(f"Error processing metrics for run {run_id_iter}: {e}. Skipping.")
            continue

    if not valid_runs_processed_for_metrics:
        print("No runs with valid data to plot.")
        return

    # --- Plotting Setup ---
    # Calculate number of subplots based on what we want to show
    num_subplots = 1  # Always show loss
    if show_area and param_pair:
        num_subplots += 1
    if show_lr:
        num_subplots += 1
    
    # Dynamic sizing based on number of subplots
    if num_subplots == 1:
        fig_width, fig_height = 12, 8
    elif num_subplots == 2:
        fig_width, fig_height = 14, 10
    else:  # 3 subplots
        fig_width, fig_height = 16, 14
    
    fig, axes = plt.subplots(num_subplots, 1, figsize=(fig_width, fig_height), sharex=True)
    
    # If only one subplot, axes won't be an array
    if num_subplots == 1:
        axes = np.array([axes])
    
    # Assign axes based on what we're showing
    current_ax = 0
    ax1 = axes[current_ax]  # Loss is always first
    current_ax += 1
    
    ax_area = None
    if show_area and param_pair:
        ax_area = axes[current_ax]
        current_ax += 1
    
    ax_lr = None
    if show_lr:
        ax_lr = axes[current_ax]

    # Use provided colors or default matplotlib colors
    if colors is not None:
        if len(colors) < len(sorted_group_keys):
            print(f"Warning: Only {len(colors)} colors provided for {len(sorted_group_keys)} groups. Repeating colors.")
        group_colors = {group_key: colors[i % len(colors)] for i, group_key in enumerate(sorted_group_keys)}
    else:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        group_colors = {group_key: default_colors[i % len(default_colors)] for i, group_key in enumerate(sorted_group_keys)}
    
    for run_id, group_key in run_to_group.items():
        if group_key is not None:
            color = group_colors.get(group_key, 'gray')

    num_runs = len(valid_runs_processed_for_metrics)
    if num_runs <= 5:
        base_alpha = 0.8
    elif num_runs <= 10:
        base_alpha = 0.6
    elif num_runs <= 20:
        base_alpha = 0.4
    else:
        base_alpha = 0.3

    # --- Plotting Data ---
    # Track which groups we've already added to the legend
    legend_entries_added = set()

    # Process runs in sorted group order to ensure legend matches sorted order
    processed_runs = []
    for group_key in sorted_group_keys:
        # Find all runs that belong to this group
        group_runs = [run for run in valid_runs_processed_for_metrics 
                     if run_to_group.get(run['run_id']) == group_key]
        processed_runs.extend(group_runs)
    
    # Add any remaining runs that don't belong to any group
    remaining_runs = [run for run in valid_runs_processed_for_metrics 
                     if run_to_group.get(run['run_id']) is None]
    processed_runs.extend(remaining_runs)

    for i, run_data_item in enumerate(processed_runs):
        run_id_iter = run_data_item['run_id']
        run_params = run_data_item['params']
        metrics = all_metrics_for_runs[run_id_iter]
        
        # Find the original index for start_step_list
        original_index = valid_runs_processed_for_metrics.index(run_data_item)
        current_start_step = start_step_list[original_index]

        # Determine group key and color using the pre-computed mapping
        group_key_tuple = run_to_group.get(run_id_iter)
        is_valid_for_grouping = group_key_tuple is not None
        
        if is_valid_for_grouping:
            color = group_colors.get(group_key_tuple, 'gray')  # Use gray for ungrouped runs
        else:
            color = 'gray'  # Use gray for runs that don't match grouping criteria

        # Create label from group key
        if not vars_list:
            # When var is None, use run_id as the label
            base_label = run_id_iter[:8]
        else:
            # Original logic for when var is specified
            label_parts = [run_id_iter[:8]]  # Start with run_id
            for v_key in vars_list:
                if v_key in run_params:
                    label_parts.append(f"{v_key}={run_params[v_key]}")
            base_label = ", ".join(label_parts) if label_parts else run_id_iter[:8]

        # Only add label if this group hasn't been added to legend yet
        should_add_label = group_key_tuple not in legend_entries_added if is_valid_for_grouping else run_id_iter not in legend_entries_added
        plot_label = base_label if should_add_label else None

        # --- Plot Loss (ax1) ---
        loss_data = metrics['loss']
        if loss_data:
            loss_steps, loss_values = zip(*loss_data)
            sampled_indices = np.arange(0, len(loss_steps), loss_step_freq)
            plot_loss_steps = np.array(loss_steps)[sampled_indices]
            plot_loss_values = np.array(loss_values)[sampled_indices]

            if log_scale:
                ax1.plot(plot_loss_steps, plot_loss_values - min_loss_overall, alpha=base_alpha, color=color, label=plot_label)
            else:
                ax1.plot(plot_loss_steps, plot_loss_values, alpha=base_alpha, color=color, label=plot_label)

        # --- Plot Parameter Pair Area (ax2) ---
        if show_area and param_pair:
            if area_step_freq % 100 != 0:
                print("Warning: area_step_freq should ideally be a multiple of 100 as nominal_area is logged every 100 steps.")
            sampling_rate = max(1, area_step_freq // 100)

            # Parse parameter pair
            param1, param2 = param_pair.split(',')
            param1 = param1.strip()
            param2 = param2.strip()
            
            # Get area data for this specific parameter pair
            area_metric_name = f"nominal_area_avg_{param1}_{param2}"
            
            try:
                area_hist_raw = client.get_metric_history(run_id_iter, area_metric_name)
                area_data = [(m.step + current_start_step, m.value) for m in area_hist_raw if np.isfinite(m.value)]
                
                # Apply step range filtering if specified
                if step_range is not None:
                    min_step, max_step = step_range
                    area_data = [(step, value) for step, value in area_data if min_step <= step <= max_step]
                
                if area_data:
                    area_steps, area_values = zip(*area_data)
                    sampled_indices = np.arange(0, len(area_steps), sampling_rate)
                    plot_area_steps = np.array(area_steps)[sampled_indices]
                    plot_area_values = np.array(area_values)[sampled_indices]

                    # Get nominal samples and area for comparison
                    try:
                        nominal_samples, target_labels, latex_labels = load_nominal_samples(run_params['cosmo_exp'], run_params['cosmo_model'], dataset=run_params['dataset'])
                        with contextlib.redirect_stdout(io.StringIO()):
                            nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
                        nominal_area = get_contour_area([nominal_samples_gd], 0.68, param1, param2)[0]["nominal_area_"+f"{param1}_{param2}"]
                        ax_area.plot(plot_area_steps, plot_area_values/nominal_area, 
                                    alpha=base_alpha, color=color, label=plot_label)
                        ax_area.axhline(1, color='black', linestyle='--', lw=1.5, alpha=0.7, label='Nominal Area')
                    except NotImplementedError:
                        ax_area.plot(plot_area_steps, plot_area_values, 
                                    alpha=base_alpha, color=color, label=plot_label)
                    
            except Exception as e:
                print(f"Warning: Could not fetch area data for {area_metric_name} in run {run_id_iter}: {e}")

        # --- Plot Learning Rate (ax3) ---
        if show_lr:
            lr_data = metrics['lr']
            if lr_data:
                lr_steps, lr_values = zip(*lr_data)
                sampled_indices = np.arange(0, len(lr_steps), lr_step_freq)
                plot_lr_steps = np.array(lr_steps)[sampled_indices]
                plot_lr_values = np.array(lr_values)[sampled_indices]

                if log_scale:
                    positive_mask = plot_lr_values > 0
                    plot_lr_steps_filtered = plot_lr_steps[positive_mask]
                    plot_lr_values_filtered = plot_lr_values[positive_mask]
                    ax_lr.plot(plot_lr_steps_filtered, plot_lr_values_filtered, alpha=base_alpha, color=color, label=plot_label)
                else:
                    ax_lr.plot(plot_lr_steps, plot_lr_values, alpha=base_alpha, color=color, label=plot_label)

        # Add this group/run to the set of legend entries
        if is_valid_for_grouping:
            legend_entries_added.add(group_key_tuple)
        else:
            legend_entries_added.add(run_id_iter)

    # --- Final Plot Configuration ---
    # Determine legend font size based on number of runs
    num_runs = len(valid_runs_processed_for_metrics)
    if num_runs > 10: legend_fontsize = 'xx-small'
    elif num_runs > 6: legend_fontsize = 'x-small'
    elif num_runs > 4: legend_fontsize = 'small'
    else: legend_fontsize = 'medium'

    # Configure ax1 (Loss)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=legend_fontsize, title="Run Args")
    if loss_limits:
        ax1.set_ylim(loss_limits)

    if show_area and param_pair:
        
        # Configure ax2 (Parameter Pair Area)
        ax_area.set_ylabel(f"Nominal Design Area Ratio to DESI - {param_pair}")
        ax_area.tick_params(axis='y')
        ax_area.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax_area.set_ylim(area_limits)

    # Configure ax3 (Learning Rate)
    if show_lr:
        ax_lr.set_xlabel("Training Step")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.tick_params(axis='y')
        ax_lr.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Apply log scale if requested
    if log_scale:
        ax1.set_yscale('log')
    else:
        # Set linear scale limits with padding
        if max_loss_overall > min_loss_overall:
            loss_pad = (max_loss_overall - min_loss_overall) * 0.05
            ax1.set_ylim(min_loss_overall - loss_pad, max_loss_overall + loss_pad)
        elif np.isfinite(min_loss_overall):
            ax1.set_ylim(min_loss_overall - 0.5, min_loss_overall + 0.5)

        if show_lr:
            if max_lr_overall > min_lr_overall:
                lr_pad = (max_lr_overall - min_lr_overall) * 0.05
                ax_lr.set_ylim(min_lr_overall - lr_pad, max_lr_overall + lr_pad)
            elif np.isfinite(min_lr_overall):
                ax_lr.set_ylim(min_lr_overall * 0.9, min_lr_overall * 1.1)

    # Set x-axis limits if step_range is specified
    if step_range is not None:
        min_step, max_step = step_range
        ax1.set_xlim(min_step, max_step)

    # Adjust layout
    title = f"Training History - {mlflow_exp}" if mlflow_exp else f"Training History - {len(run_data_list)} Run(s)"
    if param_pair:
        title += f" - Area: {param_pair}"
    fig.set_constrained_layout(True)
    fig.suptitle(title, fontsize=16)

    # Determine save path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = None
    save_path = None

    last_exp_id_for_path = experiment_id_for_save_path

    if mlflow_exp and last_exp_id_for_path:
        save_dir = f"{storage_path}/mlruns/{last_exp_id_for_path}/plots"
        save_path = f"{save_dir}/training_comparison_{timestamp}.png"
    elif len(run_data_list) == 1:
        single_run_id = run_data_list[0]['run_id']
        if not last_exp_id_for_path:
            try:
                single_run_data_item = run_data_list[0]
                if 'run_obj' in single_run_data_item and single_run_data_item['run_obj'] is not None:
                    last_exp_id_for_path = single_run_data_item['run_obj'].info.experiment_id
                else:
                    fallback_run_obj = client.get_run(single_run_id)
                    last_exp_id_for_path = fallback_run_obj.info.experiment_id
            except Exception as e:
                print(f"COMPARE_TRAINING: Warning: Could not determine experiment ID for run {single_run_id} for save_dir: {e}. Defaulting.")
        
        if last_exp_id_for_path:
            save_dir = f"{storage_path}/mlruns/{last_exp_id_for_path}/{single_run_id}/artifacts/plots"
        else:
            save_dir = f"{storage_path}/plots"
        save_path = f"{save_dir}/training_{timestamp}.png"
    else:
        save_dir = f"{storage_path}/plots"
        save_path = f"{save_dir}/training_comparison_{timestamp}.png"

    os.makedirs(save_dir, exist_ok=True)
    save_figure(save_path, fig=fig, dpi=dpi)
    plt.close(fig)

def compare_contours(
        run_ids, 
        param1, 
        param2, 
        guide_samples=10000,
        seed=1,
        device="cuda:0",
        global_rank=0,
        steps='best', 
        cosmo_exp='num_tracers',
        level=0.68, 
        show_grid=False
        ):
    samples = []
    run_ids = [run_ids] if type(run_ids) != list else run_ids
    steps = [steps] if type(steps) != list else steps
    for run_id in run_ids:
        for step in steps:
            samples.append(get_nominal_samples(
                run_id, 
                guide_samples=guide_samples,
                seed=seed,
                device=device,
                step=step, 
                cosmo_exp=cosmo_exp, 
                global_rank=global_rank
                )[0])
    
    areas_shoelace = []
    areas_grid = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # First pass: get parameter ranges
    all_x = []
    all_y = []
    for sample in samples:
        density = sample.get2DDensity(param1, param2)
        contour_level = density.getContourLevels([level])[0]
        cs = ax.contour(density.x, density.y, density.P, 
                       levels=[contour_level])
        paths = cs.collections[0].get_paths()
        for path in paths:
            vertices = path.vertices
            all_x.extend(vertices[:, 0])
            all_y.extend(vertices[:, 1])
    ax.clear()  # Clear the first pass
    
    # Calculate parameter ranges with some padding
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create a fine grid for area verification in original space
    grid_size = 1000
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    cell_area = ((x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0]))
    
    # Create custom legend
    legend_elements = []
    
    # Calculate areas and plot
    for i, sample in enumerate(samples):
        density = sample.get2DDensity(param1, param2)
        contour_level = density.getContourLevels([level])[0]
        
        # Get contour paths
        cs = ax.contour(density.x, density.y, density.P, 
                       levels=[contour_level], colors=colors[i])
        paths = cs.collections[0].get_paths()
        
        for path in paths:
            vertices = path.vertices
            x, y = vertices[:, 0], vertices[:, 1]
            
            # Calculate areas using Shoelace formula in original space
            area_shoelace = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            # Calculate area using grid points in original space
            path_orig = Path(np.column_stack((x, y)))
            inside_points = path_orig.contains_points(grid_points)
            grid_area = np.sum(inside_points) * cell_area
            
            # Store areas
            areas_shoelace.append(area_shoelace)
            areas_grid.append(grid_area)
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=colors[i], 
                                       label=f'Contour {i}, Step {steps[i]}, Area: {area_shoelace:.3f}'))
            
            # Plot grid points if requested
            if show_grid:
                inside_points_reshaped = inside_points.reshape(grid_size, grid_size)
                # Create a custom colormap using the current color cycle
                custom_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                    f"custom_cmap_{i}", [(1, 1, 1, 0), colors[i]], N=2)
                ax.imshow(inside_points_reshaped, extent=[x_min, x_max, y_min, y_max], 
                          alpha=0.3, cmap=custom_cmap, origin='lower', aspect='auto')
    
    # Set labels and title
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
def loss_area_plot(mlflow_exp, var_name, step_interval=1000, excluded_runs=[], cosmo_exp='num_tracers'):
    # Set MLflow tracking URI before creating client
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(mlflow_exp).experiment_id
    run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]
    all_run_losses = []
    all_run_areas = []
    vars_values = [] # Renamed from vars to avoid conflict with built-in
    # Sort runs by the specified parameter value in descending order
    run_ids = sorted(run_ids, key=lambda x: -float(client.get_run(x).data.params[var_name]))
    print(f"Processing runs: {run_ids}")
    if step_interval % 100 != 0:
        raise ValueError("step_interval must be a multiple of 100")
    for run_id in run_ids:
        print(f"Fetching data for run: {run_id}")
        var_value = client.get_run(run_id).data.params[var_name]
        vars_values.append(var_value)

        # Fetch metric histories
        loss_history = client.get_metric_history(run_id, 'loss')
        nominal_area_history = client.get_metric_history(run_id, 'nominal_area')

        # Create a dictionary for quick loss lookup by step
        loss_dict = {metric.step: metric.value for metric in loss_history if np.isfinite(metric.value)}

        sampled_losses = []
        sampled_areas = []

        # Iterate through nominal area history and pair with loss at the same step
        for area_metric in nominal_area_history:
            step = area_metric.step
            area_value = area_metric.value

            # Check if step is multiple of 100 and area is valid
            if step % step_interval == 0 and np.isfinite(area_value):
                # Check if a loss exists for this exact step
                if step in loss_dict:
                    sampled_losses.append(loss_dict[step])
                    sampled_areas.append(area_value)
                else:
                    print(f"Warning: Loss not found for step {step} in run {run_id}. Skipping point.")

        if not sampled_losses:
             print(f"Warning: No valid paired loss/area points found for run {run_id} at {step_interval}-step intervals.")
             # Add empty lists to maintain alignment, or handle as needed
             all_run_losses.append([])
             all_run_areas.append([])
             continue

        all_run_losses.append(sampled_losses)
        all_run_areas.append(sampled_areas)

    plt.figure(figsize=(14, 8)) # Add figure creation
    ax = plt.gca() # Get current axes

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Plot the results
    for i in range(len(run_ids)):
        if not all_run_losses[i]: # Skip if no data for this run
             continue

        losses = np.array(all_run_losses[i])
        areas = np.array(all_run_areas[i])

        # Find indices of minimum loss and minimum area
        min_loss_idx = np.argmin(losses)
        min_area_idx = np.argmin(areas)

        # Normalize loss for plotting
        losses_normalized = losses - np.min(losses) * 1.005

        # Get color for this run
        color = colors[i % len(colors)]

        # Plot all points for this run
        ax.scatter(losses_normalized, areas, label=f'{var_name}={vars_values[i]}', alpha=0.6, color=color, marker='o', s=30)

        # Highlight the minimum loss point
        ax.scatter(losses_normalized[min_loss_idx], areas[min_loss_idx],
                   marker='*', color=color, s=80, alpha=0.9, edgecolors='black', zorder=3) # No label here

        # Highlight the minimum area point
        ax.scatter(losses_normalized[min_area_idx], areas[min_area_idx],
                   marker='P', color=color, s=80, alpha=0.9, edgecolors='black', zorder=3) # No label here, 'P' is plus sign

    # Add DESI line
    ax.axhline(y=5.214, color='black', linestyle='--', label='DESI 68% Contour')

    # Configure plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Normalized Loss (Loss - Min Loss)")
    ax.set_ylabel("Nominal Area")
    ax.set_title(f"Loss vs. Nominal Area (Sampled every {step_interval} steps)")

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()

    # Add custom handles for the markers
    min_loss_legend = Line2D([0], [0], marker='*', color='w', label='Best Loss',
                             markerfacecolor='grey', markersize=12, markeredgecolor='black')
    min_area_legend = Line2D([0], [0], marker='P', color='w', label='Best Area',
                             markerfacecolor='grey', markersize=10, markeredgecolor='black') # 'P' marker

    handles.extend([min_loss_legend, min_area_legend])

    ax.legend(handles=handles)
    ax.grid(True, which="both", ls="--", alpha=0.5) # Add grid for log scale

    # Save the plot
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    os.makedirs(f"{storage_path}/mlruns/{exp_id}/plots", exist_ok=True)
    save_path = f"{storage_path}/mlruns/{exp_id}/plots/loss_area_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_figure(save_path, fig=plt.gcf())

def plot_lr_schedule(initial_lr, gamma, gamma_freq, steps=100000):
    steps = np.arange(0, steps, 1)
    lr = initial_lr * gamma ** (steps / gamma_freq)
    legend_label = f'initial_lr={initial_lr}, gamma={gamma}, gamma_freq={gamma_freq}'
    plt.plot(steps, lr, label=legend_label)

    return lr[-1]

def plot_designs_parallel_coords(run_id, cosmo_exp='num_tracers', alpha=0.6, linewidth=0.8, 
                                 figsize=(12, 6), cmap='viridis', save_path=None,
                                 color_dim=0, labels=None):
    """
    Plot design space using parallel coordinates.
    
    Each line represents one design point, connecting the 4 tracer fractions.
    Lines are colored by the specified dimension index.
    
    Args:
        run_id (str): MLflow run ID
        cosmo_exp (str): Cosmology experiment name (default: 'num_tracers')
        alpha (float): Transparency of lines (default: 0.6)
        linewidth (float): Width of lines (default: 0.8)
        figsize (tuple): Figure size (default: (12, 6))
        cmap (str): Colormap for line colors (default: 'viridis')
        save_path (str, optional): Path to save the figure. If None, displays the figure.
        color_dim (int): Dimension index (0-3) to use for coloring lines. Default: 0
        labels (tuple/list, optional): Custom labels for each dimension. If None, uses generic
                                      labels like f_0, f_1, etc. Example: ('f_BGS', 'f_LRG', 'f_ELG', 'f_QSO')
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get run data to find exp_id
    run_data_list, _, _ = get_runs_data(run_ids=run_id, cosmo_exp=cosmo_exp)
    if not run_data_list:
        raise ValueError(f"Run {run_id} not found in experiment {cosmo_exp}")
    
    run_data = run_data_list[0]
    exp_id = run_data['exp_id']
    run_obj = run_data['run_obj']
    run_args = run_data['params']
    device = "cuda:0"
    experiment = init_experiment(
        run_obj, run_args, device, 
        design_args={}, global_rank=0
    )

    num_targets = experiment.desi_tracers.groupby('class').sum()['targets'].reindex(experiment.design_labels)
    nominal_total_obs = int(experiment.desi_data.drop_duplicates(subset=['tracer'])['observed'].sum())
    # Get maximum possible tracers as fractions
    max_tracers = np.array([num_targets[target] for target in experiment.design_labels])
    max_tracers = max_tracers / nominal_total_obs

    # Build path to designs.npy
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    designs_path = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/designs.npy"
    
    # Load designs
    if not os.path.exists(designs_path):
        raise FileNotFoundError(f"Designs file not found: {designs_path}")
    
    designs = np.load(designs_path)
    
    # Ensure designs is a numpy array
    if not isinstance(designs, np.ndarray):
        designs = np.array(designs)
    
    # Validate color_dim
    if not isinstance(color_dim, int) or color_dim not in range(designs.shape[1]):
        raise ValueError(f"color_dim must be an integer 0-{designs.shape[1]-1}, got {color_dim}")
    
    # Extract dimensions
    n_dims = designs.shape[1]
    dim_data = {i: designs[:, i] for i in range(n_dims)}
    
    # Get color dimension data
    color_values = dim_data[color_dim]
    
    # Validate and set labels
    if labels is not None:
        if not isinstance(labels, (tuple, list)) or len(labels) != n_dims:
            raise ValueError(f"labels must be a tuple/list of length {n_dims}, got {labels}")
        # Use provided labels as-is (user can format with $ for LaTeX if desired)
        axis_labels = list(labels)
    else:
        # Create generic labels
        axis_labels = [f'$f_{i}$' for i in range(n_dims)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Parallel axes positions
    x_positions = np.arange(n_dims)
    
    # Color by specified dimension values
    color_min = color_values.min()
    color_max = color_values.max()
    
    # Get colormap
    colormap = plt.get_cmap(cmap)
    # Normalize color values to [0, 1] for colormap
    color_normalized = (color_values - color_min) / (color_max - color_min + 1e-10)
    colors = colormap(color_normalized)
    
    # Plot lines for each design
    for i in range(len(designs)):
        values = [dim_data[j][i] for j in range(n_dims)]
        ax.plot(x_positions, values, color=colors[i], alpha=alpha, linewidth=linewidth)
    
    ax.plot(x_positions, experiment.nominal_design.cpu().numpy(), color='black', alpha=1.0, linewidth=2, label='Nominal Design', zorder=3)

    # Plot max_tracers as upward-pointing triangles to represent maximums
    ax.scatter(x_positions, max_tracers, color='gray', s=50, zorder=10, marker='^', label='Max Tracers')
    
    # Try to load and plot the most recent eval JSON optimal design
    artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
    try:
        json_path, eig_data = load_eig_data_file(artifacts_dir)
        print(f"Loaded eval JSON: {json_path}")
        if json_path is not None and eig_data is not None:
            # Try to get optimal_design from top level first
            optimal_design = eig_data.get('optimal_design')
            
            # If not at top level, try nested in step data
            if optimal_design is None:
                # Find the highest step
                step_keys = [k for k in eig_data.keys() if k.startswith('step_')]
                if step_keys:
                    step_numbers = []
                    for k in step_keys:
                        try:
                            step_num = int(k.split('_')[1])
                            step_numbers.append((step_num, k))
                        except (ValueError, IndexError):
                            continue
                    if step_numbers:
                        step_str = max(step_numbers, key=lambda x: x[0])[1]
                        step_data = eig_data.get(step_str, {})
                        variable_data = step_data.get('variable', {})
                        optimal_design = variable_data.get('optimal_design')
            
            # Plot if we found an optimal design
            if optimal_design is not None:
                optimal_design_array = np.array(optimal_design)
                if len(optimal_design_array) == n_dims:
                    ax.plot(x_positions, optimal_design_array, color='tab:orange', 
                           linewidth=2, label='Optimal Design', zorder=4)
    except Exception as e:
        # Silently fail if we can't load the eval JSON
        pass
    
    # Set axis properties
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels, fontsize=12)
    ax.set_ylabel('Tracer Fraction', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    
    # Set y-axis limits to show full range
    all_values = np.concatenate([dim_data[j] for j in range(n_dims)])
    y_min = all_values.min() * 0.5
    y_max = all_values.max() * 1.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, dpi=300)
    
    # Display figure in interactive environment
    if _is_interactive_environment():
        _display_figure(fig)
    plt.close(fig)

def plot_design_pts(
    designs_file=None,
    run_id=None,
    cosmo_exp='num_tracers',
    figsize=(10, 8),
    marker_size=60,
    alpha=0.9,
    cmap='viridis',
    labels=None,
    save_path=None,
    title=None
):
    """
    Plot designs loaded from a Python file (`designs.py`) in up to 4 dimensions.

    Depending on the number of design dimensions, the function renders:
        - 1D: Line plot with black markers.
        - 2D: Scatter plot (grid) with black markers.
        - 3D: 3D scatter plot with black markers.
        - 4D: 3D scatter plot where the 4th dimension is encoded by color.

    Args:
        designs_file (str, optional): Path to a .npy file containing the designs array.
                                      If not provided, run_id must be supplied.
        run_id (str, optional): MLflow run ID whose artifacts include `designs.npy`.
        cosmo_exp (str): Cosmology experiment name when resolving run artifacts.
        figsize (tuple): Figure size.
        marker_size (int/float): Size of the plotted markers.
        alpha (float): Alpha for markers (used in 3D/4D plots).
        cmap (str): Colormap to use when encoding the 4th dimension.
        labels (list/tuple): Optional axis labels for each dimension.
        save_path (str, optional): Optional path to save the resulting figure.
        title (str, optional): Optional figure title. Defaults to the filename.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    if designs_file is None:
        if run_id is None:
            raise ValueError("Provide either designs_file or run_id.")
        run_data_list, _, _ = get_runs_data(run_ids=run_id, parse_params=False, cosmo_exp=cosmo_exp)
        if not run_data_list:
            raise ValueError(f"Run {run_id} not found in experiment {cosmo_exp}")
        exp_id = run_data_list[0]['exp_id']
        storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
        designs_file = os.path.join(storage_path, "mlruns", exp_id, run_id, "artifacts", "designs.npy")
    resolved_path = designs_file if os.path.isabs(designs_file) else os.path.abspath(designs_file)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Designs file not found: {resolved_path}")
    designs = np.load(resolved_path)
    if not isinstance(designs, np.ndarray):
        designs = np.asarray(designs)
    if designs.ndim == 1:
        designs = designs.reshape(-1, 1)
    elif designs.ndim != 2:
        raise ValueError("Designs must be 1D or 2D (n_designs, n_dims).")
    if designs.shape[0] == 0:
        raise ValueError("No designs found in the supplied designs file.")
    n_designs, n_dims = designs.shape
    if labels is None:
        axes_labels = [f'Dim {i+1}' for i in range(n_dims)]
    else:
        if not isinstance(labels, (list, tuple)):
            raise ValueError("labels must be a list or tuple")
        if len(labels) != n_dims:
            raise ValueError(f"labels must contain {n_dims} entries, received {len(labels)}.")
        axes_labels = list(labels)
    if n_dims < 1 or n_dims > 4:
        raise ValueError(f"plot_designs only supports 1D to 4D designs. Found {n_dims} dimensions.")
    
    figure_title = title or f"Design Space"
    
    if n_dims == 1:
        fig, ax = plt.subplots(figsize=figsize)
        x = designs[:, 0]
        y = np.zeros_like(x)
        ax.axhline(0, color='black', linewidth=1.0, alpha=0.5)
        ax.scatter(x, y, color='black', s=marker_size)
        ax.set_xlabel(axes_labels[0])
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_title(figure_title, fontsize=14, weight='bold')
        ax.grid(False)
    elif n_dims == 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(designs[:, 0], designs[:, 1], color='black', s=marker_size)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(figure_title, fontsize=14, weight='bold')
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if n_dims == 3:
            ax.scatter(
                designs[:, 0],
                designs[:, 1],
                designs[:, 2],
                color='black',
                s=marker_size,
                alpha=alpha
            )
        else:  # 4D
            color_values = designs[:, 3]
            scatter = ax.scatter(
                designs[:, 0],
                designs[:, 1],
                designs[:, 2],
                c=color_values,
                              cmap=cmap,
                s=marker_size,
                alpha=alpha
            )
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20, pad=0.1)
            cbar.set_label(axes_labels[3], fontsize=12)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_zlabel(axes_labels[2])
        ax.set_title(figure_title, fontsize=14, weight='bold', pad=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=300, display_fig=False, close_fig=False)
    
    return fig


def plot_2d_eig(
    run_id,
    step_key=None,
    cosmo_exp='num_tracers',
    save_path=None,
    figsize=(10, 8),
    dpi=400,
    cmap='viridis',
    show_optimal=True,
    show_nominal=False,
    title=None
):
    """
    Plot 2D design space with EIG values as a colormap.
    
    Args:
        run_id (str): MLflow run ID
        step_key (str or int, optional): Step key to filter eig_data files (e.g., 50000 or 'last').
                                        If None, uses most recent file.
        cosmo_exp (str): Cosmology experiment name. Default 'num_tracers'.
        save_path (str, optional): Path to save the figure. If None, doesn't save.
        figsize (tuple): Figure size. Default (10, 8).
        dpi (int): DPI for saved figure. Default 400.
        cmap (str): Colormap name. Default 'viridis'.
        show_optimal (bool): Whether to mark the optimal design. Default True.
        show_nominal (bool): Whether to mark the nominal design. Default False.
        title (str, optional): Custom title for the plot. If None, auto-generates.
    
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Set storage path
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Get run data to find exp_id
    run_data_list, experiment_id, _ = get_runs_data(
        run_ids=run_id,
        parse_params=False,
        cosmo_exp=cosmo_exp
    )
    
    if not run_data_list:
        raise ValueError(f"Run {run_id} not found in experiment {cosmo_exp}")
    
    run_data = run_data_list[0]
    exp_id = run_data['exp_id']
    
    # Find artifacts directory
    artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
    
    # Load completed eig_data file
    json_path, eig_data = load_eig_data_file(artifacts_dir, step_key=None)
    if eig_data is None:
        raise ValueError(f"No completed eig_data file found for run {run_id}")
    
    # Determine which step to use
    if step_key is not None:
        step_str = f"step_{step_key}"
    else:
        # Find the highest available step key
        step_keys = [k for k in eig_data.keys() if k.startswith('step_')]
        if len(step_keys) == 0:
            raise ValueError("No step data found in eig_data")
        # Extract step numbers and find the highest
        step_numbers = []
        for k in step_keys:
            try:
                step_num = int(k.split('_')[1])
                step_numbers.append((step_num, k))
            except (ValueError, IndexError):
                continue
        if step_numbers:
            step_str = max(step_numbers, key=lambda x: x[0])[1]
        else:
            step_str = step_keys[0]  # Fallback to first if parsing fails
    
    if step_str not in eig_data:
        raise ValueError(f"Step {step_str} not found in eig_data")
    
    step_data = eig_data[step_str]
    
    # Access nested structure
    variable_data = step_data.get('variable', {})
    
    # Extract designs from top-level input_designs and EIG values from variable subdict
    if 'input_designs' not in eig_data:
        raise ValueError(f"No input_designs found in eig_data")
    designs = np.array(eig_data['input_designs'])
    
    # Get EIG values
    if 'eigs_avg' not in variable_data:
        raise ValueError(f"No eigs_avg found in {step_str}")
    eigs = np.array(variable_data['eigs_avg'])
    
    # Validate that designs are 2D
    if designs.shape[1] != 2:
        raise ValueError(f"Expected 2D designs, but got {designs.shape[1]}D designs")
    
    # Extract design labels (try metadata first, then default)
    design_labels = ['z_1', 'z_2']  # Default
    if 'metadata' in eig_data and isinstance(eig_data.get('metadata'), dict):
        if 'design_labels' in eig_data['metadata']:
            design_labels = eig_data['metadata']['design_labels']
    if len(design_labels) < 2:
        design_labels = ['z_1', 'z_2']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create regular grid for interpolation
    x_min, x_max = designs[:, 0].min(), designs[:, 0].max()
    y_min, y_max = designs[:, 1].min(), designs[:, 1].max()
    
    # Create grid (use reasonable resolution, e.g., 200x200)
    grid_resolution = 200
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate EIG values onto the grid
    # Try cubic first, fallback to linear if not enough points
    try:
        zi = griddata(
            (designs[:, 0], designs[:, 1]),
            eigs,
            (xi_grid, yi_grid),
            method='cubic',
            fill_value=np.nan
        )
    except ValueError:
        # Fallback to linear interpolation if cubic fails (not enough points)
        zi = griddata(
            (designs[:, 0], designs[:, 1]),
            eigs,
            (xi_grid, yi_grid),
            method='linear',
            fill_value=np.nan
        )
    
    # Use imshow for continuous colormap
    # Note: imshow expects (y, x) ordering and origin='lower' for proper orientation
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(
        zi,
        extent=extent,
        origin='lower',
        cmap=cmap,
        aspect='auto',
        interpolation='bilinear'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expected Information Gain [bits]', fontsize=12, weight='bold')
    
    # Mark optimal design if available and requested
    if show_optimal:
        if 'optimal_design' in variable_data:
            optimal_design = np.array(variable_data['optimal_design'])
        else:
            optimal_design = None
        
        if optimal_design is not None and len(optimal_design) == 2:
            ax.scatter(
                optimal_design[0],
                optimal_design[1],
                marker='o',
                s=100,
                color='tab:orange',
                edgecolors='none',
                label='Optimal Design',
                zorder=5
            )
    
    # Mark nominal design if available and requested
    # Note: nominal_design is not stored in step_data, would need to get from experiment
    # For now, skip this if not available
    if show_nominal:
        # Try to get from experiment if needed, or skip
        pass
    
    # Set labels
    ax.set_xlabel(f'${design_labels[0]}$', fontsize=14)
    ax.set_ylabel(f'${design_labels[1]}$', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    # Add legend on the plot if we have markers
    if (show_optimal and 'optimal_design' in variable_data):
        ax.legend(loc='upper right', fontsize=10, frameon=True)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(save_path, fig=fig, dpi=dpi, close_fig=False)
    
    return fig


def save_figure(save_path, fig=None, close_fig=True, display_fig=True, dpi=300):
    """
    Save and optionally display a matplotlib figure.
    
    Args:
        save_path (str): Path where to save the figure
        fig (matplotlib.figure.Figure, optional): Figure to save. If None, uses current figure
        close_fig (bool): Whether to close the figure after saving
        display_fig (bool): Whether to display the figure (in non-TTY environments)
    """
    # Determine the figure to work with
    target_fig = fig if fig is not None else plt.gcf()
    
    # Save the figure
    _save_figure(target_fig, save_path, dpi=dpi)
    
    # Display figure if requested and in interactive environment
    if display_fig and _is_interactive_environment():
        _display_figure(target_fig)
    
    # Close figure if requested
    if close_fig:
        plt.close(target_fig)

def _is_interactive_environment():
    """Check if we're in an interactive environment (not TTY)."""
    try:
        return not os.isatty(sys.stdout.fileno())
    except (io.UnsupportedOperation, AttributeError):
        return True

def _save_figure(fig, save_path, dpi=300):
    """Save a figure to the specified path."""
    fig.savefig(save_path, dpi=dpi)
    print(f"Saved plot to {save_path}")

def _display_figure(fig):
    """Display a figure in an interactive environment."""
    try:
        display(fig)
    except Exception:
        plt.show()

if __name__ == "__main__":

    eval_args = {"n_samples": 30000, "device": "cuda:1", "eval_seed": 1}
    plot_training(
        mlflow_exp='base_NAF_gamma_fixed', 
        var='pyro_seed', 
        log_scale=True,
        show_checkpoints=False,
        cosmo_exp='num_tracers'
    )