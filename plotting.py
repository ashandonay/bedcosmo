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
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import matplotlib.colors
import warnings
from datetime import datetime
from pyro_oed_src import posterior_loss
import json
from IPython.display import display

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

    # Convert colors to proper format
    def convert_color(c):
        if isinstance(c, np.ndarray):
            # Convert RGBA array to hex color string
            return matplotlib.colors.to_hex(c)
        elif isinstance(c, str):
            return c
        else:
            return str(c)

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
                    nominal_samples, target_labels, latex_labels = load_nominal_samples(run_params['cosmo_exp'], run_params['cosmo_model'])
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
    nominal_samples, target_labels, latex_labels = load_nominal_samples(cosmo_exp, cosmo_model_for_desi)
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
    if var:
        title_vars_str = ', '.join(vars_list)
        title = f'Posterior comparison (levels: {levels}) grouped by {title_vars_str} ({len(run_data_list)} total runs), Step: {step}'
    else:
        plot_title_exp_part = actual_mlflow_exp_for_title if actual_mlflow_exp_for_title else "Selected Runs"
        title = f'Posterior comparison (levels: {levels}) for {plot_title_exp_part}, Step: {step}'
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

def compare_eigs(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        labels=None,
        step_key=None,
        use_sorted=True,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        colors=None,
        show_optimal=True,
        show_nominal=True,
        title=None,
        x_label=None
    ):
    """
    Compare EIG values across multiple runs by loading JSON files.
    Can take either an experiment name or a list of run IDs.
    
    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_ids (list, optional): List of MLflow run IDs to compare. If provided, mlflow_exp is ignored.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Cosmological experiment name (default: 'num_tracers')
        labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
        step_key (str or int, optional): Which step to use. If None, finds most recent eig_data file
        use_sorted (bool): If True, plot sorted EIGs. If False, plot unsorted EIGs
        save_path (str, optional): Path to save the comparison plot. If None, doesn't save
        figsize (tuple): Figure size (width, height)
        dpi (int): Resolution for saved figure
        colors (list, optional): Colors for each run. If None, uses default color cycle
        show_optimal (bool): If True, mark optimal EIG for each run
        show_nominal (bool): If True, mark nominal EIG for each run
        title (str, optional): Custom title for the plot. If None, generates default title
        x_label (str, optional): Custom x-axis label. If None, uses design label for 1D designs or "Design Index" for multi-dimensional
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import glob
    
    # Set MLflow tracking URI
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Get run data - can use either mlflow_exp or run_ids
    run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = get_runs_data(
        mlflow_exp=mlflow_exp,
        run_ids=run_ids,
        excluded_runs=excluded_runs,
        parse_params=False,
        cosmo_exp=cosmo_exp
    )
    if not run_data_list:
        raise ValueError(f"No runs found in experiment {cosmo_exp}")
    
    # Extract run_ids from run_data_list
    run_ids = [run_data['run_id'] for run_data in run_data_list]
    
    # Build a map from run_id to exp_id
    run_id_to_exp_id = {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    # Find and load JSON files for each run
    all_data = []
    found_run_ids = []
    json_paths = []
    
    for run_id in run_ids:
        if run_id not in run_id_to_exp_id:
            print(f"Warning: Run {run_id} not found in experiment {cosmo_exp}, skipping...")
            continue
        
        exp_id = run_id_to_exp_id[run_id]
        artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
        
        if not os.path.exists(artifacts_dir):
            print(f"Warning: Artifacts directory not found for run {run_id}, skipping...")
            continue
        
        # Find eig_data JSON files
        eig_files = glob.glob(f"{artifacts_dir}/eig_data_*.json")
        
        if len(eig_files) == 0:
            print(f"Warning: No eig_data JSON files found for run {run_id}, skipping...")
            continue
        
        # Filter by step_key if specified
        if step_key is not None:
            step_str = str(step_key)
            filtered_files = [f for f in eig_files if f"eig_data_{step_str}_" in os.path.basename(f)]
            if len(filtered_files) == 0:
                print(f"Warning: No eig_data files for step {step_key} found for run {run_id}, skipping...")
                continue
            eig_files = filtered_files
        
        # Use most recent file (sort by timestamp in filename)
        eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        json_path = eig_files[0]
        
        # Load JSON file
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                all_data.append(data)
                found_run_ids.append(run_id)
                json_paths.append(json_path)
        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No valid EIG data files found to compare")
    
    # Generate labels if not provided
    if labels is None:
        labels = [run_id[:8] for run_id in found_run_ids]
    
    if len(labels) != len(all_data):
        labels = labels[:len(all_data)]
    
    # Determine which step to use for each run and filter valid data
    valid_indices = []
    step_numbers = []
    for i, data in enumerate(all_data):
        # If step_key was provided, we already filtered files by that step, so use it
        if step_key is not None:
            step_num = int(step_key) if isinstance(step_key, str) else step_key
        else:
            # Use sort_step from metadata
            step_num = data.get('metadata', {}).get('sort_step', None)
            
            if step_num is None:
                # Try to find any available step
                step_keys = [k for k in data.keys() if k.startswith('step_')]
                if step_keys:
                    # Extract step number from first available step key
                    step_num = int(step_keys[0].split('_')[1])
                else:
                    print(f"Warning: Could not determine step for run {labels[i] if i < len(labels) else i}, skipping...")
                    continue
        
        valid_indices.append(i)
        step_numbers.append(step_num)
    
    # Filter data, labels, and colors to only include valid runs
    all_data = [all_data[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    colors = [colors[i] for i in valid_indices] if colors is not None else None
    
    # Generate colors if not provided (after filtering)
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(len(all_data))]
    
    # Check if designs are 1D (use first valid data to determine)
    is_1d_design = False
    design_label = None
    if len(all_data) > 0:
        first_data = all_data[0]
        if 'designs' in first_data:
            designs = np.array(first_data['designs']['sorted' if use_sorted else 'unsorted'])
            # Check if designs are 1D (shape should be (n_designs, 1) or (n_designs,))
            if len(designs.shape) == 1 or (len(designs.shape) == 2 and designs.shape[1] == 1):
                is_1d_design = True
                # Get design label from metadata if available
                if 'metadata' in first_data and isinstance(first_data.get('metadata'), dict) and 'design_labels' in first_data['metadata']:
                    design_labels = first_data['metadata']['design_labels']
                    if len(design_labels) > 0:
                        design_label = design_labels[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot EIGs for each run
    for i, (data, label, color) in enumerate(zip(all_data, labels, colors)):
        step_num = step_numbers[i]
        step_key_name = f'step_{step_num}'
        
        if step_key_name not in data:
            print(f"Warning: Step {step_num} not found in {labels[i]}, skipping...")
            continue
        
        step_data = data[step_key_name]
        
        # Get EIG values (sorted or unsorted)
        if use_sorted:
            eigs = np.array(step_data.get('eigs_sorted', []))
            eigs_std = np.array(step_data.get('eigs_std_sorted', []))
            sorted_indices = data.get('sorted_indices', [])
        else:
            eigs = np.array(step_data.get('eigs_unsorted', []))
            eigs_std = np.array(step_data.get('eigs_std_unsorted', []))
        
        # Determine x-axis values
        if is_1d_design and 'designs' in data:
            designs = np.array(data['designs']['sorted' if use_sorted else 'unsorted'])
            # Extract 1D design values
            if len(designs.shape) == 1:
                x_vals = designs
            else:
                x_vals = designs[:, 0]  # First (and only) column
        else:
            x_vals = np.arange(len(eigs))
        
        if len(eigs) == 0:
            print(f"Warning: No EIG data found for {label}, skipping...")
            continue
        
        # Plot EIG line
        ax.plot(x_vals, eigs, label=label, color=color, linewidth=2, alpha=0.8)
        
        # Plot error bars (filled region)
        if len(eigs_std) > 0 and np.any(eigs_std > 0):
            ax.fill_between(
                x_vals,
                eigs - eigs_std,
                eigs + eigs_std,
                color=color,
                alpha=0.2
            )
        
        # Mark optimal EIG if available and requested (new flat structure)
        if show_optimal and 'optimal_eig' in data and isinstance(data['optimal_eig'], (int, float)):
            optimal_eig = data['optimal_eig']
            optimal_design = data.get('optimal_design', [])
            
            if optimal_eig is not None and optimal_design and 'designs' in data:
                designs = np.array(data['designs']['sorted' if use_sorted else 'unsorted'])
                # Find matching design index
                optimal_idx = None
                for idx, design in enumerate(designs):
                    if np.allclose(design, optimal_design, rtol=1e-5):
                        optimal_idx = idx
                        break
                if optimal_idx is not None:
                    # Determine x-value for optimal point
                    if is_1d_design and 'designs' in data:
                        designs = np.array(data['designs']['sorted' if use_sorted else 'unsorted'])
                        if len(designs.shape) == 1:
                            optimal_x = designs[optimal_idx]
                        else:
                            optimal_x = designs[optimal_idx, 0]
                    else:
                        optimal_x = optimal_idx
                    
                    # Draw vertical line at optimal design
                    ax.axvline(x=optimal_x, color=color, linestyle=':', 
                              linewidth=1.5, alpha=0.6, zorder=5)
                    # Add dot at the optimal point (intersection of vertical line and optimal EIG value)
                    ax.scatter([optimal_x], [optimal_eig], 
                              color=color, marker='o', s=100, 
                              edgecolor='black', linewidth=1.5, zorder=10)
        
        # Mark nominal EIG if available and requested
        # Check both top level (new flat structure) and step_data (old structure)
        nominal_eig = None
        if show_nominal:
            if 'nominal_eig' in data and isinstance(data['nominal_eig'], (int, float)):
                nominal_eig = data['nominal_eig']
            elif 'nominal_eig' in step_data:
                nominal_eig = step_data['nominal_eig']
        
        if nominal_eig is not None:
            ax.axhline(y=nominal_eig, color=color, linestyle='--', 
                      linewidth=1.5, alpha=0.6)
    
    # Format plot
    # Determine x-axis label
    if x_label is not None:
        x_axis_label = x_label
    elif is_1d_design and design_label is not None:
        x_axis_label = design_label
    else:
        x_axis_label = 'Design Index' + (' (Sorted)' if use_sorted else '')
    
    ax.set_xlabel(x_axis_label, fontsize=14, weight='bold')
    ax.set_ylabel('Expected Information Gain [bits]', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=14, framealpha=0.9)
    
    # Set title
    if title is None:
        if mlflow_exp and actual_mlflow_exp_for_title:
            title = f'EIG per Design - {actual_mlflow_exp_for_title}'
        elif mlflow_exp:
            title = f'EIG per Design - {mlflow_exp}'
        else:
            title = 'EIG per Design'
    ax.set_title(title, fontsize=16, weight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        save_figure(save_path, fig=fig, dpi=dpi)
        print(f"Saved comparison plot to {save_path}")
    
    return fig, ax

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
    import glob
    
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
                    # Find eig_data JSON files
                    ref_eig_files = glob.glob(f"{ref_artifacts_dir}/eig_data_*.json")
                    
                    if step_key is not None:
                        step_str = str(step_key)
                        filtered_files = [f for f in ref_eig_files if f"eig_data_{step_str}_" in os.path.basename(f)]
                        if len(filtered_files) > 0:
                            ref_eig_files = filtered_files
                    
                    if len(ref_eig_files) > 0:
                        ref_eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
                        ref_json_path = ref_eig_files[0]
                        
                        with open(ref_json_path, 'r') as f:
                            ref_data = json.load(f)
                        
                        # Extract nominal_eig, optimal_eig, and optimal_design
                        if 'nominal_eig' in ref_data and isinstance(ref_data['nominal_eig'], (int, float)):
                            ref_nominal_eig = ref_data['nominal_eig']
                        elif 'step_data' in ref_data and isinstance(ref_data['step_data'], dict) and 'nominal_eig' in ref_data['step_data']:
                            ref_nominal_eig = ref_data['step_data']['nominal_eig']
                        
                        if 'nominal_eig_std' in ref_data and isinstance(ref_data['nominal_eig_std'], (int, float)):
                            ref_nominal_eig_std = ref_data['nominal_eig_std']
                        elif 'step_data' in ref_data and isinstance(ref_data['step_data'], dict) and 'nominal_eig_std' in ref_data['step_data']:
                            ref_nominal_eig_std = ref_data['step_data']['nominal_eig_std']
                        
                        if 'optimal_eig' in ref_data:
                            ref_optimal_eig = ref_data['optimal_eig']
                        
                        if 'optimal_eig_std' in ref_data and isinstance(ref_data['optimal_eig_std'], (int, float)):
                            ref_optimal_eig_std = ref_data['optimal_eig_std']
                        elif 'step_data' in ref_data and isinstance(ref_data['step_data'], dict) and 'optimal_eig_std' in ref_data['step_data']:
                            ref_optimal_eig_std = ref_data['step_data']['optimal_eig_std']
                        
                        if 'optimal_design' in ref_data:
                            ref_optimal_design = np.array(ref_data['optimal_design'])
                        
                        # Get nominal design from experiment initialization
                        if include_nominal:
                            try:
                                device = "cuda:0"
                                experiment = init_experiment(
                                    ref_run_data['run_obj'], 
                                    ref_run_data['params'], 
                                    device, 
                                    design_args={}, 
                                    global_rank=0
                                )
                                if hasattr(experiment, 'nominal_design'):
                                    ref_nominal_design = experiment.nominal_design.cpu().numpy()
                            except Exception as e:
                                print(f"Warning: Could not initialize experiment to get nominal design from ref_id: {e}")
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
        
        if not os.path.exists(artifacts_dir):
            print(f"Warning: Artifacts directory not found for run {run_id}, skipping...")
            continue
        
        # Find eig_data JSON files
        eig_files = glob.glob(f"{artifacts_dir}/eig_data_*.json")
        
        if len(eig_files) == 0:
            print(f"Warning: No eig_data JSON files found for run {run_id}, skipping...")
            continue
        
        # Filter by step_key if specified
        if step_key is not None:
            step_str = str(step_key)
            filtered_files = [f for f in eig_files if f"eig_data_{step_str}_" in os.path.basename(f)]
            if len(filtered_files) == 0:
                print(f"Warning: No eig_data files for step {step_key} found for run {run_id}, skipping...")
                continue
            eig_files = filtered_files
        
        # Use most recent file (sort by timestamp in filename)
        eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        json_path = eig_files[0]
        
        # Load JSON file
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Extract optimal design and EIG (new flat structure)
            if 'optimal_design' in data and 'optimal_eig' in data:
                optimal_design = np.array(data['optimal_design'])
                optimal_eig_value = data.get('optimal_eig', None)
                optimal_eig_std_value = data.get('optimal_eig_std', None)
                if not isinstance(optimal_eig_std_value, (int, float)):
                    optimal_eig_std_value = None
                optimal_designs.append(optimal_design)
                optimal_eigs.append(optimal_eig_value)
                optimal_eig_stds.append(optimal_eig_std_value)
                found_run_ids.append(run_id)
                
                # Get design labels from metadata (use first run's labels as reference)
                if design_labels is None and 'metadata' in data and isinstance(data.get('metadata'), dict) and 'design_labels' in data['metadata']:
                    design_labels = data['metadata']['design_labels']
                
                # Extract nominal_eig from first run's data (for scatter plot in compare_increasing_design)
                # Only do this in compare_increasing_design - check if include_nominal is in scope
                if 'include_nominal' in locals() and include_nominal:
                    if 'nominal_eig' in data and isinstance(data['nominal_eig'], (int, float)):
                        if nominal_eig is None:
                            nominal_eig = data['nominal_eig']
                        if nominal_eig_std is None and 'nominal_eig_std' in data and isinstance(data['nominal_eig_std'], (int, float)):
                            nominal_eig_std = data['nominal_eig_std']
                    elif 'step_data' in data and isinstance(data['step_data'], dict) and 'nominal_eig' in data['step_data']:
                        if nominal_eig is None:
                            nominal_eig = data['step_data']['nominal_eig']
                        if nominal_eig_std is None and 'nominal_eig_std' in data['step_data'] and isinstance(data['step_data']['nominal_eig_std'], (int, float)):
                            nominal_eig_std = data['step_data']['nominal_eig_std']
            else:
                print(f"Warning: No optimal_design data found in {json_path}, skipping...")
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
            # Override fixed_design to False to avoid validation errors when we just need nominal_total_obs
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
                    fontsize=14,
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
        use_fractional=False,
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
        use_fractional (bool): Whether to plot fractional values or absolute quantities (default: False)
        include_nominal (bool): Whether to include nominal design bars on the left (default: False)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    import glob
    
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
        
        if not os.path.exists(artifacts_dir):
            print(f"Warning: Artifacts directory not found for run {run_id}, skipping...")
            continue
        
        # Find eig_data JSON files
        eig_files = glob.glob(f"{artifacts_dir}/eig_data_*.json")
        
        if len(eig_files) == 0:
            print(f"Warning: No eig_data JSON files found for run {run_id}, skipping...")
            continue
        
        # Filter by step_key if specified
        if step_key is not None:
            step_str = str(step_key)
            filtered_files = [f for f in eig_files if f"eig_data_{step_str}_" in os.path.basename(f)]
            if len(filtered_files) == 0:
                print(f"Warning: No eig_data files for step {step_key} found for run {run_id}, skipping...")
                continue
            eig_files = filtered_files
        
        # Use most recent file (sort by timestamp in filename)
        eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        json_path = eig_files[0]
        
        # Load JSON file
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Extract optimal design and EIG (new flat structure)
            if 'optimal_design' in data and 'optimal_eig' in data:
                optimal_design = np.array(data['optimal_design'])
                optimal_eig_value = data.get('optimal_eig', None)
                optimal_designs.append(optimal_design)
                optimal_eigs.append(optimal_eig_value)
                found_run_ids.append(run_id)
                
                # Get design labels from metadata (use first run's labels as reference)
                if design_labels is None and 'metadata' in data and isinstance(data.get('metadata'), dict) and 'design_labels' in data['metadata']:
                    design_labels = data['metadata']['design_labels']
            else:
                print(f"Warning: No optimal_design data found in {json_path}, skipping...")
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
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                 for i in range(len(optimal_designs))]
    else:
        if len(colors) < len(optimal_designs):
            colors = [colors[i % len(colors)] for i in range(len(optimal_designs))]
        else:
            colors = colors[:len(optimal_designs)]
    
    # Initialize experiment from first run to get nominal_total_obs, nominal_design, and design labels if needed
    nominal_total_obs = None
    nominal_design = None
    experiment = None
    # Always initialize if use_fractional=False (need nominal_total_obs for conversion), design_labels missing, or include_nominal=True
    if not use_fractional or design_labels is None or include_nominal:
        try:
            first_run_data = run_data_list[0]
            device = "cuda:0"
            # Override fixed_design to False to avoid validation errors when we just need nominal_total_obs
            # We don't need to initialize designs, just need the experiment object
            experiment = init_experiment(
                first_run_data['run_obj'], 
                first_run_data['params'], 
                device, 
                design_args={}, 
                global_rank=0
            )
            # Always get nominal_total_obs when use_fractional=False (needed for conversion)
            # Also get it if design_labels is missing or include_nominal=True (needed for nominal design conversion)
            if not use_fractional or design_labels is None or include_nominal:
                nominal_total_obs = experiment.nominal_total_obs
            # Get nominal_design if include_nominal is True
            if include_nominal and hasattr(experiment, 'nominal_design'):
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
        bars = ax.bar(x_pos, design_values, color=plot_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
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
        for i, (design, label, color) in enumerate(zip(plot_designs, plot_labels, plot_colors)):
            offset = (i - n_designs_to_plot/2 + 0.5) * width
            bars = ax.bar(x + offset, design, width, label=label, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            
        
        ax.set_xlabel('Design Dimension', fontsize=12, weight='bold')
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
                        nominal_samples, target_labels, latex_labels = load_nominal_samples(run_params['cosmo_exp'], run_params['cosmo_model'])
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

    # Set axis properties
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels, fontsize=12)
    ax.set_ylabel('Tracer Fraction', fontsize=12)
    ax.set_title(f'Design Space Displayed by Parallel Coordinates', fontsize=16, pad=5, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
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

def plot_designs_3d(run_id, cosmo_exp='num_tracers', figsize=(10, 8), 
                   cmap='viridis', s=60, alpha=0.8, save_path=None, 
                   show_nominal=True, dim_mapping=(1, 2, 3, 0), labels=None):
    """
    Plot design space in 3D with customizable axis and color mappings.
    
    Creates a 3D scatter plot where you can specify which dimension maps to each axis and color.
    
    Args:
        run_id (str): MLflow run ID
        cosmo_exp (str): Cosmology experiment name (default: 'num_tracers')
        figsize (tuple): Figure size (default: (10, 8))
        cmap (str): Colormap for color dimension (default: 'viridis')
        s (float): Size of scatter points (default: 60)
        alpha (float): Transparency of points (default: 0.8)
        save_path (str, optional): Path to save the figure. If None, displays the figure.
        show_nominal (bool): Whether to show nominal design marker (default: True)
        dim_mapping (tuple): Tuple of 4 indices (x_idx, y_idx, z_idx, color_idx) specifying which
                           dimension (0-3) maps to x-axis, y-axis, z-axis, and color.
                           Default: (1, 2, 3, 0) means x=dim1, y=dim2, z=dim3, color=dim0
                           Example: (0, 1, 2, 3) means x=dim0, y=dim1, z=dim2, color=dim3
                           Example: (1, 2, 0, 3) means x=dim1, y=dim2, z=dim0, color=dim3
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
    
    # Validate and extract dimension mapping
    if not isinstance(dim_mapping, (tuple, list)) or len(dim_mapping) != 4:
        raise ValueError(f"dim_mapping must be a tuple/list of 4 indices, got {dim_mapping}")
    
    x_idx, y_idx, z_idx, color_idx = dim_mapping
    
    # Validate indices are in range (will check against actual design dimensions later)
    n_dims = designs.shape[1]
    for idx, name in zip([x_idx, y_idx, z_idx, color_idx], ['x', 'y', 'z', 'color']):
        if not isinstance(idx, int) or idx not in range(n_dims):
            raise ValueError(f"{name}_idx must be an integer 0-{n_dims-1}, got {idx}")
    
    # Check that x, y, z axes are all different (color can match one of them)
    axis_indices = [x_idx, y_idx, z_idx]
    if len(set(axis_indices)) < 3:
        raise ValueError(f"X, Y, and Z axes must be different. "
                        f"Got: x_idx={x_idx}, y_idx={y_idx}, z_idx={z_idx}")
    
    # Extract dimension data
    n_dims = designs.shape[1]
    dim_data = {i: designs[:, i] for i in range(n_dims)}
    
    # Get data for each axis and color
    x_data = dim_data[x_idx]
    y_data = dim_data[y_idx]
    z_data = dim_data[z_idx]
    color_data = dim_data[color_idx]
    
    # Validate and set labels
    if labels is not None:
        if not isinstance(labels, (tuple, list)) or len(labels) != n_dims:
            raise ValueError(f"labels must be a tuple/list of length {n_dims}, got {labels}")
        # Use provided labels as-is (user can format with $ for LaTeX if desired)
        dim_labels = list(labels)
    else:
        # Create generic labels
        dim_labels = [f'$f_{i}$' for i in range(n_dims)]
    
    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(111, projection='3d')
    
    # Set colorbar limits
    cbar_min = color_data.min()
    cbar_max = color_data.max()
    
    # 3D scatter plot with customizable mappings
    scatter_3d = ax_3d.scatter(x_data, y_data, z_data,
                              c=color_data,
                              cmap=cmap,
                              s=s,
                              alpha=alpha,
                              marker='o',
                              vmin=cbar_min,
                              vmax=cbar_max)
    
    # Plot nominal design if requested
    if show_nominal:
        nominal_design = experiment.nominal_design.cpu().numpy()
        # Nominal design should have same number of dimensions as designs
        if len(nominal_design) == n_dims:
            ax_3d.scatter(nominal_design[x_idx], nominal_design[y_idx], nominal_design[z_idx],
                        c=nominal_design[color_idx],
                        cmap=cmap,
                        s=s*1.5,
                        alpha=1.0,
                        vmin=cbar_min,
                        vmax=cbar_max,
                        marker='*',
                        label='Nominal Design',
                        edgecolors='black',
                        linewidths=1)
            ax_3d.legend(fontsize=12)
    
    # Configure 3D plot - use fig.suptitle to center title on entire figure (including colorbar)
    fig.suptitle('Design Space', fontsize=16, y=0.88, weight='bold')
    ax_3d.set_xlabel(dim_labels[x_idx], fontsize=14)
    ax_3d.set_ylabel(dim_labels[y_idx], fontsize=14)
    ax_3d.set_zlabel(dim_labels[z_idx], fontsize=14)
    ax_3d.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.8, aspect=20)
    cbar.set_label(dim_labels[color_idx], fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust top margin to accommodate suptitle
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, dpi=300)
    
    # Display figure in interactive environment
    if _is_interactive_environment():
        _display_figure(fig)
    plt.close(fig)


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