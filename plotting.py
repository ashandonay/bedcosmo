import sys
import os
import io
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
import contextlib
import torch
from datetime import datetime
from pyro_oed_src import posterior_loss
import json


def plot_posterior(samples, colors, legend_labels=None, show_scatter=False, line_style='-', levels=[0.68, 0.95]):
    """
    Plots posterior distributions using GetDist triangle plots.

    Args:
        samples (list): List of GetDist MCSamples objects.
        colors (list): List of colors for each sample.
        legend_labels (list, optional): List of legend labels for each sample.
        show_scatter (bool): If True, show scatter/histograms on the 1D/2D plots.
        line_style (str): Line style for contours.
        levels (float or list, optional): Contour levels to use (e.g., 0.68 or [0.68, 0.95]).
            If a single float is provided, it is converted to a list.
            If None, the default GetDist settings are used.
    Returns:
        g: GetDist plotter object with the generated triangle plot.
    """
    g = plots.get_subplot_plotter(width_inch=7)

    if type(samples) != list:
        samples = [samples]
    if type(colors) != list:
        colors = [colors]
    if type(legend_labels) != list and legend_labels is not None:
        legend_labels = [legend_labels]

    # Add more line styles to handle all samples
    g.settings.line_styles = [line_style] * len(samples)

    # Prepare contour_args with custom levels if provided
    contour_args = {'ls': line_style}

    # Set contour levels if provided
    if levels is not None:
        if isinstance(levels, float):
            levels = [levels]
        for sample in samples:
            sample.updateSettings({'contours': levels})
        g.settings.num_plot_contours = len(levels)

    # Create triangle plot
    g.triangle_plot(
        samples,
        colors=colors,
        legend_labels=legend_labels,
        filled=False,
        normalized=True,
        diag1d_kwargs={
            'colors': colors,
            'normalized': True,
            'linestyle': line_style
        },
        contour_args=contour_args
    )

    param_names = g.param_names_for_root(samples[0])
    param_name_list = [p.name for p in param_names.names]

    if show_scatter:
        for i, param in enumerate(param_name_list):
            if i < len(g.subplots) and i < len(g.subplots[i]):
                ax = g.subplots[i][i]
                current_ylim = ax.get_ylim()
                for k, sample in enumerate(samples):
                    param_index = sample.paramNames.list().index(param)
                    if param_index is not None:
                        values = sample.samples[:, param_index]
                        ax.hist(values, bins=30, alpha=0.5, color=colors[k],
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
                            g.add_2d_scatter(
                                sample,
                                param_x,
                                param_y,
                                color=colors[k],
                                ax=ax,
                                scatter_size=1,
                                alpha=0.8,
                            )

    return g

def plot_run(run_id, eval_args, show_scatter=False):
    samples = run_eval(run_id, eval_args, exp='num_tracers')
    g = plot_posterior(samples, ["tab:blue"], show_scatter=show_scatter)
    plt.show()

def posterior_steps(run_id, steps, eval_args, level=0.68,   type='all', cosmo_exp='num_tracers'):
    """
    Plots posterior distributions at different training steps for either a single run, 
    multiple specific runs, or all runs in an experiment.
    
    Args:
        exp_name (str, optional): Name of the MLflow experiment. If provided and run_ids is None, 
                                  all runs in this experiment will be used.
        run_ids (str or list, optional): Single run ID or list of specific MLflow run IDs to plot. 
                                        If provided, exp_name is ignored.
        plot_steps (list): List of steps to plot. Can include 'last' or 'best' as special values.
        eval_args (dict): Arguments for run_eval.
        type (str): Type of steps to plot. Can be 'all', 'area', or 'loss'.
        cosmo_exp (str): Name of the cosmology experiment.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"

    exp_id = client.get_run(run_id).info.experiment_id
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    all_samples = []
    all_areas = []
    color_list = []

    checkpoint_dir = f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/'
    checkpoint_files = os.listdir(checkpoint_dir)
    plot_checkpoints = get_checkpoints(steps, checkpoint_files, type, cosmo_exp, verbose=False)

    for i, step in enumerate(plot_checkpoints):
        samples = run_eval(run_id, eval_args, step=step, cosmo_exp=cosmo_exp)
        all_samples.append(samples)
        color_list.append(colors[i % len(colors)])
        area = get_contour_area(samples, 'Om', 'hrdrag', level)[0]
        all_areas.append(area)

    desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
    desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', level)[0]
    all_samples.append(desi_samples_gd)
    color_list.append('black')  
    all_areas.append(desi_area)
    g = plot_posterior(all_samples, color_list, levels=[level])
    # Remove existing legends if any
    if g.fig.legends:
        for legend in g.fig.legends:
            legend.remove()

    # Create custom legend
    custom_legend = []
    # If we have a single run, show each step with its area
    for i, step in enumerate(plot_checkpoints):
        step_label = step
        if step == 'last':
            step_label = client.get_run(run_id).data.params['steps']
        elif step == 'best':
            step_label = 'Best Loss'
        
        # Get the area for this step (first run's area from the tuple)
        area = all_areas[i]
        
        custom_legend.append(
            Line2D([0], [0], color=colors[i % len(colors)], 
                    label=f'Step {step_label}, 68% Area: {area:.2f}')
        )
    custom_legend.append(
        Line2D([0], [0], color='black', 
            label=f'DESI, Area (68% Contour): {desi_area:.3f}')
    )
    g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
    save_path = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    show_figure(save_path)

def plot_training(
        exp_name=None,
        run_id=None,
        var=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        log_scale=False,
        show_best=False,
        eval_args=None, # eval_args is not used in this function currently
        loss_step_freq=25,
        area_step_freq=100,
        lr_step_freq=1 # Plot LR more frequently if needed
        ):
    """
    Compares training loss, learning rate, and posterior contour area evolution
    for multiple MLflow runs using three vertically stacked subplots sharing the x-axis.
    Top plot: Loss.
    Middle plot: Learning Rate.
    Bottom plot: Posterior Contour Area.
    Can take either an experiment name or a list of run IDs.

    Args:
        exp_name (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_id (str or list, optional): Individual or list of specific MLflow run IDs to compare. If provided, exp_name is ignored.
        var (str or list): Parameter(s) from MLflow run params to include in the label.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Experiment name or ID (if needed for path).
        log_scale (bool): If True, use log scale for the y-axes (Loss, LR, Area). Loss values <= 0 will be omitted in log scale.
        show_best (bool): If True, also plot contour area for best loss checkpoints on the bottom plot.
        eval_args (dict, optional): Not currently used in this plotting function.
        loss_step_freq (int): Sampling frequency for plotting loss points.
        area_step_freq (int): Sampling frequency for plotting nominal area points (must be multiple of 100).
        lr_step_freq (int): Sampling frequency for plotting learning rate points.
    """
    client = MlflowClient()
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"

    # Get run IDs either from experiment name or directly from input
    if exp_name is not None:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                print(f"Error: Experiment '{exp_name}' not found.")
                return
            exp_id = exp.experiment_id
            run_infos = client.search_runs(exp_id)
            # Filter runs based on status (e.g., only FINISHED) if desired
            # run_infos = [r for r in run_infos if r.info.status == 'FINISHED']
            run_ids = [run.info.run_id for run in run_infos if run.info.run_id not in excluded_runs]
        except Exception as e:
            print(f"Error fetching runs for experiment '{exp_name}': {e}")
            return
    elif run_id is not None:
        run_ids = [run_id] if isinstance(run_id, str) else run_id
        # Filter out excluded runs
        run_ids = [rid for rid in run_ids if rid not in excluded_runs]
        # Try to get experiment ID from the first valid run if needed for saving later
        if run_ids:
            try:
                exp_id = client.get_run(run_ids[0]).info.experiment_id
            except Exception as e:
                print(f"Warning: Could not determine experiment ID from run {run_ids[0]}: {e}")
                exp_id = None # Fallback
        else:
            exp_id = None
    else:
        print("Either exp_name or run_ids must be provided.")
        return

    if not run_ids:
        print("No runs found to process.")
        return

    # Convert var to list if it's a single variable
    vars_list = var if isinstance(var, list) else [var] if var is not None else []

    if vars_list:
        # Sort runs based on the values of the specified parameters
        def get_sort_key(run_id):
            try:
                run = client.get_run(run_id)
                # Handle potential missing parameters gracefully
                key_tuple = []
                for v in vars_list:
                    param_val_str = run.data.params.get(v)
                    if param_val_str is not None:
                        try:
                            # Attempt conversion to float for numerical sorting
                            key_tuple.append(float(param_val_str))
                        except ValueError:
                            # If conversion fails, treat as string
                            key_tuple.append(param_val_str)
                    else:
                        # Assign a value that sorts missing params last
                        key_tuple.append(float('inf'))
                return tuple(key_tuple)

            except Exception as e:
                print(f"Warning: Could not fetch params for run {run_id} for sorting: {e}")
                return tuple(float('inf') for _ in vars_list) # Sort problematic runs last

        run_ids = sorted(run_ids, key=get_sort_key)

    # --- Data Fetching ---
    all_metrics = {}
    min_loss_overall = float('inf')
    max_loss_overall = float('-inf')
    min_lr_overall = float('inf')
    max_lr_overall = float('-inf')
    min_area_overall = float('inf')
    max_area_overall = float('-inf')

    valid_run_ids = []
    for run_id_iter in run_ids: # Use different variable name to avoid confusion with outer scope run_id if it was a single string
        try:
            # Fetch metrics
            loss_hist = client.get_metric_history(run_id_iter, 'loss')
            lr_hist = client.get_metric_history(run_id_iter, 'lr')
            nom_area_hist = client.get_metric_history(run_id_iter, 'nominal_area')
            best_area_hist = client.get_metric_history(run_id_iter, 'best_nominal_area') if show_best else []

            # Process and filter NaNs/Infs
            loss = [(m.step, m.value) for m in loss_hist if np.isfinite(m.value)]
            lr = [(m.step, m.value) for m in lr_hist if np.isfinite(m.value)]
            nom_area = [(m.step, m.value) for m in nom_area_hist if np.isfinite(m.value)]
            best_area = [(m.step, m.value) for m in best_area_hist if np.isfinite(m.value)] if show_best else []

            if not loss:
                print(f"Warning: No valid loss points found for run {run_id_iter}. Skipping.")
                continue

            all_metrics[run_id_iter] = {
                'loss': loss,
                'lr': lr,
                'nominal_area': nom_area,
                'best_area': best_area
            }
            valid_run_ids.append(run_id_iter) # Add to list of runs we can actually plot

            # Update overall min/max for axis scaling (only from valid points)
            run_losses = [v for s, v in loss]
            run_lrs = [v for s, v in lr]
            run_nom_areas = [v for s, v in nom_area]
            run_best_areas = [v for s, v in best_area]

            if run_losses:
                min_loss_overall = min(min_loss_overall, np.min(run_losses))
                max_loss_overall = max(max_loss_overall, np.max(run_losses))
            if run_lrs:
                min_lr_overall = min(min_lr_overall, np.min(run_lrs))
                max_lr_overall = max(max_lr_overall, np.max(run_lrs))
            if run_nom_areas:
                min_area_overall = min(min_area_overall, np.min(run_nom_areas))
                max_area_overall = max(max_area_overall, np.max(run_nom_areas))
            if run_best_areas: # Only update if show_best is True and data exists
                min_area_overall = min(min_area_overall, np.min(run_best_areas))
                max_area_overall = max(max_area_overall, np.max(run_best_areas))

        except Exception as e:
            print(f"Error processing metrics for run {run_id_iter}: {e}. Skipping.")
            continue

    if not valid_run_ids:
        print("No runs with valid data to plot.")
        return

    run_ids = valid_run_ids # Use only the runs we successfully processed

    # --- Plotting Setup ---
    # Create figure with three subplots, sharing x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Plotting Data ---
    for i, run_id_iter in enumerate(run_ids):
        metrics = all_metrics[run_id_iter]
        color = colors[i % len(colors)]

        # --- Prepare Labels ---
        label_parts = []
        run_params = {}
        try:
            run_params = client.get_run(run_id_iter).data.params
        except Exception as e:
            print(f"Warning: Could not fetch params for run {run_id_iter} for label: {e}")
        for v in vars_list:
            param_value = run_params.get(v, "N/A")
            label_parts.append(f"{v}={param_value}")
        base_label = ", ".join(label_parts) if label_parts else f"Run {i+1}"

        # --- Plot Loss (ax1) ---
        loss_data = metrics['loss']
        if loss_data:
            loss_steps, loss_values = zip(*loss_data)
            # Apply sampling frequency
            sampled_indices = np.arange(0, len(loss_steps), loss_step_freq)
            plot_loss_steps = np.array(loss_steps)[sampled_indices]
            plot_loss_values = np.array(loss_values)[sampled_indices]

            # Filter non-positive values if log scale is requested for this axis
            if log_scale:
                ax1.plot(plot_loss_steps, plot_loss_values - min_loss_overall, alpha=0.8, color=color, label=base_label)
            else:
                ax1.plot(plot_loss_steps, plot_loss_values, alpha=0.8, color=color, label=base_label)

        # --- Plot Nominal Area (ax2) ---
        if area_step_freq % 100 != 0:
            print("Warning: area_step_freq should ideally be a multiple of 100 as nominal_area is logged every 100 steps.")
        sampling_rate = max(1, area_step_freq // 100) # Sample every 'sampling_rate' points from the metric history

        nom_area_data = metrics['nominal_area']
        if nom_area_data:
            nom_area_steps, nom_area_values = zip(*nom_area_data)
            # Apply sampling frequency
            sampled_indices = np.arange(0, len(nom_area_steps), sampling_rate)
            plot_nom_area_steps = np.array(nom_area_steps)[sampled_indices]
            plot_nom_area_values = np.array(nom_area_values)[sampled_indices]

            # Filter non-positive values if log scale is requested for this axis
            if log_scale:
                positive_mask = plot_nom_area_values > 0
                plot_nom_area_steps_filtered = plot_nom_area_steps[positive_mask]
                plot_nom_area_values_filtered = plot_nom_area_values[positive_mask]
                if len(plot_nom_area_values_filtered) < len(plot_nom_area_values):
                     print(f"Warning: Run {run_id_iter} - Omitted {len(plot_nom_area_values) - len(plot_nom_area_values_filtered)} non-positive Nominal Area values for log scale plot.")
                line_nom_area, = ax2.plot(plot_nom_area_steps_filtered, plot_nom_area_values_filtered, alpha=0.9, linewidth=1.5, color=color, label=base_label)
            else:
                 line_nom_area, = ax2.plot(plot_nom_area_steps, plot_nom_area_values, alpha=0.9, linewidth=1.5, color=color, label=base_label)

        # --- Plot Best Area (ax2) ---
        if show_best:
            best_area_data = metrics['best_area']
            if best_area_data:
                best_steps, best_areas = zip(*best_area_data)
                plot_best_area_steps = np.array(best_steps)
                plot_best_area_values = np.array(best_areas)

                # Filter non-positive values if log scale is requested for this axis
                if log_scale:
                    positive_mask = plot_best_area_values > 0
                    plot_best_area_steps_filtered = plot_best_area_steps[positive_mask]
                    plot_best_area_values_filtered = plot_best_area_values[positive_mask]
                    if len(plot_best_area_values_filtered) < len(plot_best_area_values):
                        print(f"Warning: Run {run_id_iter} - Omitted {len(plot_best_area_values) - len(plot_best_area_values_filtered)} non-positive Best Area values for log scale plot.")
                    # Plot line
                    line_area_best, = ax2.plot(plot_best_area_steps_filtered, plot_best_area_values_filtered, alpha=1.0, linestyle='-.', linewidth=2, color=color, label=f"{base_label} (Best)")
                    # Plot star at the last *positive* best step if log scale
                    if len(plot_best_area_steps_filtered) > 0:
                        ax2.plot(plot_best_area_steps_filtered[-1], plot_best_area_values_filtered[-1], '*', markersize=8, zorder=10, color=color)

                else:
                    # Plot line
                    line_area_best, = ax2.plot(plot_best_area_steps, plot_best_area_values, alpha=1.0, linestyle='-.', linewidth=2, color=color, label=f"{base_label} (Best)")
                    # Plot star at the last best step
                    if len(plot_best_area_steps) > 0:
                        ax2.plot(plot_best_area_steps[-1], plot_best_area_values[-1], '*', markersize=8, zorder=10, color=color)

        # --- Plot Learning Rate (ax3) ---
        lr_data = metrics['lr']
        if lr_data:
            lr_steps, lr_values = zip(*lr_data)
            # Apply sampling frequency
            sampled_indices = np.arange(0, len(lr_steps), lr_step_freq)
            plot_lr_steps = np.array(lr_steps)[sampled_indices]
            plot_lr_values = np.array(lr_values)[sampled_indices]

            # Filter non-positive values if log scale is requested for this axis
            if log_scale:
                positive_mask = plot_lr_values > 0
                plot_lr_steps_filtered = plot_lr_steps[positive_mask]
                plot_lr_values_filtered = plot_lr_values[positive_mask]
                if len(plot_lr_values_filtered) < len(plot_lr_values):
                     print(f"Warning: Run {run_id_iter} - Omitted {len(plot_lr_values) - len(plot_lr_values_filtered)} non-positive LR values for log scale plot.")
                ax3.plot(plot_lr_steps_filtered, plot_lr_values_filtered, alpha=0.8, color=color, label=base_label)
            else:
                ax3.plot(plot_lr_steps, plot_lr_values, alpha=0.8, color=color, label=base_label)

    # get only unique cosmo models
    cosmo_models = list(set([client.get_run(run_id).data.params['cosmo_model'] for run_id in run_ids]))
    for cm in cosmo_models:
        desi_samples_gd = get_desi_samples(cm)
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', 0.68)[0]
        ax2.axhline(desi_area, color='black', linestyle='--', label=f'DESI ({cm}), Area: {desi_area:.3f}', alpha=0.5)
    # --- Final Plot Configuration ---

    # Determine legend font size based on number of runs
    num_runs = len(run_ids)
    if num_runs > 10: legend_fontsize = 'xx-small'
    elif num_runs > 6: legend_fontsize = 'x-small'
    elif num_runs > 4: legend_fontsize = 'small'
    else: legend_fontsize = 'medium'

    # Configure ax1 (Loss)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Configure ax2 (Contour Area)
    ax2.set_ylabel("Posterior Contour Area")
    ax2.tick_params(axis='y')
    ax2.legend(loc='best', fontsize=legend_fontsize, title="Runs")
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Configure ax3 (Learning Rate)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Learning Rate")
    ax3.tick_params(axis='y')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)


    # Apply log scale if requested
    if log_scale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        # Limits are automatically handled by matplotlib for log scale based on filtered data
    else:
        # Set linear scale limits with padding only if data range is valid
        if max_loss_overall > min_loss_overall:
            loss_pad = (max_loss_overall - min_loss_overall) * 0.05
            ax1.set_ylim(min_loss_overall - loss_pad, max_loss_overall + loss_pad)
        elif np.isfinite(min_loss_overall): # Handle constant loss case
             ax1.set_ylim(min_loss_overall - 0.5, min_loss_overall + 0.5) # Example padding

        if max_area_overall > min_area_overall:
            area_pad = (max_area_overall - min_area_overall) * 0.05
            ax2.set_ylim(min_area_overall - area_pad, max_area_overall + area_pad)
        elif np.isfinite(min_area_overall): # Handle constant area case
             ax2.set_ylim(min_area_overall - 0.5, min_area_overall + 0.5) # Example padding

        if max_lr_overall > min_lr_overall :
             lr_pad = (max_lr_overall - min_lr_overall) * 0.05
             ax3.set_ylim(min_lr_overall - lr_pad, max_lr_overall + lr_pad)
        elif np.isfinite(min_lr_overall): # Handle constant LR case
             ax3.set_ylim(min_lr_overall * 0.9, min_lr_overall * 1.1) # Relative padding


    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Add space for title

    # --- Title and Saving ---
    title_suffix = f" - {exp_name}" if exp_name else f" - {len(run_ids)} Runs"
    fig.suptitle(f"Training Evolution{title_suffix}", fontsize=16)

    # Determine save path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if exp_name and exp_id:
        save_dir = f"{storage_path}/mlruns/{exp_id}/plots"
        save_path = f"{save_dir}/training_comparison_{timestamp}.png"
    elif len(run_ids) == 1 and exp_id:
        # Use the specific run's artifact location
        run_id_single = run_ids[0]
        save_dir = f"{storage_path}/mlruns/{exp_id}/{run_id_single}/artifacts/plots"
        save_path = f"{save_dir}/training_{timestamp}.png"
    else:
        # Fallback save location if experiment context is lost or multiple non-experiment runs
        save_dir = f"{storage_path}/plots"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/training_comparison_{timestamp}.png"


    os.makedirs(save_dir, exist_ok=True) # Ensure directory exists before saving
    show_figure(save_path)
    plt.close(fig) # Close the figure after saving/showing

def compare_posterior(
        exp_name=None, 
        run_ids=None, 
        var=None, 
        filter_string=None, 
        eval_args=None, 
        show_scatter=False, 
        excluded_runs=[],
        level=0.68,
        cosmo_exp='num_tracers', 
        step='best_loss'
        ):
    """Compares posterior distributions across multiple runs.
    
    Args:
        exp_name (str, optional): Name of the MLflow experiment. Required if run_ids not provided.
        run_ids (list, optional): List of specific run IDs to compare. If provided, exp_name is used only for grouping.
        var (str or list, optional): Parameter(s) to group runs by (e.g., 'num_tracers' or ['num_tracers', 'gamma']).
                                    If provided, runs are grouped by these parameter(s).
        filter_string (str, optional): MLflow filter string to select runs (e.g., "params.param_name = 'value'").
        eval_args (dict, optional): Arguments for run_eval. Defaults to {"device": "cpu"}.
        show_scatter (bool): Whether to show scatter points on the plot.
        excluded_runs (list): List of run IDs to exclude.
        cosmo_exp (str): Name of the cosmology experiment folder.
        step (str or int): Which checkpoint to evaluate ('best', 'last', or specific step number).
    """
    client = MlflowClient()
    
    # Set default eval_args if not provided
    if eval_args is None:
        eval_args = {"n_samples": 10000, "device": "cuda:0", "eval_seed": 1}
    
    # Get run IDs based on input parameters
    if run_ids is None:
        if exp_name is None:
            raise ValueError("Either exp_name or run_ids must be provided.")
        
        exp_id = client.get_experiment_by_name(exp_name).experiment_id
        
        if filter_string:
            print(f"Searching runs with filter: {filter_string}")
            filtered_runs = client.search_runs(exp_id, filter_string=filter_string)
            run_ids = [run.info.run_id for run in filtered_runs if run.info.run_id not in excluded_runs]
        else:
            run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]
    else:
        # Filter out excluded runs
        run_ids = [run_id for run_id in run_ids if run_id not in excluded_runs]
        # Try to get experiment ID from the first run if exp_name not provided
        if exp_name is None and run_ids:
            exp_id = client.get_run(run_ids[0]).info.experiment_id
            exp = client.get_experiment(exp_id)
            exp_name = exp.name if exp else "Unknown Experiment"
    
    if not run_ids:
        print("No runs found to process.")
        return

    print(f"Found {len(run_ids)} runs to analyze.")
    
    # If grouping by variable(s)
    if var:
        # Convert var to list if it's a single string
        vars_list = var if isinstance(var, list) else [var]
        print(f"Grouping runs by {', '.join(vars_list)}")
        
        # Group runs by the specified variable(s)
        grouped_runs = {}
        for run_id in run_ids:
            run = client.get_run(run_id)
            
            # Create a tuple of group values for all variables
            group_values = []
            valid_run = True
            for v in vars_list:
                if v in run.data.params:
                    group_values.append(run.data.params[v])
                else:
                    valid_run = False
                    break
            
            if valid_run:
                group_key = tuple(group_values)
                if group_key not in grouped_runs:
                    grouped_runs[group_key] = []
                grouped_runs[group_key].append(run_id)
        
        # Sort the groups
        def sort_key(group_tuple):
            try:
                # Convert each value to float if possible, otherwise keep as string
                key_tuple = []
                for val in group_tuple:
                    try:
                        key_tuple.append(float(val))
                    except ValueError:
                        key_tuple.append(val)
                return tuple(key_tuple)
            except Exception as e:
                print(f"Warning: Could not convert group values for sorting: {e}")
                return tuple(float('inf') for _ in group_tuple)  # Sort problematic groups last
        
        try:
            sorted_groups = sorted(grouped_runs.keys(), key=sort_key)
        except (ValueError, TypeError):
            # Fallback to default sorting if numeric conversion fails
            sorted_groups = sorted(grouped_runs.keys())
        
        # Lists to store representative samples and their colors/areas
        representative_samples = []
        colors = []
        avg_areas = []
        std_areas = []
        
        # Process each group to find the representative sample
        for i, group_key in enumerate(sorted_groups):
            group_run_ids = grouped_runs[group_key]
            group_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]
            colors.append(group_color)
            
            # Collect samples and calculate areas for this group
            group_samples = []
            group_areas = []
            for run_id in group_run_ids:
                samples = run_eval(run_id, eval_args, step=step, cosmo_exp=cosmo_exp)
                group_samples.append(samples)
                area = get_contour_area([samples], 'Om', 'hrdrag', level)[0]
                group_areas.append(area)
                
            # Calculate average area for the legend
            avg_areas.append(np.mean(group_areas))
            std_areas.append(np.std(group_areas))
            
            # Find the sample closest to the median area
            representative_idx = np.argmin(np.abs(np.array(group_areas) - np.median(group_areas)))
            representative_samples.append(group_samples[representative_idx])
            print(f"Representative sample run id for group {', '.join([f'{vars_list[j]}={val}' for j, val in enumerate(group_key)])}: {group_run_ids[representative_idx]}")

        desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', level)[0]

        representative_samples.append(desi_samples_gd)
        colors.append('black')  
        # Plot only the representative samples
        g = plot_posterior(representative_samples, colors, show_scatter=show_scatter, levels=[level])

        # Create custom legend
        if g.fig.legends:  # Check if a legend exists
            for legend in g.fig.legends:
                legend.remove()  # Remove all existing legends

        custom_legend = []
        for i, group_key in enumerate(sorted_groups):
            group_color = colors[i]
            # Create a descriptive label for the group
            group_label = ', '.join([f'{vars_list[j]}={val}' for j, val in enumerate(group_key)])
            
            if std_areas[i] > 0:
                custom_legend.append(
                    Line2D([0], [0], color=group_color, 
                        label=f'{group_label}, Area (68% Contour): {avg_areas[i]:.3f}, Std: {std_areas[i]:.3f}')
                )
            else:
                custom_legend.append(
                    Line2D([0], [0], color=group_color, 
                        label=f'{group_label}, Area (68% Contour): {avg_areas[i]:.3f}')
                )
        custom_legend.append(
            Line2D([0], [0], color='black', 
                label=f'DESI, Area (68% Contour): {desi_area:.3f}')
        )
        g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
        
        if len(run_ids) > 1:
            vars_str = ', '.join(vars_list)
            g.fig.suptitle(f'Posterior comparison grouped by {vars_str} with {len(run_ids)} total runs', y=1.03)
        else:
            vars_str = ', '.join(vars_list)
            g.fig.suptitle(f'Posterior comparison grouped by {vars_str}', y=1.03)
    
    # If not grouping, just plot all runs
    else:
        all_samples = []
        # Ensure colors cycle if there are many runs
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] for i in range(len(run_ids))]
        
        # Get labels for the runs
        labels = []
        for run_id in run_ids:
            run = client.get_run(run_id)
            # Create a descriptive label if possible
            label_parts = []
            for param in ['num_tracers', 'num_hidden', 'learning_rate', 'batch_size']:
                if param in run.data.params:
                    label_parts.append(f"{param}={run.data.params[param]}")
            
            if label_parts:
                labels.append(", ".join(label_parts))
            else:
                labels.append(run_id[:8])  # Use shortened run ID if no params
            
            samples = run_eval(run_id, eval_args, step=step, cosmo_exp=cosmo_exp)
            all_samples.append(samples)
        
        desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', 0.68)[0]
        all_samples.append(desi_samples_gd)
        colors.append('black')
        labels.append('DESI')
        g = plot_posterior(all_samples, colors, show_scatter=show_scatter, legend_labels=labels, levels=[level])
        title = f'Posterior comparison for {exp_name}' if exp_name else 'Posterior comparison'
        if filter_string:
            title += f' (filter: {filter_string})'
        g.fig.suptitle(title, y=1.03)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    if exp_name:
        os.makedirs(f"{storage_path}/mlruns/{exp_id}/plots", exist_ok=True)
        save_path = f"{storage_path}/mlruns/{exp_id}/plots/posterior_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        save_path = f"{storage_path}/plots/posterior_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    show_figure(save_path)

def get_contour_area(samples, param1, param2, level=0.68):
    samples = [samples] if type(samples) != list else samples
    areas = []

    # Create a temporary figure that won't be shown
    temp_fig, temp_ax = plt.subplots()
    for sample in samples:
        density = sample.get2DDensity(param1, param2)
        contour_level = density.getContourLevels([level])[0]
        
        # Plot contour on the temporary axes
        cs = temp_ax.contour(density.x, density.y, density.P, 
                           levels=[contour_level])
        
        # Check if any contours were found at this level
        if not cs.collections or not cs.collections[0].get_paths():
            warnings.warn(f"No contour found for level {level}. Assigning area 0.")
            areas.append(0.0)
            continue # Skip to the next sample if no path found

        paths = cs.collections[0].get_paths()
        total_area = 0.0
        
        for path in paths:
            vertices = path.vertices
            x, y = vertices[:, 0], vertices[:, 1]
            
            # Calculate areas using Shoelace formula
            area_norm = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            total_area = area_norm # Assuming only one closed contour contributes significantly
        
        areas.append(total_area)
        
    # Close only the temporary figure
    plt.close(temp_fig) 
    return areas

def compare_contours(run_ids, param1, param2, eval_args, steps='best', level=0.68, show_grid=False):
    samples = []
    run_ids = [run_ids] if type(run_ids) != list else run_ids
    steps = [steps] if type(steps) != list else steps
    for run_id in run_ids:
        for step in steps:
            samples.append(run_eval(run_id, eval_args, step=step))
    
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
    
def loss_area_plot(exp_name, var_name, step_interval=1000, excluded_runs=[], cosmo_exp='num_tracers'):
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]
    all_run_losses = []
    all_run_areas = []
    vars_values = [] # Renamed from vars to avoid conflict with built-in
    # Sort runs by the specified parameter value
    run_ids = sorted(run_ids, key=lambda x: float(client.get_run(x).data.params[var_name]))
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
    show_figure(save_path)

def plot_lr_schedule(initial_lr, gamma, gamma_freq, steps=100000):
    steps = np.arange(0, steps, 1)
    lr = initial_lr * gamma ** (steps / gamma_freq)
    legend_label = f'initial_lr={initial_lr}, gamma={gamma}, gamma_freq={gamma_freq}'
    plt.plot(steps, lr, label=legend_label)

    return lr[-1]

def show_figure(save_path):
    # Check if running in a TTY, otherwise assume interactive (like notebook) and show
    try:
        is_tty = os.isatty(sys.stdout.fileno())
    except (io.UnsupportedOperation, AttributeError):
        is_tty = False # Treat as non-TTY if fileno() fails (e.g., in notebooks)

    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    if not is_tty:
        plt.show()

def plot_eig_steps(run_id, steps, eval_args, cosmo_exp='num_tracers', verbose=False):
    client = MlflowClient()
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    run_args = parse_mlflow_params(run.data.params)
    # Load designs directly using np.load on the downloaded path
    designs_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="designs.npy")
    designs = np.load(designs_path)
    designs = torch.tensor(designs, device=eval_args["device"])
    with open(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="classes.json")) as f:
        classes = json.load(f)
    checkpoint_files = os.listdir(f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints")
    checkpoint_steps = get_checkpoints(
        steps, 
        checkpoint_files,
        type='all', 
        cosmo_exp=cosmo_exp, 
        verbose=verbose
        )
    plt.figure(figsize=(10, 6))
    # load in model at steps
    for s in checkpoint_steps:
        num_tracers, posterior_flow = load_model(run_id, s, eval_args, cosmo_exp)
        with torch.no_grad():
            _, eigs = posterior_loss(design=designs,
                                            model=num_tracers.pyro_model,
                                            guide=posterior_flow,
                                            num_particles=eval_args["n_samples"],
                                            observation_labels=["y"],
                                            target_labels=num_tracers.cosmo_params,
                                            evaluation=True,
                                            nflow=True,
                                            analytic_prior=False,
                                            condition_design=run_args["condition_design"])
        eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
        plt.plot(eigs_bits, label=f'Step {s}')

    nominal_design = torch.tensor(num_tracers.desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=eval_args["device"])
    with torch.no_grad():
        _, nominal_eig = posterior_loss(design=nominal_design.unsqueeze(0),
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=eval_args["n_samples"],
                                        observation_labels=["y"],
                                        target_labels=num_tracers.cosmo_params,
                                        evaluation=True,
                                        nflow=True,
                                        analytic_prior=False,
                                        condition_design=run_args["condition_design"])
    nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
    plt.axhline(y=nominal_eig_bits, color='black', linestyle='--', label='Nominal EIG')
    plt.xlabel("Design Index")
    plt.ylabel("EIG")
    plt.legend()
    plt.tight_layout()
    show_figure(f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/plots/eig_steps.png")

if __name__ == "__main__":

    eval_args = {"n_samples": 30000, "device": "cuda:1", "eval_seed": 1}
    plot_training(
        exp_name='base_NAF_gamma_fixed', 
        var='pyro_seed', 
        eval_args=eval_args,
        log_scale=True,
        show_best=True,
        show_checkpoints=False,
        cosmo_exp='num_tracers'
    )