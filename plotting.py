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
from util import run_eval, get_desi_samples
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import matplotlib.colors
import warnings
import contextlib
from datetime import datetime


def plot_posterior(samples, colors, legend_labels=None, show_scatter=False, line_style='-'):
    g = plots.get_subplot_plotter(width_inch=7)
    
    if type(samples) != list:
        samples = [samples]
    if type(colors) != list:
        colors = [colors]
    if type(legend_labels) != list and legend_labels is not None:
        legend_labels = [legend_labels]

    # Add more line styles to handle all samples
    g.settings.line_styles = [line_style] * len(samples)
    
    # First create the triangle plot with default settings
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
        contour_args={
            'ls': line_style
        }
    )
    
    # Get parameter names for the first root (assuming all samples have same parameters)
    param_names = g.param_names_for_root(samples[0])
    param_name_list = [p.name for p in param_names.names]
    
    if show_scatter:
        # Add histograms to diagonal plots without clearing
        for i, param in enumerate(param_name_list):
            if i < len(g.subplots) and i < len(g.subplots[i]):
                ax = g.subplots[i][i]  # Diagonal subplot
                
                # Get the current y limits to restore later
                current_ylim = ax.get_ylim()
                
                # Iterate through each sample and add histogram
                for k, sample in enumerate(samples):
                    # Get parameter values using the string name
                    param_index = sample.paramNames.list().index(param)
                    
                    if param_index is not None:
                        values = sample.samples[:, param_index]
                        
                        # Plot histogram with reduced alpha to not obscure the density curve
                        ax.hist(values, bins=30, alpha=0.5, color=colors[k], 
                               density=True, histtype='stepfilled', zorder=1)  # Lower zorder to keep behind the line
                
                # Restore original y limits to match the density curve
                ax.set_ylim(current_ylim)
        
        # Add scatter plots to lower triangle
        # Generate all combinations of parameters for scatter plots
        param_combinations = []
        for i in range(len(param_name_list)):
            for j in range(i+1, len(param_name_list)):
                param_combinations.append((param_name_list[i], param_name_list[j]))
        
        # Add scatter plots for each parameter combination
        for param_x, param_y in param_combinations:
            # Find the corresponding subplot
            for i in range(len(g.subplots)):
                for j in range(i):
                    if param_name_list[i] == param_y and param_name_list[j] == param_x:
                        ax = g.subplots[i][j]
                        # Add scatter for each sample set
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
    samples = run_eval(run_id, eval_args, device=eval_args["device"], exp='num_tracers')
    g = plot_posterior(samples, ["tab:blue"], show_scatter=show_scatter)
    plt.show()

def posterior_steps(run_id, steps, eval_args, type='all', cosmo_exp='num_tracers'):
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
        show_scatter (bool): Whether to show scatter points on the plot.
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
    if type == 'all':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_') and f.endswith('.pt') and not f.endswith('last.pt') and not f.endswith('loss.pt') and not f.endswith('area.pt')
        ])
    elif type == 'area':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_area')
        ])
    elif type == 'loss':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_loss')
        ])
    # get the checkpoints closest to the steps
    plot_checkpoints = [checkpoints[np.argmin(np.abs(np.array(checkpoints) - step))] for step in steps if step != 'best_loss' and step != 'best_nominal_area']
    # check for duplicates in plot_checkpoints and replace them with the next checkpoint index 
    for i, step in enumerate(plot_checkpoints):
        if step in plot_checkpoints[:i]:
            if np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1 < len(checkpoints):
                plot_checkpoints[i] = checkpoints[np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1]
            else:
                print(f"Warning: No next checkpoint found for step {steps[i]}")
                plot_checkpoints.pop(i)
    # remove any steps before 1000
    plot_checkpoints = [step for step in plot_checkpoints if step > 1000]
    # add best_loss and best_nominal_area to the plot_checkpoints if requested
    if 'best_loss' in steps:
        plot_checkpoints.append('best_loss')
    if 'best_nominal_area' in steps:
        plot_checkpoints.append('best_nominal_area')

    for i, step in enumerate(plot_checkpoints):
        samples = run_eval(run_id, eval_args, step=step, device=eval_args["device"], cosmo_exp=cosmo_exp)
        all_samples.append(samples)
        color_list.append(colors[i % len(colors)])
        area = get_contour_area(samples, 'Om', 'hrdrag', 0.68)[0]
        all_areas.append(area)

    desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
    desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', 0.68)[0]
    all_samples.append(desi_samples_gd)
    color_list.append('black')  
    all_areas.append(desi_area)
    g = plot_posterior(all_samples, color_list)
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
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
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
        eval_args=None, 
        loss_step_freq=25, 
        area_step_freq=100
        ):
    """
    Compares training loss and posterior contour area evolution for multiple MLflow runs
    on a single plot with two y-axes. Can take either an experiment name or a list of run IDs.

    Args:
        exp_name (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
        run_id (str or list, optional): Individual or list of specific MLflow run IDs to compare. If provided, exp_name is ignored.
        var (str or list): Parameter(s) from MLflow run params to include in the label.
        excluded_runs (list): List of run IDs to exclude.
        exp (str): Experiment name or ID (if needed for path).
        log_scale (bool): If True, use log scale for the loss y-axis.
        show_best (bool): If True, also plot contour area for best loss checkpoints.
        eval_args (dict, optional): Arguments for the run_eval function. If None, defaults to {"device": "cpu"}.
    """
    client = MlflowClient()
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
    # Set default eval_args if not provided
    if eval_args is None:
        eval_args = {"device": "cpu"}
    
    # Get run IDs either from experiment name or directly from input
    if exp_name is not None:
        exp_id = client.get_experiment_by_name(exp_name).experiment_id
        run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]
    elif run_id is not None:
        # make run_ids into a list if it's not already
        run_ids = [run_id] if isinstance(run_id, str) else run_id
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
            run = client.get_run(run_id)
            return tuple(float(run.data.params[v]) if v in run.data.params else float('inf') for v in vars_list)
        
        run_ids = sorted(run_ids, key=get_sort_key)
    
    # Create a figure and a primary axes (for loss)
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Create a secondary axes sharing the same x-axis (for contour area)
    ax2 = ax1.twinx()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines = [] # To store lines for the combined legend
    labels = [] # To store labels for the combined legend

    min_loss_overall = float('inf') # Keep track of min loss across all runs

    # First pass to get all losses and determine min loss for scaling
    all_losses = {}
    all_nominal_areas = {}
    for run_id in run_ids:
        loss_history = client.get_metric_history(run_id, 'loss')
        # Filter out potential NaN/inf values from loss history
        loss = [metric.value for metric in loss_history if np.isfinite(metric.value)]
        nominal_area = client.get_metric_history(run_id, 'nominal_area')
        nominal_area = [metric.value for metric in nominal_area if np.isfinite(metric.value)]
        all_losses[run_id] = loss
        all_nominal_areas[run_id] = nominal_area
        current_min_loss = np.min(loss)
        if current_min_loss < min_loss_overall:
            min_loss_overall = current_min_loss

    areas = []
    # Second pass to plot data
    for i, run_id in enumerate(run_ids):
        if run_id not in all_losses: # Skip if run had no valid loss points
            continue
        losses = all_losses[run_id][::loss_step_freq]
        training_steps_axis = np.arange(len(losses)) * loss_step_freq

        # Create label with all variables
        label_parts = []
        for v in vars_list:
            try:
                param_value = client.get_run(run_id).data.params[v]
                label_parts.append(f"{v}={param_value}")
            except:
                print(f"Warning: Parameter {v} not found for run {run_id}")
                
        base_label = ", ".join(label_parts)
        
        # --- Plot Loss on ax1 ---
        loss_plot_label = f"{base_label} (Loss)"
        if log_scale:
            line_loss, = ax1.plot(training_steps_axis, losses - min_loss_overall * 1.1, alpha=0.7, color=colors[i % len(colors)], label=loss_plot_label)
        else:
            # Original plotting for linear scale
            line_loss, = ax1.plot(training_steps_axis, losses, alpha=0.7, color=colors[i % len(colors)], label=loss_plot_label)
        lines.append(line_loss)
        labels.append(loss_plot_label)
        areas = []

        # --- Plot Nominal area on ax2 ---
        if area_step_freq % 100 != 0:
            raise ValueError("area_step_freq must be a factor of 100")
        nominal_area = all_nominal_areas[run_id][::area_step_freq//100]
        area_steps = np.arange(len(nominal_area)) * area_step_freq # recorded every 25th step
        line_nominal_area, = ax2.plot(area_steps, nominal_area, alpha=1.0, zorder=5, linewidth=2, color=colors[i % len(colors)], label='Nominal Area')
        lines.append(line_nominal_area)
        labels.append('Nominal Area')
        areas.append(nominal_area)

        if show_best:
            best_area = client.get_metric_history(run_id, 'best_nominal_area')
            best_areas = [metric.value for metric in best_area if np.isfinite(metric.value)]
            best_steps = [metric.step for metric in best_area if np.isfinite(metric.value)]
            if not best_areas: # Skip run if no valid loss points
                print(f"Warning: No valid best areas found for run {run_id}. Skipping.")
                continue
            # Use another unique variable name
            line_area_best, = ax2.plot(best_steps, best_areas, alpha=1.0, linestyle='-.', label='Best Area', color=colors[i % len(colors)], zorder=5, linewidth=2)
            # plot a star at the best last step
            ax2.plot(best_steps[-1], best_areas[-1], '*', markersize=8, zorder=10, color=colors[i % len(colors)])
            lines.append(line_area_best) # Append the correct line object
            labels.append('Best Area')
            areas.append(best_areas[-1])

    # --- Final Plot Configuration ---
    ax1.set_xlabel("Training Step")

    # Configure ax1 (Loss)
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    if log_scale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        # Adjust ylim slightly to give padding, avoid setting bottom exactly at min_loss_overall if not log
        ax1.set_ylim(min_loss_overall - np.abs(min_loss_overall) * 0.1, 1)
        ax2.set_ylim(min(areas) - np.abs(min(areas)) * 0.5, max(areas) + np.abs(max(areas)) * 0.5)

    # Configure ax2 (Contour Area)
    ax2.set_ylabel("Posterior Contour Area")
    ax2.tick_params(axis='y')

    # Determine legend font size based on number of runs
    num_runs = len(run_ids)
    if num_runs > 6: # Adjust this threshold as needed
        legend_fontsize = 'x-small' # Or specify point size, e.g., 8
    elif num_runs > 4:
        legend_fontsize = 'small'   # Or specify point size, e.g., 9
    else:
        legend_fontsize = 'medium'  # Or specify point size, e.g., 10

    # Combined Legend - Placed below the plot, columns match runs, adjusted font size
    ax1.legend(lines, labels, loc='upper center', 
               bbox_to_anchor=(0.5, -0.15), # Adjust y-anchor for spacing
               ncol=num_runs, 
               fontsize=legend_fontsize)

    # Adjust layout to make room for the legend below the plot
    # Increase bottom margin; adjust the value (0.0 to 1.0) as needed
    plt.subplots_adjust(bottom=0.25) 
    # fig.tight_layout(rect=[0, 0.1, 1, 0.95]) # Alternative using tight_layout rect
    
    # Set title based on what was provided
    if exp_name:
        plt.title(f"Training Loss and Posterior Area Evolution - {exp_name}")
        os.makedirs(f"{storage_path}/mlruns/{exp_id}/plots", exist_ok=True)
        save_path = f"{storage_path}/mlruns/{exp_id}/plots/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    elif len(run_ids) == 1:
        plt.title("Training Loss and Posterior Area Evolution")
        exp_id = client.get_run(run_ids[0]).info.experiment_id
        os.makedirs(f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/plots", exist_ok=True)
        save_path = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/plots/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        plt.title("Training Loss and Posterior Area Evolution")
        os.makedirs(f"{storage_path}/plots", exist_ok=True)
        save_path = f"{storage_path}/plots/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    show_figure(save_path)

def compare_posterior(
        exp_name=None, 
        run_ids=None, 
        var=None, 
        filter_string=None, 
        eval_args=None, 
        show_scatter=False, 
        excluded_runs=[], 
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
        eval_args = {"post_samples": 10000, "device": "cuda:0", "eval_seed": 1}
    
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
            return tuple(float(x) if x.replace('.', '', 1).isdigit() else x for x in group_tuple)
        
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
                samples = run_eval(run_id, eval_args, step=step, device=eval_args["device"], cosmo_exp=cosmo_exp)
                group_samples.append(samples)
                area = get_contour_area([samples], 'Om', 'hrdrag', 0.68)[0]
                group_areas.append(area)
                
            # Calculate average area for the legend
            avg_areas.append(np.mean(group_areas))
            std_areas.append(np.std(group_areas))
            
            # Find the sample closest to the median area
            representative_idx = np.argmin(np.abs(np.array(group_areas) - np.median(group_areas)))
            representative_samples.append(group_samples[representative_idx])
            print(f"Representative sample run id for group {', '.join([f'{vars_list[j]}={val}' for j, val in enumerate(group_key)])}: {group_run_ids[representative_idx]}")

        desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', 0.68)[0]

        representative_samples.append(desi_samples_gd)
        colors.append('black')  
        # Plot only the representative samples
        g = plot_posterior(representative_samples, colors, show_scatter=show_scatter)

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
            
            samples = run_eval(run_id, eval_args, step=step, device=eval_args["device"], cosmo_exp=cosmo_exp)
            all_samples.append(samples)
        
        desi_samples_gd = get_desi_samples(run.data.params['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', 0.68)[0]
        all_samples.append(desi_samples_gd)
        colors.append('black')
        labels.append('DESI')
        g = plot_posterior(all_samples, colors, show_scatter=show_scatter, legend_labels=labels)
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
            samples.append(run_eval(run_id, eval_args, step=step, device=eval_args["device"]))
    
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

if __name__ == "__main__":

    eval_args = {"post_samples": 30000, "device": "cuda:1", "eval_seed": 1}
    plot_training(
        exp_name='base_NAF_gamma_fixed', 
        var='pyro_seed', 
        eval_args=eval_args,
        log_scale=True,
        show_best=True,
        show_checkpoints=False,
        cosmo_exp='num_tracers'
    )