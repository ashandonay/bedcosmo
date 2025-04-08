import os
import mlflow
from mlflow.tracking import MlflowClient
import getdist
import numpy as np
from getdist import plots
from num_tracers import NumTracers
from util import run_eval
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import matplotlib.colors


def plot_posterior(samples, colors, legend_labels=None, show_scatter=False):
    g = plots.get_subplot_plotter(width_inch=7)
    
    if type(samples) != list:
        samples = [samples]
    if type(colors) != list:
        colors = [colors]
    if type(legend_labels) != list and legend_labels is not None:
        legend_labels = [legend_labels]

    # Add more line styles to handle all samples
    g.settings.line_styles = ['-'] * len(samples)
    
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
            'linestyle': '-'
        },
        contour_args={
            'ls': '-'
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
    samples = run_eval(run_id, eval_args, device=eval_args["device"])
    g = plot_posterior(samples, ["tab:blue"], show_scatter=show_scatter)
    plt.show()

def plot_run_steps(run_id, plot_steps, eval_args, show_scatter=False):
    client = MlflowClient()
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    all_samples = []
    all_areas = []
    color_list = []

    for i, step in enumerate(plot_steps):
        samples = run_eval(run_id, eval_args, step=step, device=eval_args["device"])
        all_samples.append(samples)
        
        # Assign a unique color for each step
        color = colors[i % len(colors)]
        color_list.append(color)
        
        # Calculate average 68% contour area for this step
        areas = get_contour_area(samples, 'Om', 'hrdrag', 0.68)[0]
        all_areas.append(areas)
    
    g = plot_posterior(all_samples, color_list, show_scatter=show_scatter)
    
    if g.fig.legends:
        for legend in g.fig.legends:
            legend.remove()

    custom_legend = []
    for i, step_label in enumerate(plot_steps):
        if step_label == 'last':
            step_label = client.get_run(run_id).data.params['steps']
        else:
            step_label = str(step_label)
        custom_legend.append(
            Line2D([0], [0], color=color_list[i], 
                   label=f'Step {step_label}, 68% Area: {all_areas[i]:.2f}')
        )
    g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
    plt.show()
    
def plot_exp_steps(exp_name, plot_steps, eval_args, show_scatter=False):
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    run_ids = [run.info.run_id for run in client.search_runs(exp_id)]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    all_samples = []
    all_areas = []
    
    for step in plot_steps:
        # For each step, collect samples from all runs
        step_samples = []
        for j, run_id in enumerate(run_ids):
            samples = run_eval(run_id, eval_args, step=step, device=eval_args["device"])
            all_samples.append(samples)
            step_samples.append(samples)
            
        # Calculate average 68% contour area for this step
        areas = get_contour_area(step_samples, 'Om', 'hrdrag', 0.68)
        
        avg_area = np.mean(areas) if areas else 0
        all_areas.append(avg_area)
    
    # Create a color list that assigns the same color to samples from the same step
    expanded_colors = []
    for i, color in enumerate(colors):
        expanded_colors.extend([color] * len(run_ids))
    
    g = plot_posterior(all_samples, expanded_colors, show_scatter=show_scatter)

    if g.fig.legends:  # Check if a legend exists
        for legend in g.fig.legends:
            legend.remove()  # Remove all existing legends

    # Create custom legend with group values and their average areas
    custom_legend = []
    for i, step_label in enumerate(plot_steps):

        if step_label == 'last':
            step_label = client.get_run(run_ids[0]).data.params['steps']
        else:
            step_label = str(step_label)
        group_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plot_steps)]
        custom_legend.append(
            Line2D([0], [0], color=group_color, 
                  label=f'Step {step_label}, Avg 68% Area: {all_areas[i]:.2f}')
        )
    g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
    plt.show()

def compare_runs_posterior(exp_name, filter_string, eval_args, show_scatter=False, excluded_runs=[]):
    """Compares runs from an experiment that match a given filter string.

    Args:
        exp_name: Name of the MLflow experiment.
        filter_string: An MLflow filter string to select runs (e.g., "params.param_name = 'value'").
        eval_args: Arguments for run_eval.
        show_scatter: Whether to show scatter points on the plot.
        excluded_runs: List of run IDs to exclude.
    """
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    
    # Select runs using the MLflow filter string
    print(f"Searching runs with filter: {filter_string}")
    filtered_runs = client.search_runs(exp_id, filter_string=filter_string)
    run_ids = [run.info.run_id for run in filtered_runs if run.info.run_id not in excluded_runs]
    
    if not run_ids:
        print("No runs found matching the filter.")
        return
        
    print(f"Found {len(run_ids)} matching runs.")
    
    all_samples = []
    # Ensure colors cycle if there are many runs
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] for i in range(len(run_ids))]
    
    for run_id in run_ids:
        samples = run_eval(run_id, eval_args, device=eval_args["device"])
        all_samples.append(samples) 
        
    # Use run IDs as labels for clarity
    g = plot_posterior(all_samples, colors, show_scatter=show_scatter, legend_labels=run_ids)
    g.fig.suptitle(f'Comparison of runs matching: {filter_string}', y=1.03)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def compare_exp_training(exp_name, var, excluded_runs=[]):
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert var to list if it's a single variable
    vars_list = var if isinstance(var, list) else [var]
    
    # Sort runs based on the first variable if it's numeric
    if type(eval(client.get_run(run_ids[0]).data.params[vars_list[0]])) is float or type(eval(client.get_run(run_ids[0]).data.params[vars_list[0]])) is int:
        run_ids = sorted(run_ids, key=lambda x: float(client.get_run(x).data.params[vars_list[0]]))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(run_ids)))
    losses = []
    steps = []
    
    for i, run_id in enumerate(run_ids):
        loss_history = client.get_metric_history(run_id, 'loss')
        loss = [metric.value for metric in loss_history]
        # append losses to one list
        losses.append(loss)
        eigs = np.load(f"../mlruns/{exp_id}/{run_id}/artifacts/eigs.npy")
        ax2.plot(eigs, alpha=0.7, linestyle='--', color=colors[i])
        if os.path.exists(f"../mlruns/{exp_id}/{run_id}/artifacts/brute_force_eigs.npy"):
            brute_force_eigs = np.load(f"../mlruns/{exp_id}/{run_id}/artifacts/brute_force_eigs.npy")
            ax2.plot(brute_force_eigs, color=colors[i])
    
    loss_min = np.array(losses).min()
    
    for i, run_id in enumerate(run_ids):
        loss_history = client.get_metric_history(run_id, 'loss')
        loss = np.array([metric.value for metric in loss_history])
        steps = len(loss)
        
        # Create label with all variables
        label_parts = []
        for v in vars_list:
            param_value = client.get_run(run_id).data.params[v]
            label_parts.append(f"{v}={param_value}")
        
        label = ", ".join(label_parts)
        
        ax1.plot(np.arange(steps)*25, loss-loss_min, alpha=0.5, color=colors[i], label=label)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')
    ax2.set_xlabel("Design Index")
    ax2.set_ylabel("Expected Information Gain [bits]")
    ax1.legend()
    plt.show()

def compare_runs_training(run_ids, var):

    client = MlflowClient()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert var to list if it's a single variable
    vars_list = var if isinstance(var, list) else [var]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    losses = []
    loss_mins = []
    steps = []
    
    for i, run_id in enumerate(run_ids):
        exp_id = client.get_run(run_id).info.experiment_id
        loss_history = client.get_metric_history(run_id, 'loss')
        loss = [metric.value for metric in loss_history]
        steps.append(len(loss))
        loss_mins.append(np.array(loss).min())
        losses.append(loss)
        eigs = np.load(f"../mlruns/{exp_id}/{run_id}/artifacts/eigs.npy")
        ax2.plot(eigs, alpha=0.7, linestyle='--', color=colors[i])
        if os.path.exists(f"../mlruns/{exp_id}/{run_id}/artifacts/brute_force_eigs.npy"):
            brute_force_eigs = np.load(f"../mlruns/{exp_id}/{run_id}/artifacts/brute_force_eigs.npy")
            ax2.plot(brute_force_eigs, color=colors[i])
    
    mins = np.array(loss_mins)
    losses = np.array(losses)
    
    for i, run_id in enumerate(run_ids):
        
        # Create label with all variables
        label_parts = []
        for v in vars_list:
            param_value = client.get_run(run_id).data.params[v]
            label_parts.append(f"{v}={param_value}")
        
        label = ", ".join(label_parts)
        
        ax1.plot(np.arange(steps[i])*25, losses[i]-mins[i], alpha=0.5, color=colors[i], label=label)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')
    ax2.set_xlabel("Design Index")
    ax2.set_ylabel("Expected Information Gain [bits]")
    ax1.legend()
    plt.show()

def compare_exp_posterior(exp_name, group_by_var, eval_args, show_scatter=False, excluded_runs=[]):
    print(f"Getting samples for {exp_name}, grouping by {group_by_var}")
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(exp_name).experiment_id
    all_run_ids = [run.info.run_id for run in client.search_runs(exp_id) if run.info.run_id not in excluded_runs]
    
    # Group runs by the specified variable
    grouped_runs = {}
    for run_id in all_run_ids:
        run = client.get_run(run_id)
        if group_by_var in run.data.params:
            group_value = run.data.params[group_by_var]
            if group_value not in grouped_runs:
                grouped_runs[group_value] = []
            grouped_runs[group_value].append(run_id)
    
    # Sort the groups
    sorted_groups = sorted(grouped_runs.keys(), 
                          key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)
    
    # Lists to store representative samples and their colors/areas
    representative_samples = []
    colors = []
    avg_areas = []
    std_areas = []
    # Process each group to find the representative sample
    for i, group_value in enumerate(sorted_groups):
        run_ids = grouped_runs[group_value]
        group_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]
        colors.append(group_color)
        
        # Collect samples and calculate areas for this group
        group_samples = []
        group_areas = []
        for run_id in run_ids:
            samples = run_eval(run_id, eval_args, device=eval_args["device"])
            group_samples.append(samples)
            area = get_contour_area([samples], 'Om', 'hrdrag', 0.68)[0]
            group_areas.append(area)
            
        # Calculate average area for the legend
        avg_areas.append(np.mean(group_areas))
        std_areas.append(np.std(group_areas))
        
        # Find the sample closest to the median area
        representative_samples.append(group_samples[np.argmin(np.abs(np.array(group_areas) - np.median(group_areas)))])
        # print the run id of the representative sample
        print(f"Representative sample run id for group {group_by_var}={group_value}: {run_ids[np.argmin(np.abs(np.array(group_areas) - np.median(group_areas)))]}")

    # Plot only the representative samples
    g = plot_posterior(representative_samples, colors, show_scatter=show_scatter)

    if g.fig.legends:  # Check if a legend exists
        for legend in g.fig.legends:
            legend.remove()  # Remove all existing legends

    # Create custom legend
    custom_legend = []
    for i, group_value in enumerate(sorted_groups):
        group_color = colors[i]
        if std_areas[i] > 0:
            custom_legend.append(
                Line2D([0], [0], color=group_color, 
                    label=f'n={group_value}, Avg 68% Contour Area: {avg_areas[i]:.3f}, Std: {std_areas[i]:.3f}')
            )
        else:
            custom_legend.append(
                Line2D([0], [0], color=group_color, 
                    label=f'n={group_value}, Avg 68% Contour Area: {avg_areas[i]:.3f}')
            )
    
    # Fallback to figure legend if the target axes is somehow None
    # Use bbox_to_anchor to position the legend precisely
    g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
    
    if len(run_ids) > 1:
        g.fig.suptitle(f'Posterior comparison for {group_by_var} with {len(run_ids)} runs per group', y=1.03)
    else:
        g.fig.suptitle(f'Posterior comparison for {group_by_var}', y=1.03)
    plt.show()

def get_contour_area(samples, param1, param2, level=0.68):
    samples = [samples] if type(samples) != list else samples
    areas = []

    for sample in samples:
        density = sample.get2DDensity(param1, param2)
        contour_level = density.getContourLevels([level])[0]
        
        # Plot in original space
        cs = plt.contour(density.x, density.y, density.P, 
                         levels=[contour_level])
        paths = cs.collections[0].get_paths()
        total_area = 0.0
        
        for path in paths:
            vertices = path.vertices
            x, y = vertices[:, 0], vertices[:, 1]
            
            # Calculate areas using Shoelace formula
            area_norm = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            
            total_area = area_norm
        
        areas.append(total_area)
        
    plt.close("all")
    return areas

def compare_contours(run_ids, param1, param2, eval_args, steps='last', level=0.68, show_grid=False):
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