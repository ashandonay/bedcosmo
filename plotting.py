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
        areas = []
        for sample in step_samples:
            area = get_contour_area(sample, 'Om', 'hrdrag', 0.68)
            areas.append(area)
        
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
    g.fig.legend(handles=custom_legend, loc='upper right')
    plt.show()


def compare_training(exp_name, var, excluded_runs=[]):
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

def compare_exp(exp_name, group_by_var, eval_args, show_scatter=False, excluded_runs=[]):
    all_samples = []
    avg_areas = []
    colors = []
    
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
    # Process each group
    for i, group_value in enumerate(sorted_groups):
        run_ids = grouped_runs[group_value]
        group_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]  # Cycle through colors
        
        samples = []
        for run_id in run_ids:
            run_samples = run_eval(run_id, eval_args, device=eval_args["device"])
            samples.append(run_samples)
        
        all_samples += samples
        colors += [group_color] * len(run_ids)
        
        # Calculate areas for this group
        areas = {}
        for run_id, s in zip(run_ids, samples):
            area = get_contour_area(s, 'Om', 'hrdrag', 0.68)
            areas[run_id] = area
        print(f"Areas for {group_by_var}={group_value}: {areas}")
        avg_areas.append(np.mean(list(areas.values())))
    
    g = plot_posterior(all_samples, colors, show_scatter=show_scatter)

    if g.fig.legends:  # Check if a legend exists
        for legend in g.fig.legends:
            legend.remove()  # Remove all existing legends

    # Create custom legend with group values and their average areas
    custom_legend = []
    for i, group_value in enumerate(sorted_groups):
        group_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10]
        custom_legend.append(
            Line2D([0], [0], color=group_color, 
                  label=f'{group_by_var}={group_value}, Avg 68% Area: {avg_areas[i]:.2f}')
        )
    
    g.fig.legend(handles=custom_legend, loc='upper right')
    plt.show()

def get_contour_area(samples, param1, param2, level=0.68):
    """
    Calculate the area of a 2D contour at a given confidence level.
    
    Parameters:
    -----------
    samples : MCSamples
        GetDist MCSamples object
    param1, param2 : str
        Names of the parameters to plot
    level : float
        Confidence level (e.g., 0.68 for 1σ, 0.95 for 2σ)
    
    Returns:
    --------
    float
        Area of the contour
    """
    # Get the density grid
    density = samples.get2DDensity(param1, param2)
    
    # Calculate the contour level by finding the value that encloses the specified probability
    sorted_density = np.sort(density.P.ravel())[::-1]
    cumsum = np.cumsum(sorted_density)
    cumsum = cumsum / cumsum[-1]  # Normalize to 1
    contour_level = sorted_density[np.where(cumsum <= level)[0][-1]]
    
    # Get the contour points
    X, Y = np.meshgrid(density.x, density.y)
    cs = plt.contour(X, Y, density.P.T, levels=[contour_level])
    paths = cs.collections[0].get_paths()
    
    # Calculate total area of all contours at this level
    total_area = 0.0
    for path in paths:
        vertices = path.vertices
        x, y = vertices[:, 0], vertices[:, 1]
        
        # Shoelace formula for polygon area
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        total_area += area
    
    plt.close('all')
    return total_area

def compare_contour_areas(samples_list, param1, param2, level=0.68):
    """
    Compare contour areas for multiple sample sets
    
    Parameters:
    -----------
    samples_list : list of MCSamples
        List of GetDist MCSamples objects
    param1, param2 : str
        Names of the parameters to plot
    level : float
        Confidence level
    
    Returns:
    --------
    dict
        Dictionary mapping sample indices to their contour areas
    """
    areas = {}
    for i, samples in enumerate(samples_list):
        area = get_contour_area(samples, param1, param2, level)
        areas[i] = area
    return areas