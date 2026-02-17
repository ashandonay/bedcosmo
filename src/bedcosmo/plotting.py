import sys
import os
import io
import contextlib

import torch
import mlflow
from mlflow.tracking import MlflowClient
import getdist
import numpy as np
from getdist import plots
from bedcosmo.util import (
    get_runs_data, init_experiment, init_nf, load_model, auto_seed, convert_color,
    load_nominal_samples, get_contour_area, get_nominal_samples, sort_key_for_group_tuple,
)
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
from bedcosmo.pyro_oed_src import posterior_loss
import json
import yaml
from IPython.display import display
import glob as glob_module
import traceback

home_dir = os.environ["HOME"]
try:
    from desi_y1_plotting import KP7StylePaper
    style = KP7StylePaper()
except ImportError:
    KP7StylePaper = None
    style = None

# Disable LaTeX rendering globally to avoid "latex could not be found" errors
# This overrides any LaTeX settings that might be set by the style object
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times', 'serif']


# ============================================================================
# Base Plotter Class
# ============================================================================

class BasePlotter:
    """
    Base class for plotting functionality with common MLflow setup and utilities.
    Provides shared methods for MLflow client initialization, storage path management,
    run data fetching, and figure saving.
    """
    
    def __init__(self, cosmo_exp='num_tracers'):
        """
        Initialize the base plotter.
        
        Args:
            cosmo_exp (str): Cosmology experiment folder name.
        """
        self.cosmo_exp = cosmo_exp
        self.storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
        self._client = None
        self._mlflow_uri_set = False
    
    @property
    def client(self):
        """Lazy initialization of MLflow client."""
        if self._client is None or not self._mlflow_uri_set:
            mlflow.set_tracking_uri(self.storage_path + "/mlruns")
            self._client = MlflowClient()
            self._mlflow_uri_set = True
        return self._client
    
    def get_runs_data(self, mlflow_exp=None, run_ids=None, excluded_runs=None, 
                      filter_string=None, parse_params=True):
        """
        Fetch run data from MLflow.
        
        Args:
            mlflow_exp (str, optional): Name of the MLflow experiment.
            run_ids (str or list, optional): A single run ID or a list of run IDs.
            excluded_runs (list, optional): A list of run IDs to exclude.
            filter_string (str, optional): An MLflow filter string.
            parse_params (bool): Whether to parse MLflow parameters.
            
        Returns:
            tuple: (run_data_list, experiment_id, actual_mlflow_exp)
        """
        return get_runs_data(
            mlflow_exp=mlflow_exp,
            run_ids=run_ids,
            excluded_runs=excluded_runs or [],
            filter_string=filter_string,
            parse_params=parse_params,
            cosmo_exp=self.cosmo_exp
        )
    
    def get_save_dir(self, run_id=None, experiment_id=None, subdir="plots"):
        """
        Get the directory path for saving plots.
        
        Args:
            run_id (str, optional): Specific run ID for run-specific plots.
            experiment_id (str, optional): Experiment ID for experiment-level plots.
            subdir (str): Subdirectory name (default: "plots").
            
        Returns:
            str: Path to the save directory.
        """
        if run_id and experiment_id:
            return f"{self.storage_path}/mlruns/{experiment_id}/{run_id}/artifacts/{subdir}"
        elif experiment_id:
            return f"{self.storage_path}/mlruns/{experiment_id}/{subdir}"
        else:
            return f"{self.storage_path}/{subdir}"
    
    def save_figure(self, fig, filename, run_id=None, experiment_id=None, 
                   subdir="plots", dpi=300, close_fig=True, display_fig=True):
        """
        Save a figure to the appropriate directory.
        
        Args:
            fig: Matplotlib figure object.
            filename (str): Filename for the saved figure.
            run_id (str, optional): Specific run ID for run-specific plots.
            experiment_id (str, optional): Experiment ID for experiment-level plots.
            subdir (str): Subdirectory name (default: "plots").
            dpi (int): DPI for saving the figure.
            close_fig (bool): Whether to close the figure after saving.
            display_fig (bool): Whether to display the figure.
            
        Returns:
            str: Path to the saved figure.
        """
        save_dir = self.get_save_dir(run_id=run_id, experiment_id=experiment_id, subdir=subdir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        # Save the figure
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved plot to {save_path}")
        
        # Display figure if requested and in interactive environment
        if display_fig and self._is_interactive_environment():
            self._display_figure(fig)
        
        # Close figure if requested
        if close_fig:
            plt.close(fig)
        
        return save_path
    
    def _is_interactive_environment(self):
        """Check if we're in an interactive environment (not TTY)."""
        try:
            return not os.isatty(sys.stdout.fileno())
        except (io.UnsupportedOperation, AttributeError):
            return True
    
    def _display_figure(self, fig):
        """Display a figure in an interactive environment."""
        try:
            display(fig)
        except Exception:
            plt.show()
    
    def generate_filename(self, prefix, suffix="png", timestamp=True):
        """
        Generate a filename with optional timestamp.
        
        Args:
            prefix (str): Filename prefix.
            suffix (str): File extension (default: "png").
            timestamp (bool): Whether to include timestamp.
            
        Returns:
            str: Generated filename.
        """
        if timestamp:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{prefix}_{timestamp_str}.{suffix}"
        else:
            return f"{prefix}.{suffix}"

    def eig_designs(
        self,
        eig_values,
        input_designs,
        design_labels,
        nominal_design,
        eig_std_values=None,
        nominal_eig=None,
        grid_eig_values=None,
        grid_eig_std_values=None,
        nominal_grid_eig=None,
        sort=True,
        include_nominal=True,
        title=None,
        save_path=None,
        color="tab:green",
    ):
        """
        Plots sorted EIG values and corresponding designs.

        This core method accepts data directly, enabling reuse from both
        MLflow-based RunPlotter and standalone grid-based contexts.

        Args:
            eig_values (np.ndarray): EIG values per design.
            input_designs (np.ndarray): Design array with shape (n_designs, n_design_dims).
            design_labels (list): Labels for each design dimension.
            nominal_design (np.ndarray): Nominal design values.
            eig_std_values (np.ndarray, optional): EIG std per design.
            nominal_eig (float, optional): Nominal EIG value.
            grid_eig_values (np.ndarray, optional): Grid EIG values.
            grid_eig_std_values (np.ndarray, optional): Grid EIG std values.
            nominal_grid_eig (float, optional): Nominal grid EIG value.
            sort (bool): Whether to sort designs by EIG (default: True).
            include_nominal (bool): Whether to include the nominal EIG in the plot (default: True).
            title (str, optional): Custom title for the plot.
            save_path (str, optional): Explicit save path. If None, uses self.save_figure/generate_filename.

        Returns:
            tuple: (fig, (ax0, ax1)) matplotlib figure and axes objects. ax1 may be None for 1D designs without sorting.
        """
        print(f"Generating EIG designs plot...")

        eig_values = np.asarray(eig_values)
        input_designs = np.asarray(input_designs)
        nominal_design = np.asarray(nominal_design)

        if eig_values.size == 0:
            raise ValueError("No EIG values provided")

        if len(input_designs) <= 1:
            print("Warning: Single or no designs available, skipping EIG designs plot.")
            return None

        if eig_std_values is None:
            eig_std_values = np.zeros_like(eig_values)
        else:
            eig_std_values = np.asarray(eig_std_values)

        if grid_eig_values is not None:
            grid_eig_values = np.asarray(grid_eig_values)
        if grid_eig_std_values is not None:
            grid_eig_std_values = np.asarray(grid_eig_std_values)

        # Sort designs if requested
        if sort:
            sorted_idx = np.argsort(eig_values)[::-1]
        else:
            sorted_idx = np.arange(len(eig_values))

        sorted_designs = input_designs[sorted_idx]
        sorted_eigs = eig_values[sorted_idx]
        sorted_eigs_std = eig_std_values[sorted_idx]
        has_grid = (
            grid_eig_values is not None
            and grid_eig_values.size == eig_values.size
            and grid_eig_values.size > 0
        )
        if has_grid:
            sorted_grid_eigs = grid_eig_values[sorted_idx]
            if grid_eig_std_values is not None and grid_eig_std_values.size == eig_values.size:
                sorted_grid_eigs_std = grid_eig_std_values[sorted_idx]
            else:
                sorted_grid_eigs_std = np.zeros_like(sorted_grid_eigs)

        # Check if designs are 1D or multi-dimensional
        is_1d_design = (sorted_designs.shape[1] == 1)

        # Find nominal design position
        nominal_sorted_pos = None
        try:
            nominal_idx = int(np.argmin(np.linalg.norm(input_designs - nominal_design, axis=1)))
            matches = np.where(sorted_idx == nominal_idx)[0]
            if matches.size > 0:
                nominal_sorted_pos = int(matches[0])
        except:
            pass

        # Create figure with subplots
        if is_1d_design and not sort:
            fig, ax0 = plt.subplots(figsize=(16, 6))
            ax1 = None
            x_vals = sorted_designs[:, 0]
        else:
            if is_1d_design and sort:
                height_ratios = [0.65, 0.15]
            else:
                height_ratios = [0.6, 0.2]
            fig = plt.figure(figsize=(16, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.0)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            fig.subplots_adjust(left=0.06, right=0.88)
            bbox0 = ax0.get_position()
            bbox1 = ax1.get_position()
            cbar_width = 0.02
            cbar_gap = 0.01
            cbar_left = bbox1.x1 + cbar_gap
            cbar_bottom = bbox1.y0
            cbar_height = bbox0.y1 - bbox1.y0
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            plt.setp(ax0.get_xticklabels(), visible=False)
            ax0.tick_params(axis='x', which='both', length=0)
            x_vals = np.arange(len(sorted_eigs))

        # Plot EIG with error bars
        from matplotlib.colors import to_rgba
        fill_color = to_rgba(color, alpha=0.3)
        ax0.fill_between(x_vals, sorted_eigs - sorted_eigs_std, sorted_eigs + sorted_eigs_std,
                         color=fill_color, zorder=1)
        ax0.plot(x_vals, sorted_eigs, label='EIG', color=color, linewidth=2.5, zorder=5)
        if has_grid:
            ax0.fill_between(
                x_vals,
                sorted_grid_eigs - sorted_grid_eigs_std,
                sorted_grid_eigs + sorted_grid_eigs_std,
                color='tab:green',
                alpha=0.15,
                zorder=1,
            )
            ax0.plot(
                x_vals,
                sorted_grid_eigs,
                label='Grid-based EIG',
                color='tab:green',
                linewidth=2.0,
                zorder=6,
            )

        # Plot nominal EIG
        if include_nominal and nominal_eig is not None:
            ax0.axhline(y=nominal_eig, color='tab:blue', linestyle='--',
                       label='Nominal EIG', linewidth=2, zorder=10)
        if include_nominal and nominal_grid_eig is not None:
            ax0.axhline(
                y=nominal_grid_eig,
                color='tab:green',
                linestyle=':',
                label='Nominal Grid EIG',
                linewidth=2,
                zorder=10,
            )

        # Plot optimal design
        optimal_idx = np.argmax(sorted_eigs)
        optimal_x = x_vals[optimal_idx]
        optimal_y = sorted_eigs[optimal_idx]
        ax0.axvline(optimal_x, color='tab:orange', linestyle=':', linewidth=2, zorder=8)
        ax0.plot(optimal_x, optimal_y, 'o', color='tab:orange', markersize=8, zorder=9,
                label='Optimal Design')

        # Set axis labels
        if is_1d_design and not sort:
            ax0.set_xlabel(f'${design_labels[0]}$', fontsize=12, weight='bold')
            ax0.set_xlim(x_vals.min(), x_vals.max())
        else:
            ax0.set_xlim(-0.5, len(sorted_eigs) - 0.5)

        ax0.set_ylabel("Expected Information Gain [bits]", fontsize=12, weight='bold')
        ax0.legend(loc='lower left', fontsize=9, framealpha=0.9)
        ax0.grid(True, alpha=0.3)

        # Plot design heatmap
        if ax1 is not None:
            if is_1d_design and sort:
                im = ax1.imshow(sorted_designs.T, aspect='auto', cmap='viridis')
                ax1.set_ylabel('')
                ax1.set_yticks([0])
                # Format label: use as-is if already has $, otherwise wrap with $
                label0 = design_labels[0] if design_labels[0].startswith('$') else f'${design_labels[0]}$'
                ax1.set_yticklabels([label0])
                ax1.tick_params(axis='y', length=0)
                if nominal_sorted_pos is not None:
                    ax1.axvline(nominal_sorted_pos, color='black', linestyle=':', linewidth=1.5)
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Design Value', labelpad=10, fontsize=12, weight='bold')
            else:
                # Multi-dimensional designs - use ratio to nominal
                if np.any(nominal_design == 0):
                    plot_data = sorted_designs.T
                    cmap = 'viridis'
                    im = ax1.imshow(plot_data, aspect='auto', cmap=cmap)
                else:
                    plot_data = (sorted_designs / nominal_design[np.newaxis, :]).T
                    cmap = 'RdBu'
                    from matplotlib.colors import TwoSlopeNorm
                    vmin = plot_data.min()
                    vmax = plot_data.max()
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                    im = ax1.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm)

                ax1.set_xlabel("Design Index (sorted by EIG)" if sort else "Design Index",
                              fontsize=12, weight='bold')
                ax1.set_yticks(np.arange(len(design_labels)))
                # Format labels: use as-is if already has $, otherwise wrap with $
                formatted_labels = [label if label.startswith('$') else f'${label}$' for label in design_labels]
                ax1.set_yticklabels(formatted_labels)
                ax1.set_xlim(-0.5, len(sorted_designs) - 0.5)
                ax1.set_ylabel('')
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label('Ratio to Nominal Design' if cmap == 'RdBu' else 'Design Value',
                              labelpad=15, fontsize=12, weight='bold')
            ax0.spines['bottom'].set_visible(False)
            ax1.spines['bottom'].set_visible(True)

        sort_title = "Sorted" if sort else ""
        if title is None:
            title = f'{sort_title} EIG per Design'
        fig.suptitle(title, fontsize=16, y=0.95, weight='bold')

        # Save figure
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved plot to {save_path}")

        return fig, (ax0, ax1)

    def generate_posterior(
        self,
        experiment,
        posterior_flow=None,
        input_designs=None,
        eig_values=None,
        nominal_eig=None,
        display=('nominal', 'optimal'),
        levels=(0.68,),
        guide_samples=1000,
        device="cuda:0",
        seed=1,
        plot_prior=False,
        transform_output=True,
        title=None,
        grid_samples=None,
        nominal_grid_eig=None,
        save_path=None,
    ):
        """
        Generates posterior plots for nominal and/or optimal designs.

        This core method accepts data directly, enabling reuse from both
        MLflow-based RunPlotter and standalone grid-based contexts.

        Args:
            experiment: Experiment object (for cosmo_params, latex_labels, central_val, etc.)
            posterior_flow: Optional normalizing flow model. If None, NF samples are not generated.
            input_designs (np.ndarray, optional): All designs array.
            eig_values (np.ndarray, optional): EIG values (to find optimal).
            nominal_eig (float, optional): Nominal EIG value.
            display (tuple/list): Designs to display. Can include 'nominal' and/or 'optimal'.
            levels (tuple/list): Contour level(s) to plot.
            guide_samples (int): Number of samples to generate.
            device (str): Device to use.
            seed (int): Random seed.
            plot_prior (bool): If True, also plot the prior.
            transform_output (bool): Whether to transform output to physical space.
            title (str, optional): Title of the plot.
            grid_samples (np.ndarray, optional): Grid-based posterior parameter samples.
            nominal_grid_eig (float, optional): Nominal grid EIG value.
            save_path (str, optional): Explicit save path. If None, figure is not auto-saved by this method.

        Returns:
            GetDist plotter object.
        """
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            levels = [levels]

        auto_seed(seed)

        all_samples = []
        all_colors = []
        all_alphas = []
        all_line_styles = []
        legend_labels = []

        # Generate samples for nominal design if requested
        if 'nominal' in display and posterior_flow is not None:
            nominal_context = experiment.nominal_context
            nominal_samples_gd = experiment.get_guide_samples(
                posterior_flow, nominal_context,
                num_samples=guide_samples,
                transform_output=transform_output
            )

            all_samples.append(nominal_samples_gd)
            all_colors.append('tab:blue')
            all_alphas.append(1.0)
            all_line_styles.append('-')
            eig_str = f", EIG: {nominal_eig:.3f} bits" if nominal_eig is not None else ""
            legend_labels.append(f'Nominal Design (NF){eig_str}')

        # Generate samples for optimal design if requested
        if input_designs is not None:
            input_designs = np.asarray(input_designs)
        has_multiple_designs = input_designs is not None and len(input_designs) > 1

        if 'optimal' in display and posterior_flow is not None and has_multiple_designs and eig_values is not None:
            eig_values = np.asarray(eig_values)
            optimal_idx = np.argmax(eig_values)
            optimal_design = input_designs[optimal_idx]
            optimal_eig = float(eig_values[optimal_idx])

            optimal_design_tensor = torch.tensor(optimal_design, device=device, dtype=torch.float64)
            optimal_context = torch.cat([optimal_design_tensor, experiment.central_val], dim=-1)

            optimal_samples_gd = experiment.get_guide_samples(
                posterior_flow, optimal_context,
                num_samples=guide_samples,
                transform_output=transform_output
            )

            all_samples.append(optimal_samples_gd)
            all_colors.append('tab:orange')
            all_alphas.append(1.0)
            all_line_styles.append('-')
            eig_str = f", EIG: {optimal_eig:.3f} bits"
            legend_labels.append(f'Optimal Design (NF){eig_str}')
        elif 'optimal' in display and posterior_flow is not None and input_designs is not None and not has_multiple_designs:
            input_design_tensor = torch.tensor(input_designs[0], device=device, dtype=torch.float64)
            input_context = torch.cat([input_design_tensor, experiment.central_val], dim=-1)

            input_samples_gd = experiment.get_guide_samples(
                posterior_flow, input_context,
                num_samples=guide_samples,
                transform_output=transform_output
            )

            all_samples.append(input_samples_gd)
            all_colors.append('tab:orange')
            all_alphas.append(1.0)
            all_line_styles.append('-')
            optimal_eig = float(np.asarray(eig_values)[0]) if eig_values is not None else None
            eig_str = f", EIG: {optimal_eig:.3f} bits" if optimal_eig is not None else ""
            legend_labels.append(f'Input Design (NF){eig_str}')

        if grid_samples is not None:
            grid_samples_np = np.asarray(grid_samples, dtype=np.float64)
            if grid_samples_np.ndim == 1:
                grid_samples_np = grid_samples_np.reshape(-1, 1)
            if grid_samples_np.ndim != 2:
                raise ValueError(
                    f"grid_samples must be 2D or 1D, got shape {grid_samples_np.shape}."
                )
            expected_dim = len(experiment.cosmo_params)
            if grid_samples_np.shape[1] != expected_dim:
                raise ValueError(
                    "grid_samples dimension does not match parameter count: "
                    f"{grid_samples_np.shape[1]} vs {expected_dim}."
                )

            grid_samples_tensor = torch.as_tensor(
                grid_samples_np, device=experiment.device, dtype=torch.float64
            )
            if not transform_output:
                grid_samples_tensor = experiment.params_to_unconstrained(grid_samples_tensor)
            grid_samples_np = grid_samples_tensor.detach().cpu().numpy()

            with contextlib.redirect_stdout(io.StringIO()):
                grid_samples_gd = getdist.MCSamples(
                    samples=grid_samples_np,
                    names=experiment.cosmo_params,
                    labels=experiment.latex_labels,
                )

            all_samples.append(grid_samples_gd)
            all_colors.append('tab:green')
            all_alphas.append(1.0)
            all_line_styles.append('-')
            eig_str = (
                f", EIG: {nominal_grid_eig:.3f} bits"
                if nominal_grid_eig is not None
                else ""
            )
            legend_labels.append(f'Nominal Design (Grid){eig_str}')

        # Get DESI MCMC samples for reference
        if self.cosmo_exp == 'num_tracers':
            try:
                nominal_samples_mcmc = experiment.get_nominal_samples(transform_output = not transform_output)
                all_samples.append(nominal_samples_mcmc)
                all_colors.append('black')
                all_alphas.append(1.0)
                all_line_styles.append('--')
                legend_labels.append('Nominal Design (MCMC)')
            except NotImplementedError:
                print(f"Warning: get_nominal_samples not implemented for {self.cosmo_exp}, skipping MCMC reference.")

        if plot_prior and hasattr(experiment, 'get_prior_samples'):
            prior_samples_gd = experiment.get_prior_samples(num_samples=guide_samples)
            all_samples.append(prior_samples_gd)
            all_colors.append('black')
            all_alphas.append(1.0)
            all_line_styles.append('-')
            legend_labels.append('Prior')

        if not all_samples:
            print("Warning: No samples to plot.")
            return None

        # Set label attribute on each MCSamples object so GetDist uses correct labels
        for sample, label in zip(all_samples, legend_labels):
            sample.label = label

        # Create plot
        plot_width = 10
        g = self.plot_posterior(
            all_samples,
            all_colors,
            legend_labels=legend_labels,
            levels=levels,
            width_inch=plot_width,
            alpha=all_alphas,
            line_style=all_line_styles
        )

        if self.cosmo_exp == 'num_visits':
            g.subplots[0, 0].axvline(experiment.central_z, color='black', linestyle='--')

        # Calculate dynamic font sizes
        n_params = len(all_samples[0].paramNames.names)
        base_fontsize = max(6, min(18, plot_width * (0.2 + 0.42 * np.sqrt(n_params))))
        if n_params == 1:
            base_fontsize = max(base_fontsize, 12)
        title_fontsize = base_fontsize * 1.15
        legend_fontsize = base_fontsize * 0.65
        if n_params == 1:
            legend_fontsize = max(legend_fontsize, 10)

        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        # Create custom legend with proper formatting
        custom_legend = []
        for i, label in enumerate(legend_labels):
            color = all_colors[i]
            custom_legend.append(
                Line2D([0], [0], color=color, label=label, linewidth=1.2, linestyle=all_line_styles[i])
            )

        if title is None:
            title = "Posterior Evaluation"
        g.fig.suptitle(title, fontsize=title_fontsize, weight='bold')
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), fontsize=legend_fontsize)
        leg.set_in_layout(False)

        # Save figure
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            g.fig.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved plot to {save_path}")

        return g

    def load_eig_data_file(self, artifacts_dir, eval_step=None):
        """
        Load the most recent completed eig_data JSON file from the artifacts directory.
        
        Args:
            artifacts_dir (str): Path to the artifacts directory containing eig_data files
            eval_step (str or int, optional): If provided, verify that the loaded file contains this step
        
        Returns:
            tuple: (json_path, data) where json_path is the path to the file and data is the loaded JSON.
                   Returns (None, None) if no valid file is found.
        
        Raises:
            ValueError: If no completed eig_data files are found, if file cannot be loaded, or if eval_step is not found in the data.
        """
        if not os.path.exists(artifacts_dir):
            raise ValueError(f"Artifacts directory not found: {artifacts_dir}")
        
        # Find all eig_data JSON files
        eig_files = glob_module.glob(f"{artifacts_dir}/eig_data_*.json")
        
        if len(eig_files) == 0:
            raise ValueError(f"No eig_data JSON files found in {artifacts_dir}")
        
        # Sort by filename (most recent first)
        eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)
        
        # Check each file for completion status (most recent first)
        for json_path in eig_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if evaluation is completed using status field
                status = data.get('status')
                if status != 'complete':
                    # Skip incomplete files
                    continue
                
                # If eval_step is provided, verify it exists in the data
                if eval_step is not None:
                    step_str = f"step_{eval_step}" if not str(eval_step).startswith('step_') else str(eval_step)
                    if step_str not in data:
                        # This file doesn't have the requested step, try next file
                        continue
                
                # Found a complete file (and it has the requested step if eval_step was provided)
                return json_path, data
                    
            except Exception as e:
                # Skip files that can't be loaded and continue to next
                print(f"Warning: Error loading {json_path}: {e}, skipping...")
                continue
        
        # No completed files found (or no file with the requested eval_step)
        if eval_step is not None:
            raise ValueError(f"No completed eig_data files with step {eval_step} found in {artifacts_dir}")
        else:
            raise ValueError(f"No completed eig_data files found in {artifacts_dir}")
    
    def plot_posterior(
        self,
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
        Shared method available to all plotter classes.

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
        getdist_2D_ratio = 1.0
        g = plots.get_single_plotter(width_inch=width_inch, ratio=getdist_2D_ratio, scaling=True)
        
        # Apply style settings if provided (like KP7StylePaper)
        if style is not None:
            g.settings.__dict__.update(style.settings.__dict__)
        
        # Calculate dynamic font sizes for axis labels based on plot width and number of parameters
        if isinstance(samples, list) and len(samples) > 0:
            n_params = len(samples[0].paramNames.names)
            axis_label_fontsize = max(9, min(22, width_inch * (0.3 + 0.5 * np.sqrt(n_params))))
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

        if isinstance(line_style, str):
            line_style = [line_style] * len(samples)
        elif isinstance(line_style, list):
            if len(line_style) < len(samples):
                # Extend the list by repeating the last element
                line_style = line_style + [line_style[-1]] * (len(samples) - len(line_style))
            elif len(line_style) > len(samples):
                # Truncate if too many elements
                line_style = line_style[:len(samples)]

        if isinstance(show_scatter, bool):
            show_scatter = [show_scatter] * len(samples)
        elif isinstance(show_scatter, list):
            if len(show_scatter) < len(samples):
                # Extend the list by repeating the last element
                show_scatter = show_scatter + [show_scatter[-1]] * (len(samples) - len(show_scatter))
            elif len(show_scatter) > len(samples):
                # Truncate if too many elements
                show_scatter = show_scatter[:len(samples)]

        # Set line styles in GetDist settings (each entry needs linestyle + color or legend gets color='')
        g.settings.line_styles = list(zip(line_style, colors))
        if isinstance(alpha, (int, float)):
            adjusted_alpha = alpha * contour_alpha_factor
            g.settings.plot_args = {'alpha': adjusted_alpha}
        else:
            adjusted_alpha = [a * contour_alpha_factor for a in alpha]

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
                        if i == j:
                            lines = ax.get_lines()
                            n_lines_per_sample = len(lines) // len(samples) if len(samples) > 0 else 0
                            for sample_idx in range(len(samples)):
                                start_idx = sample_idx * n_lines_per_sample
                                end_idx = (sample_idx + 1) * n_lines_per_sample
                                for line in lines[start_idx:end_idx]:
                                    line.set_alpha(adjusted_alpha[sample_idx])
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


# ============================================================================
# Run Plotter Class
# ============================================================================

class RunPlotter(BasePlotter):
    """
    Plotter class for generating plots about a specific MLflow run.
    Handles training plots, evaluation plots, and other run-specific visualizations.
    """
    
    def __init__(self, run_id, cosmo_exp='num_tracers'):
        """
        Initialize the run plotter.
        
        Args:
            run_id (str): MLflow run ID to plot.
            cosmo_exp (str): Cosmology experiment folder name.
        """
        super().__init__(cosmo_exp=cosmo_exp)
        self.run_id = run_id
        self._run_data = None
        self._experiment_id = None
    
    def _get_artifacts_dir(self):
        """
        Helper method to get the artifacts directory for this run.
        
        Returns:
            str: Path to the artifacts directory
        """
        if self._experiment_id is not None:
            artifacts_dir = f"{self.storage_path}/mlruns/{self._experiment_id}/{self.run_id}/artifacts"
            if os.path.exists(artifacts_dir):
                return artifacts_dir
        
        # Search for the run_id in the mlruns directory structure
        mlruns_dir = f"{self.storage_path}/mlruns"
        if not os.path.exists(mlruns_dir):
            raise ValueError(f"MLruns directory not found: {mlruns_dir}")
        
        # Search through all experiment directories for this run_id
        artifacts_dir = None
        for exp_dir in glob_module.glob(f"{mlruns_dir}/*"):
            if os.path.isdir(exp_dir):
                run_dir = os.path.join(exp_dir, self.run_id)
                if os.path.exists(run_dir):
                    artifacts_dir = os.path.join(run_dir, "artifacts")
                    # Cache the experiment_id for future use
                    self._experiment_id = os.path.basename(exp_dir)
                    break
        
        if artifacts_dir is None or not os.path.exists(artifacts_dir):
            raise ValueError(f"Could not find artifacts directory for run {self.run_id}")
        
        return artifacts_dir
    
    @property
    def run_data(self):
        """Lazy load run data."""
        if self._run_data is None:
            run_data_list, exp_id, _ = self.get_runs_data(run_ids=[self.run_id], parse_params=True)
            if not run_data_list:
                raise ValueError(f"Run {self.run_id} not found.")
            self._run_data = run_data_list[0]
            self._experiment_id = exp_id
        
        return self._run_data
    
    @property
    def experiment_id(self):
        """Get experiment ID for this run."""
        if self._experiment_id is None:
            _ = self.run_data  # Trigger lazy loading
        return self._experiment_id
    
    def plot_training(self, var=None, log_scale=True, loss_step_freq=10, 
                     start_step=0, area_step_freq=100, lr_step_freq=1,
                     show_area=True, area_limits=[0.5, 2.0], show_lr=True,
                     dpi=300, step_range=None):
        """
        Plot training loss, learning rate, and posterior contour area evolution
        for this run using three vertically stacked subplots sharing the x-axis.
        
        Args:
            var (str or list): Parameter(s) from MLflow run params to include in the label.
            log_scale (bool): If True, use log scale for the y-axes.
            loss_step_freq (int): Sampling frequency for plotting loss points.
            start_step (int): Starting step offset for x-axis.
            area_step_freq (int): Sampling frequency for plotting nominal area points.
            lr_step_freq (int): Sampling frequency for plotting learning rate points.
            show_area (bool): If True, show the area subplot.
            area_limits (list): Y-axis limits for area plot.
            show_lr (bool): If True, show the learning rate subplot.
            dpi (int): DPI for saving the figure.
            step_range (tuple, optional): Tuple of (min_step, max_step) to limit the x-axis range.
            
        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects.
        """
        run_params = self.run_data['params']
        
        # Check if we should show area plot
        if show_area:
            log_nominal_area = run_params.get('log_nominal_area', False)
            if isinstance(log_nominal_area, str):
                has_area_data = log_nominal_area.lower() == 'true'
            else:
                has_area_data = bool(log_nominal_area)
            if not has_area_data:
                print(f"Warning: Run {self.run_id} does not have area data (log_nominal_area=False). Area plot will not be shown.")
                show_area = False
        
        # Fetch metrics
        try:
            loss_hist_raw = self.client.get_metric_history(self.run_id, 'loss')
            lr_hist_raw = self.client.get_metric_history(self.run_id, 'lr')
            
            # Process and filter NaNs/Infs
            loss = [(m.step + start_step, m.value) for m in loss_hist_raw if np.isfinite(m.value)]
            lr = [(m.step + start_step, m.value) for m in lr_hist_raw if np.isfinite(m.value)]
            
            if not loss:
                print(f"Warning: No valid loss points found for run {self.run_id}.")
                return None, None
            
            # Apply step range filtering if specified
            if step_range is not None:
                min_step, max_step = step_range
                loss = [(step, value) for step, value in loss if min_step <= step <= max_step]
                lr = [(step, value) for step, value in lr if min_step <= step <= max_step]
                
                if not loss:
                    print(f"Warning: No loss points found in step range {step_range} for run {self.run_id}.")
                    return None, None
            
            # Get area data if needed
            nom_area = {}
            if show_area:
                all_metrics = self.client.get_run(self.run_id).data.metrics
                area_metrics = {k: v for k, v in all_metrics.items() if k.startswith('nominal_area_avg')}
                if area_metrics:
                    for metric_name in area_metrics.keys():
                        metric_hist = self.client.get_metric_history(self.run_id, metric_name)
                        area_data = [(m.step + start_step, m.value) for m in metric_hist if np.isfinite(m.value)]
                        
                        if step_range is not None:
                            min_step, max_step = step_range
                            area_data = [(step, value) for step, value in area_data if min_step <= step <= max_step]
                        
                        nom_area[metric_name] = area_data
                        
        except Exception as e:
            print(f"Error processing metrics for run {self.run_id}: {e}.")
            return None, None
        
        # Calculate number of subplots
        num_subplots = 1  # Always show loss
        if show_area and len(nom_area) > 0:
            num_subplots += 1
        if show_lr:
            num_subplots += 1
        
        fig_height = 4 * num_subplots
        fig, axes = plt.subplots(num_subplots, 1, figsize=(14, fig_height), sharex=True)
        
        if num_subplots == 1:
            axes = np.array([axes])
        
        # Assign axes
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
        base_label = ", ".join(label_parts) if label_parts else self.run_id[:8]
        
        # Plot Loss
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
        
        # Plot Nominal Area
        if show_area and len(nom_area) > 0:
            if area_step_freq % 100 != 0:
                print("Warning: area_step_freq should ideally be a multiple of 100 as nominal_area is logged every 100 steps.")
            sampling_rate = max(1, area_step_freq // 100)
            
            area_line_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            for area_idx, (metric_name, area_data) in enumerate(nom_area.items()):
                if area_data:
                    pair_name = metric_name.replace('nominal_area_avg_', '')
                    param1, param2 = pair_name.split('_')[:2]
                    
                    area_steps, area_values = zip(*area_data)
                    sampled_indices = np.arange(0, len(area_steps), sampling_rate)
                    plot_area_steps = np.array(area_steps)[sampled_indices]
                    plot_area_values = np.array(area_values)[sampled_indices]
                    
                    line_color = area_line_colors[area_idx % len(area_line_colors)]
                    
                    try:
                        nominal_samples, target_labels, latex_labels = load_nominal_samples(
                            run_params['cosmo_exp'], run_params['cosmo_model'], dataset=run_params['dataset'])
                        with contextlib.redirect_stdout(io.StringIO()):
                            nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
                        nominal_area = get_contour_area([nominal_samples_gd], 0.68, param1, param2)[0]["nominal_area_"+pair_name]
                        ax_area.plot(plot_area_steps, plot_area_values/nominal_area, 
                                    color=line_color, label=pair_name.replace('_', ', '))
                        ax_area.axhline(1, color='black', linestyle='--', lw=1.5)
                    except NotImplementedError:
                        ax_area.plot(plot_area_steps, plot_area_values, 
                                    color=line_color, label=pair_name.replace('_', ', '))
            
            ax_area.set_ylabel("Nominal Design Area Ratio to DESI")
            ax_area.set_ylim(area_limits)
            ax_area.tick_params(axis='y')
            ax_area.legend(loc='best', title="Parameter Pair")
            ax_area.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # Plot Learning Rate
        if show_lr and lr:
            lr_steps, lr_values = zip(*lr)
            sampled_indices = np.arange(0, len(lr_steps), lr_step_freq)
            plot_lr_steps = np.array(lr_steps)[sampled_indices]
            plot_lr_values = np.array(lr_values)[sampled_indices]
            
            ax_lr.plot(plot_lr_steps, plot_lr_values, color='tab:gray', label=base_label)
        
        # Final plot configuration
        ax1.set_ylabel("Loss")
        ax1.tick_params(axis='y')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        if show_lr:
            ax_lr.set_xlabel("Training Step")
            ax_lr.set_ylabel("Learning Rate")
            ax_lr.tick_params(axis='y')
            ax_lr.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        if log_scale:
            ax1.set_yscale('log')
        
        if step_range is not None:
            min_step, max_step = step_range
            ax1.set_xlim(min_step, max_step)
        
        fig.set_constrained_layout(True)
        fig.suptitle(f"Training History - Run: {self.run_id[:8]}", fontsize=16)
        
        # Save figure
        filename = self.generate_filename("training")
        self.save_figure(fig, filename, run_id=self.run_id, experiment_id=self.experiment_id, dpi=dpi)
        
        return fig, axes
    
    def _extract_posterior_data(self, eval_step=None, device="cuda:0"):
        """Extract posterior plotting data from MLflow artifacts into a kwargs dict for generate_posterior()."""
        eig_data, artifacts_dir = self._get_eig_data(eval_step=eval_step)
        eval_step, step_str = self._resolve_step(eig_data, eval_step)

        step_data = eig_data[step_str]
        variable_data = step_data.get('variable', {})
        nominal_data = step_data.get('nominal', {})

        input_designs = np.array(eig_data.get('input_designs', []))
        if input_designs.size == 0:
            raise ValueError("No input designs found in EIG data")

        eig_values = np.array(variable_data.get('eigs_avg', []))
        if eig_values.size == 0:
            raise ValueError("No EIG values found in EIG data")

        nominal_eig = nominal_data.get('eigs_avg')
        if isinstance(nominal_eig, list):
            nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
        nominal_eig = float(nominal_eig) if nominal_eig is not None else None
        nominal_grid_eig = None
        nominal_grid_data = nominal_data.get('grid', {})
        if isinstance(nominal_grid_data, dict) and 'eigs_avg' in nominal_grid_data:
            nominal_grid_eig = nominal_grid_data.get('eigs_avg')
            if isinstance(nominal_grid_eig, list):
                nominal_grid_eig = nominal_grid_eig[0] if len(nominal_grid_eig) > 0 else None
            nominal_grid_eig = float(nominal_grid_eig) if nominal_grid_eig is not None else None

        run_obj = self.run_data['run_obj']
        run_args = self.run_data['params'].copy()

        # Extract step number for model loading
        if isinstance(eval_step, str) and eval_step.startswith('step_'):
            step_num = int(eval_step.split('_')[1])
        elif isinstance(eval_step, str):
            step_num = int(eval_step)
        else:
            step_num = eval_step

        step_for_model = 'last' if eval_step is None else step_num

        experiment = init_experiment(run_obj, run_args, device=device, design_args=None, global_rank=0)
        posterior_flow, selected_step = load_model(experiment, step_for_model, run_obj, run_args, device, global_rank=0)

        title = f"Posterior Evaluation - Run: {self.run_id[:8]}"

        filename = self.generate_filename("posterior_eval")
        save_dir = self.get_save_dir(run_id=self.run_id, experiment_id=self.experiment_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        return dict(
            experiment=experiment,
            posterior_flow=posterior_flow,
            input_designs=input_designs,
            eig_values=eig_values,
            nominal_eig=nominal_eig,
            nominal_grid_eig=nominal_grid_eig,
            title=title,
            save_path=save_path,
        )

    def generate_posterior(self, eval_step=None, display=['nominal', 'optimal'], levels=[0.68],
                          guide_samples=1000, device="cuda:0", seed=1, plot_prior=False,
                          transform_output=True, title=None, grid_samples=None):
        """Loads EIG data and model from MLflow artifacts and calls BasePlotter.generate_posterior()."""
        if isinstance(levels, (int, float)):
            levels = [levels]

        data = self._extract_posterior_data(eval_step, device=device)
        if title is not None:
            data['title'] = title

        return super().generate_posterior(
            **data,
            display=display,
            levels=levels,
            guide_samples=guide_samples,
            device=device,
            seed=seed,
            plot_prior=plot_prior,
            transform_output=transform_output,
            grid_samples=grid_samples,
        )
    
    def plot_designs(
        self,
        design_args=None,
        designs_file=None,
        mode=None,
        figsize=(12, 6),
        alpha=0.6,
        linewidth=0.8,
        marker_size=60,
        cmap='viridis',
        color_dim=0,
        labels=None,
        save_path=None,
        title=None
    ):
        """
        Plot design space using either parallel coordinates or spatial plots.
        
        When mode=None (default), automatically selects:
        - "parallel" for designs with > 4 dimensions
        - "spatial" for designs with <= 4 dimensions
        
        Args:
            design_args (dict or str, optional): Design arguments dictionary or path to YAML file.
                If None and designs_file is None, will try to load from run artifacts.
            designs_file (str, optional): Path to .npy file containing designs array.
                If None, designs will be loaded/generated from design_args or run artifacts.
            mode (str, optional): Plotting mode. "parallel" for parallel coordinates, "spatial" for spatial plots.
                If None, auto-selects based on dimensionality (parallel if n_dims > 4, else spatial).
            figsize (tuple): Figure size (default: (12, 6) for parallel, (10, 8) for spatial).
            alpha (float): Transparency (default: 0.6 for parallel, 0.9 for spatial).
            linewidth (float): Width of lines for parallel coordinates (default: 0.8).
            marker_size (int/float): Size of markers for spatial plots (default: 60).
            cmap (str): Colormap for coloring (default: 'viridis').
            color_dim (int): Dimension index to use for coloring in parallel mode (default: 0).
            labels (list/tuple, optional): Custom labels for each dimension.
            save_path (str, optional): Path to save the figure.
            title (str, optional): Figure title.
        
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        device = "cuda:0"
        experiment = None
        nominal_design = None
        optimal_design = None
        
        # Load designs
        if designs_file is not None:
            # Load from file
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
            
            # Get labels if not provided
            if labels is None:
                n_dims = designs.shape[1]
                axes_labels = [f'Dim {i+1}' for i in range(n_dims)]
            else:
                if not isinstance(labels, (list, tuple)):
                    raise ValueError("labels must be a list or tuple")
                if len(labels) != designs.shape[1]:
                    raise ValueError(f"labels must contain {designs.shape[1]} entries, received {len(labels)}.")
                axes_labels = list(labels)
        else:
            # Load/generate from design_args or run artifacts
            run_obj = self.run_data['run_obj']
            run_args = self.run_data['params']
            
            # If design_args is None, try to load from artifacts
            if design_args is None:
                artifact_uri = run_obj.info.artifact_uri
                if artifact_uri.startswith("file://"):
                    artifact_path = artifact_uri[7:]
                else:
                    artifact_path = artifact_uri
                
                design_args_artifact_path = artifact_path + "/design_args.yaml"
                if os.path.exists(design_args_artifact_path):
                    print(f"Loading design_args from run artifacts: {design_args_artifact_path}")
                    with open(design_args_artifact_path, 'r') as f:
                        design_args = yaml.safe_load(f)
                else:
                    raise FileNotFoundError(f"design_args not found in run artifacts at {design_args_artifact_path} and design_args was not provided")
            
            # Load design_args from YAML file if it's a string path
            if isinstance(design_args, str):
                if not os.path.exists(design_args):
                    raise FileNotFoundError(f"Design args file not found: {design_args}")
                with open(design_args, 'r') as f:
                    design_args = yaml.safe_load(f)
            
            # Validate design_args
            if not isinstance(design_args, dict):
                raise ValueError("design_args must be a dictionary or path to a YAML file")
            
            # Initialize experiment to get designs
            experiment = init_experiment(
                run_obj, run_args, device=device, 
                design_args=design_args, global_rank=0
            )
            
            designs = experiment.designs.cpu().numpy()
            nominal_design = experiment.nominal_design.cpu().numpy()
            exp_id = self.experiment_id
            optimal_design_path = f"{self.storage_path}/mlruns/{exp_id}/{self.run_id}/artifacts/optimal_design.npy"
            if os.path.exists(optimal_design_path):
                optimal_design = np.load(optimal_design_path)
            
            # Get labels
            if labels is not None:
                if not isinstance(labels, (tuple, list)) or len(labels) != designs.shape[1]:
                    raise ValueError(f"labels must be a tuple/list of length {designs.shape[1]}, got {labels}")
                axes_labels = list(labels)
            else:
                if hasattr(experiment, 'design_labels') and experiment.design_labels is not None:
                    axes_labels = [f'${label}$' if not label.startswith('$') else label for label in experiment.design_labels]
                elif 'labels' in design_args:
                    axes_labels = [f'${label}$' if not label.startswith('$') else label for label in design_args['labels']]
                else:
                    n_dims = designs.shape[1]
                    axes_labels = [f'$f_{i}$' for i in range(n_dims)]
        
        # Validate designs
        if designs.shape[0] == 0:
            raise ValueError("No designs found.")
        n_designs, n_dims = designs.shape
        
        # Auto-select mode if not specified
        if mode is None:
            mode = "parallel" if n_dims > 4 else "spatial"
        
        if mode not in ["parallel", "spatial"]:
            raise ValueError(f"mode must be 'parallel' or 'spatial', got '{mode}'")
        
        # Validate spatial mode dimensions
        if mode == "spatial" and (n_dims < 1 or n_dims > 4):
            raise ValueError(f"spatial mode only supports 1D to 4D designs. Found {n_dims} dimensions.")
        
        # Set figure title
        figure_title = title or "Design Space"
        
        if mode == "parallel":
            # Parallel coordinates mode
            # Validate color_dim
            if not isinstance(color_dim, int) or color_dim not in range(n_dims):
                raise ValueError(f"color_dim must be an integer 0-{n_dims-1}, got {color_dim}")
            
            # Extract dimensions
            dim_data = {i: designs[:, i] for i in range(n_dims)}
            color_values = dim_data[color_dim]
            
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
            
            if nominal_design is not None:
                ax.plot(x_positions, nominal_design, color='black', alpha=1.0, linewidth=2, label='Nominal Design', zorder=3)
            
            if optimal_design is not None:
                ax.plot(x_positions, optimal_design, color='red', alpha=1.0, linewidth=2, linestyle='--', label='Optimal Design', zorder=3)
            
            # Set axis properties
            ax.set_xticks(x_positions)
            ax.set_xticklabels(axes_labels, fontsize=12)
            ax.set_ylabel('Tracer Fraction', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')

            if nominal_design is not None or optimal_design is not None:
                ax.legend()
            
            # Set y-axis limits to show full range
            all_values = np.concatenate([dim_data[j] for j in range(n_dims)])
            y_min = all_values.min() * 0.5
            y_max = all_values.max() * 1.1
            ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            
        else:
            # Spatial mode
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
                # 3D or 4D
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
            # If save_path is provided, extract filename and use it
            filename = os.path.basename(save_path)
            self.save_figure(fig, filename, run_id=self.run_id, experiment_id=self.experiment_id, dpi=300, close_fig=False, display_fig=False)
        else:
            # Auto-save with generated filename
            filename = self.generate_filename(f"designs_{mode}")
            self.save_figure(fig, filename, run_id=self.run_id, experiment_id=self.experiment_id, dpi=300, close_fig=False, display_fig=False)
        
        return fig
    
    def _get_eig_data(self, eval_step=None):
        """
        Helper method to load EIG data from artifacts directory.
        Finds the artifacts directory by searching for the run_id without needing experiment_id.
        
        Args:
            eval_step (str or int, optional): Step to load data for.
            
        Returns:
            tuple: (eig_data_dict, artifacts_dir)
        """
        if self._experiment_id is not None:
            artifacts_dir = f"{self.storage_path}/mlruns/{self._experiment_id}/{self.run_id}/artifacts"
        else:
            # Search for the run_id in the mlruns directory structure
            mlruns_dir = f"{self.storage_path}/mlruns"
            if not os.path.exists(mlruns_dir):
                raise ValueError(f"MLruns directory not found: {mlruns_dir}")
            
            # Search through all experiment directories for this run_id
            artifacts_dir = None
            for exp_dir in glob_module.glob(f"{mlruns_dir}/*"):
                if os.path.isdir(exp_dir):
                    run_dir = os.path.join(exp_dir, self.run_id)
                    if os.path.exists(run_dir):
                        artifacts_dir = os.path.join(run_dir, "artifacts")
                        # Cache the experiment_id for future use
                        self._experiment_id = os.path.basename(exp_dir)
                        break
            
            if artifacts_dir is None or not os.path.exists(artifacts_dir):
                raise ValueError(f"Could not find artifacts directory for run {self.run_id}")
        
        json_path, eig_data = self.load_eig_data_file(artifacts_dir, eval_step=eval_step)
        return eig_data, artifacts_dir
    
    def _init_experiment_for_plotting(self):
        """
        Helper method to initialize experiment for plotting (needed for design labels, nominal design, etc.).
        
        Returns:
            experiment object
        """
        device = "cuda:0"
        run_obj = self.run_data['run_obj']
        run_args = self.run_data['params']
        experiment = init_experiment(
            run_obj, run_args, device=device, 
            design_args=None, global_rank=0
        )
        return experiment
    
    def design_comparison(self, eval_step=None, width=0.2, log_scale=True, use_fractional=False):
        """
        Plots a bar chart comparing the nominal and optimal design.
        
        Args:
            eval_step (str or int, optional): Step to evaluate. If None, uses most recent.
            width (float): Width of the bars in the bar chart (default: 0.2)
            log_scale (bool): Whether to use log scale for y-axis (default: True)
            use_fractional (bool): Whether to plot fractional values or absolute quantities (default: False)
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        print(f"Generating design comparison plot...")
        
        # Load EIG data
        eig_data, artifacts_dir = self._get_eig_data(eval_step=eval_step)
        
        # Get step data
        if eval_step is None:
            # Find the most recent step
            step_keys = [k for k in eig_data.keys() if k.startswith('step_')]
            if not step_keys:
                raise ValueError("No step data found in EIG data file")
            eval_step = max(step_keys, key=lambda x: int(x.split('_')[1]))
        
        step_str = str(eval_step) if not isinstance(eval_step, str) or not eval_step.startswith('step_') else eval_step
        if not step_str.startswith('step_'):
            step_str = f"step_{step_str}"
        
        if step_str not in eig_data:
            raise ValueError(f"Step {eval_step} not found in EIG data")
        
        step_data = eig_data[step_str]
        variable_data = step_data.get('variable', {})
        nominal_data = step_data.get('nominal', {})
        
        # Get designs and EIGs
        input_designs = np.array(eig_data.get('input_designs', []))
        if input_designs.size == 0:
            raise ValueError("No input designs found in EIG data")
        
        eig_values = np.array(variable_data.get('eigs_avg', []))
        if eig_values.size == 0:
            raise ValueError("No EIG values found in EIG data")
        
        # Get optimal design
        optimal_idx = np.argmax(eig_values)
        optimal_design = input_designs[optimal_idx]
        optimal_eig = float(eig_values[optimal_idx])
        
        # Get nominal EIG
        nominal_eig = nominal_data.get('eigs_avg')
        if isinstance(nominal_eig, list):
            nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
        nominal_eig = float(nominal_eig) if nominal_eig is not None else None
        
        # Initialize experiment to get design labels and nominal design
        experiment = self._init_experiment_for_plotting()
        nominal_design = experiment.nominal_design.cpu().numpy()
        nominal_total_obs = experiment.nominal_total_obs
        design_labels = experiment.design_labels
        
        # Determine whether to use fractional or absolute values
        if use_fractional:
            nominal_design_plot = nominal_design
            design_plot = optimal_design
            max_tracers = np.array([experiment.num_targets[target] for target in design_labels])
            max_tracers = max_tracers / nominal_total_obs
            ylabel = 'Fraction of Total Tracers'
        else:
            nominal_design_plot = nominal_design * nominal_total_obs
            design_plot = optimal_design * nominal_total_obs
            max_tracers = np.array([experiment.num_targets[target] for target in design_labels])
            ylabel = 'Number of Tracers'
        
        # Set the positions for the bars
        x = np.arange(len(design_labels))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        for i, max_val in enumerate(max_tracers):
            rect = Rectangle((x[i] - width, 0), width*2, max_val,
                           facecolor='gray', alpha=0.3, edgecolor='none')
            ax.add_patch(rect)
            x_left = x[i] - width
            x_right = x[i] + width
            y_top = max_val
            ax.plot([x_left, x_right], [y_top, y_top], 'k:', linewidth=1.5, alpha=0.7)
            ax.plot([x_left, x_left], [0, y_top], 'k:', linewidth=1.5, alpha=0.7)
            ax.plot([x_right, x_right], [0, y_top], 'k:', linewidth=1.5, alpha=0.7)
        
        # Create labels with EIG values
        nominal_label = 'Nominal Design'
        if nominal_eig is not None:
            nominal_label += f', EIG: {nominal_eig:.3f} bits'
        optimal_label = 'Optimal Design'
        if optimal_eig is not None:
            optimal_label += f', EIG: {optimal_eig:.3f} bits'
        
        bars1 = ax.bar(x - width/2, nominal_design_plot, width, label=nominal_label, color='tab:blue')
        bars2 = ax.bar(x + width/2, design_plot, width, label=optimal_label, color='tab:orange')
        
        max_label = 'Max Possible Tracers' if not use_fractional else 'Max Possible Fraction'
        ax.plot([], [], 'k:', linewidth=1.5, alpha=0.7, label=max_label)
        
        ax.set_xlabel('Tracer Class', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(design_labels)
        ax.legend(fontsize=14)
        if log_scale:
            ax.set_yscale('log')

        title = f"Design Comparison - Run: {self.run_id[:8]}"
        plt.suptitle(title, fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Save figure automatically
        filename = self.generate_filename("design_comparison")
        self.save_figure(fig, filename, run_id=self.run_id, experiment_id=self.experiment_id, dpi=400, close_fig=False, display_fig=False)
        
        return fig
    
    def posterior_steps(self, steps, levels=[0.68], eval_step=None):
        """
        Plots posterior distributions at different training steps for a single run.
        
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'loss_best' as special values.
            levels (float or list): Contour level(s) to plot (default: [0.68]).
            eval_step (str or int, optional): If provided, used to load EIG data for reference.
            
        Returns:
            GetDist plotter object.
        """
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        print(f"Running posterior steps evaluation...")
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(steps)))
        
        all_samples = []
        all_colors = []
        custom_legend = []
        
        device = "cuda:0"
        run_obj = self.run_data['run_obj']
        run_args = self.run_data['params']
        
        # Initialize experiment once
        experiment = init_experiment(run_obj, run_args, device=device, design_args=None, global_rank=0)
        
        for i, step in enumerate(steps):
            # Load model for this step
            posterior_flow, _ = load_model(experiment, step, run_obj, run_args, device, global_rank=0)
            auto_seed(1)  # Use consistent seed
            
            # Use experiment.get_guide_samples directly (returns MCSamples object)
            nominal_context = experiment.nominal_context
            samples_gd = experiment.get_guide_samples(
                posterior_flow, nominal_context, 
                num_samples=1000,
                transform_output=None
            )
            
            all_samples.append(samples_gd)
            color_hex = matplotlib.colors.to_hex(colors[i % len(colors)])
            all_colors.append(color_hex)
            
            if step == 'last':
                step_label = run_args.get("total_steps", "Last")
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            custom_legend.append(
                Line2D([0], [0], color=color_hex, 
                        label=f'Step {step_label}', linewidth=1.2)
            )
        
        # Get nominal samples using reference experiment (already initialized above)
        try:
            nominal_samples_gd = experiment.get_nominal_samples(transform_output=None)
            all_samples.append(nominal_samples_gd)
            all_colors.append('black')
        except NotImplementedError:
            print(f"Warning: get_nominal_samples not implemented for {self.cosmo_exp}, skipping nominal design plot.")
        
        plot_width = 12
        g = self.plot_posterior(all_samples, all_colors, levels=levels, width_inch=plot_width)
        
        # Calculate dynamic font sizes
        n_params = len(all_samples[0].paramNames.names)
        base_fontsize = max(6, min(18, plot_width * (0.2 + 0.42 * np.sqrt(n_params))))
        title_fontsize = base_fontsize * 1.15
        legend_fontsize = base_fontsize * 0.80

        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        custom_legend.append(
            Line2D([0], [0], color='black', label=f'DESI', linewidth=1.2)
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), fontsize=legend_fontsize)
        leg.set_in_layout(False)
        title = f"Posterior Steps - Run: {self.run_id[:8]}"
        g.fig.suptitle(title, fontsize=title_fontsize, weight='bold')
        
        # Save figure automatically
        filename = self.generate_filename("posterior_steps")
        self.save_figure(g.fig, filename, run_id=self.run_id, experiment_id=self.experiment_id, dpi=400, close_fig=False, display_fig=False)
        
        return g
    
    def _resolve_step(self, eig_data, eval_step):
        """Resolve eval_step to a step string key in eig_data."""
        if eval_step is None:
            step_keys = [k for k in eig_data.keys() if k.startswith('step_')]
            if not step_keys:
                raise ValueError("No step data found in EIG data file")
            eval_step = max(step_keys, key=lambda x: int(x.split('_')[1]))

        step_str = str(eval_step) if not isinstance(eval_step, str) or not eval_step.startswith('step_') else eval_step
        if not step_str.startswith('step_'):
            step_str = f"step_{step_str}"

        if step_str not in eig_data:
            raise ValueError(f"Step {eval_step} not found in EIG data")

        return eval_step, step_str

    def _extract_eig_data(self, eval_step=None, include_nominal=True):
        """Extract EIG plotting data from MLflow artifacts into a kwargs dict for eig_designs()."""
        eig_data, artifacts_dir = self._get_eig_data(eval_step=eval_step)
        eval_step, step_str = self._resolve_step(eig_data, eval_step)

        step_data = eig_data[step_str]
        variable_data = step_data.get('variable', {})
        nominal_data = step_data.get('nominal', {})

        input_designs = np.array(eig_data.get('input_designs', []))
        if input_designs.size == 0:
            raise ValueError("No input designs found in EIG data")

        eig_values = np.array(variable_data.get('eigs_avg', []))
        eig_std_values = np.array(variable_data.get('eigs_std', [])) if 'eigs_std' in variable_data else np.zeros_like(eig_values)
        grid_data = variable_data.get('grid', {})
        grid_eig_values = np.array(grid_data.get('eigs_avg', [])) if grid_data else None
        grid_eig_std_values = np.array(grid_data.get('eigs_std', [])) if grid_data else None

        nominal_eig = None
        nominal_grid_eig = None
        if include_nominal:
            nominal_eig = nominal_data.get('eigs_avg')
            if isinstance(nominal_eig, list):
                nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
            nominal_eig = float(nominal_eig) if nominal_eig is not None else None
            nominal_grid_data = nominal_data.get('grid', {})
            if isinstance(nominal_grid_data, dict) and 'eigs_avg' in nominal_grid_data:
                nominal_grid_eig = nominal_grid_data.get('eigs_avg')
                if isinstance(nominal_grid_eig, list):
                    nominal_grid_eig = nominal_grid_eig[0] if len(nominal_grid_eig) > 0 else None
                nominal_grid_eig = float(nominal_grid_eig) if nominal_grid_eig is not None else None

        experiment = self._init_experiment_for_plotting()

        sort_title = "Sorted"
        title = f'{sort_title} EIG per Design - Run: {self.run_id[:8]}'

        filename = self.generate_filename("eig_designs")
        save_dir = self.get_save_dir(run_id=self.run_id, experiment_id=self.experiment_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        return dict(
            eig_values=eig_values,
            input_designs=input_designs,
            design_labels=experiment.design_labels,
            nominal_design=experiment.nominal_design.cpu().numpy(),
            eig_std_values=eig_std_values,
            nominal_eig=nominal_eig,
            grid_eig_values=grid_eig_values,
            grid_eig_std_values=grid_eig_std_values,
            nominal_grid_eig=nominal_grid_eig,
            include_nominal=include_nominal,
            title=title,
            save_path=save_path,
        ), eval_step, eig_data

    def eig_designs(self, eval_step=None, sort=True, sort_step=None, include_nominal=True):
        """Loads EIG data from MLflow artifacts and calls BasePlotter.eig_designs()."""
        data, _, _ = self._extract_eig_data(eval_step, include_nominal)
        return super().eig_designs(**data, sort=sort, color="black")


# ============================================================================
# Comparison Plotter Class
# ============================================================================

class ComparisonPlotter(BasePlotter):
    """
    Plotter class for generating comparison plots across multiple MLflow runs.
    Handles posterior comparisons, training comparisons, EIG comparisons, etc.
    """
    
    def __init__(self, cosmo_exp='num_tracers', mlflow_exp=None, run_ids=None, 
                 excluded_runs=None, filter_string=None, run_labels=None):
        """
        Initialize the comparison plotter.
        
        Args:
            cosmo_exp (str): Cosmology experiment folder name.
            mlflow_exp (str, optional): Name of the MLflow experiment. If provided, all runs in this experiment will be used.
            run_ids (list, optional): List of specific run IDs to compare. If provided, mlflow_exp is ignored.
            excluded_runs (list, optional): List of run IDs to exclude. Especially useful when mlflow_exp is passed to skip specific runs across all comparison methods.
            filter_string (str, optional): MLflow filter string.
            run_labels (list, optional): Labels for each run (same order as run_ids / run_data_list). Used by compare_optimal_designs and compare_eigs.
        """
        super().__init__(cosmo_exp=cosmo_exp)
        
        if mlflow_exp is None and run_ids is None:
            raise ValueError("Either mlflow_exp or run_ids must be provided")
        if mlflow_exp is not None and run_ids is not None:
            raise ValueError("Cannot provide both mlflow_exp and run_ids. Use run_ids to override mlflow_exp.")
        
        self.mlflow_exp = mlflow_exp
        self.run_ids = run_ids if isinstance(run_ids, list) else [run_ids] if run_ids is not None else None
        self.excluded_runs = excluded_runs or []
        self.filter_string = filter_string
        self.run_labels = run_labels
        
        # Cache for run data (will be populated on first use)
        self._run_data_cache = {}
    
    def _get_run_data_list(self, parse_params=True, filter_string=None):
        """
        Get run data list, using cached values if available.
        
        Args:
            parse_params (bool): Whether to parse MLflow parameters.
            filter_string (str, optional): Override filter_string from __init__.
            
        Returns:
            tuple: (run_data_list, experiment_id, actual_mlflow_exp)
        """
        filter_str = filter_string if filter_string is not None else self.filter_string
        excluded = self.excluded_runs
        cache_key = (parse_params, tuple(excluded) if excluded else (), filter_str)
        
        if cache_key not in self._run_data_cache:
            run_data_list, experiment_id, actual_mlflow_exp = self.get_runs_data(
                mlflow_exp=self.mlflow_exp,
                run_ids=self.run_ids,
                excluded_runs=excluded,
                filter_string=filter_str,
                parse_params=parse_params
            )
            self._run_data_cache[cache_key] = (run_data_list, experiment_id, actual_mlflow_exp)
        
        return self._run_data_cache[cache_key]
    
    def _get_run_id_to_exp_id(self, run_data_list):
        """
        Build a mapping from run_id to exp_id from run_data_list.
        
        Args:
            run_data_list (list): List of run data dictionaries.
            
        Returns:
            dict: Mapping from run_id to exp_id.
        """
        return {run_data['run_id']: run_data['exp_id'] for run_data in run_data_list}
    
    def _resolve_run_labels(self, run_data_list, run_ids, var=None):
        """
        Resolve run labels using self.run_labels, var parameters, or run_id.
        
        Priority: self.run_labels (if available) + var parameters (if specified) > var parameters > run_id[:8]
        When var is specified, it is appended to the base label (from self.run_labels or run_id).
        
        Args:
            run_data_list (list): List of run data dictionaries (full list from _get_run_data_list).
            run_ids (list): List of run IDs to resolve labels for (may be a subset of run_data_list).
            var (str or list, optional): Parameter(s) from MLflow run params to include in the label.
        
        Returns:
            list: List of labels corresponding to run_ids.
        """
        # Convert run_ids to list if it's a single value
        if not isinstance(run_ids, list):
            run_ids = [run_ids]
        
        # Get base labels from self.run_labels if available, otherwise use run_id[:8]
        if self.run_labels is not None and len(self.run_labels) == len(run_data_list):
            rid2lab = {r['run_id']: self.run_labels[i] for i, r in enumerate(run_data_list)}
            base_labels = [rid2lab.get(rid, rid[:8]) for rid in run_ids]
        else:
            base_labels = [rid[:8] for rid in run_ids]
        
        # If var is specified, append var parameters to base labels
        if var is not None:
            vars_list = var if isinstance(var, list) else [var]
            run_labels = []
            for run_id, base_label in zip(run_ids, base_labels):
                run_data_item = next((r for r in run_data_list if r['run_id'] == run_id), None)
                if run_data_item and 'params' in run_data_item:
                    run_params = run_data_item['params']
                    var_parts = [f"{v_key}={run_params[v_key]}" for v_key in vars_list if v_key in run_params]
                    if var_parts:
                        # Append var parameters to base label
                        run_labels.append(f"{base_label}, {', '.join(var_parts)}")
                    else:
                        run_labels.append(base_label)
                else:
                    run_labels.append(base_label)
            return run_labels
        else:
            return base_labels
    
    def _set_legend(self, ax, num_runs, title=None, loc='best', framealpha=0.9, 
                    handles=None, labels=None, fontsize=None):
        """
        Set legend for a plot with appropriate font size based on number of runs.
        
        Args:
            ax: Matplotlib axis object.
            num_runs (int): Number of runs/entries in the legend.
            title (str, optional): Legend title.
            loc (str): Legend location. Default 'best'.
            framealpha (float): Legend frame alpha. Default 0.9.
            handles (list, optional): Explicit legend handles. If None, uses existing handles from ax.
            labels (list, optional): Explicit legend labels. Must be provided if handles is provided.
            fontsize (str or int, optional): Explicit font size. If None, determined automatically based on num_runs.
        
        Returns:
            Legend object.
        """
        # Determine font size if not explicitly provided
        if fontsize is None:
            if num_runs > 10:
                fontsize = 'xx-small'
            elif num_runs > 6:
                fontsize = 'x-small'
            elif num_runs > 4:
                fontsize = 'small'
            else:
                fontsize = 'medium'
        
        # Build legend kwargs
        legend_kwargs = {
            'loc': loc,
            'fontsize': fontsize,
            'framealpha': framealpha
        }
        
        if title is not None:
            legend_kwargs['title'] = title
        
        # Use explicit handles/labels if provided, otherwise let matplotlib auto-detect
        if handles is not None and labels is not None:
            return ax.legend(handles, labels, **legend_kwargs)
        else:
            return ax.legend(**legend_kwargs)
    
    def compare_posterior(self, var=None, guide_samples=10000, show_scatter=False,
                         step='loss_best', seed=1, device="cuda:0",
                         global_rank=0, dpi=300, levels=[0.68, 0.95], width_inch=10,
                         colors=None, filter_string=None):
        """
        Compare posterior distributions across multiple runs in a triangle plot.
        
        Args:
            var (str or list, optional): Parameter(s) to group runs by.
            guide_samples (int): Number of samples to draw from the posterior.
            show_scatter (bool): Whether to show scatter points.
            step (str or int): Checkpoint to evaluate.
            seed (int): Random seed.
            device (str): Device to use.
            global_rank (int or list): Global rank(s) to evaluate.
            dpi (int): DPI for saving the figure.
            levels (list): List of contour levels to plot.
            width_inch (float): Width of the triangle plot in inches.
            colors (list, optional): List of colors to use for each group.
            filter_string (str, optional): Override filter_string from __init__.
            
        Returns:
            GetDist plotter object.
        """
        global_ranks = global_rank if isinstance(global_rank, list) else [global_rank]
        
        if not isinstance(levels, list):
            levels = [levels]
        
        run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = self._get_run_data_list(
            parse_params=True,
            filter_string=filter_string
        )
        if not run_data_list:
            return None
        
        if 'params' in run_data_list[0] and 'cosmo_model' in run_data_list[0]['params']:
            cosmo_model_for_desi = run_data_list[0]['params']['cosmo_model']
        else:
            raise ValueError("Could not determine cosmo_model from the first run for DESI plot.")
        
        # Create groups based on var parameter
        vars_list = var if isinstance(var, list) else [var] if var is not None else []
        
        grouped_runs = {}
        for run_data_item in run_data_list:
            if var:
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
                    continue
            else:
                group_key = run_data_item['run_id']
            
            if group_key not in grouped_runs:
                grouped_runs[group_key] = []
            grouped_runs[group_key].append(run_data_item)
        
        if not grouped_runs:
            print("No valid groups found. Cannot plot.")
            return None
        
        # Create run_id to label mapping if self.run_labels is available
        run_id_to_label = {}
        if self.run_labels is not None and len(self.run_labels) == len(run_data_list):
            run_id_to_label = {r['run_id']: self.run_labels[i] for i, r in enumerate(run_data_list)}
        
        # Sort groups
        if var:
            def custom_sort_key(group_key):
                base_sort_key = sort_key_for_group_tuple(group_key)
                if isinstance(base_sort_key, tuple):
                    modified_values = []
                    for val in base_sort_key:
                        if isinstance(val, (int, float)):
                            modified_values.append(-val)
                        else:
                            modified_values.append(val)
                    return tuple(modified_values)
                else:
                    if isinstance(base_sort_key, (int, float)):
                        return -base_sort_key
                    else:
                        return base_sort_key
            
            sorted_group_keys = sorted(grouped_runs.keys(), key=custom_sort_key)
        else:
            sorted_group_keys = sorted(grouped_runs.keys())
        
        # Collect samples
        all_samples = []
        all_colors = []
        legend_handles = []
        
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
            
            group_samples = []
            
            for run_data_item in group_runs:
                for rank in global_ranks:
                    samples_obj, selected_step = get_nominal_samples(
                        run_data_item['run_obj'], 
                        run_data_item['params'], 
                        guide_samples=guide_samples,
                        seed=seed,
                        device=device,
                        step=step, 
                        cosmo_exp=self.cosmo_exp, 
                        global_rank=rank
                    )
                    if samples_obj is not None:
                        group_samples.append(samples_obj)
            
            if not group_samples:
                print(f"Warning: No valid samples for group {group_key}. Skipping.")
                continue
            
            all_samples.extend(group_samples)
            all_colors.extend([group_color] * len(group_samples))
            
            # Determine group label: combine run_labels and var if both are available
            label_parts = []
            
            # Get run_label if available
            if run_id_to_label:
                # Get labels for all runs in this group
                group_run_labels = [run_id_to_label.get(run_data_item['run_id'], None) 
                                   for run_data_item in group_runs]
                # Filter out None values
                group_run_labels = [label for label in group_run_labels if label is not None]
                
                if group_run_labels:
                    # If all runs in group have the same label, use it; otherwise use first label
                    if len(set(group_run_labels)) == 1:
                        label_parts.append(group_run_labels[0])
                    else:
                        # Multiple different labels in group - use first one
                        label_parts.append(group_run_labels[0])
            
            # Add var information if specified
            if var:
                var_label = ', '.join([f'{vars_list[j]}={val}' for j, val in enumerate(group_key)])
                label_parts.append(var_label)
            
            # Combine label parts, or fall back to run_id if nothing available
            if label_parts:
                group_label = ', '.join(label_parts)
            else:
                group_label = group_key[:8] if isinstance(group_key, str) else str(group_key)[:8]
            
            legend_handles.append(
                Line2D([0], [0], color=group_color, label=group_label)
            )
        
        if not all_samples:
            print("No samples generated for any group. Cannot plot.")
            return None

        nominal_samples, target_labels, latex_labels = load_nominal_samples(
            self.cosmo_exp, cosmo_model_for_desi, dataset=run_data_list[0]['params']['dataset'])
        with contextlib.redirect_stdout(io.StringIO()):
            nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
        nominal_label = f'Nominal ({cosmo_model_for_desi})'
        
        all_samples.append(nominal_samples_gd)
        all_colors.append('black')
        
        legend_handles.append(
            Line2D([0], [0], color='black', label=nominal_label)
        )
        
        g = self.plot_posterior(all_samples, all_colors, show_scatter=show_scatter, levels=levels, width_inch=width_inch)

        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        
        title = f'Posterior Comparison, Step: {step}'
        filter_str = filter_string if filter_string is not None else self.filter_string
        if filter_str:
            title += f' (filter: {filter_str})'
        
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        g.fig.suptitle(title)
        
        # Save figure
        filename_prefix = "posterior_comp"
        single_run_id_for_path = None
        if self.mlflow_exp and experiment_id_for_save_path:
            save_dir = self.get_save_dir(experiment_id=experiment_id_for_save_path)
        elif not self.mlflow_exp and self.run_ids and len(run_data_list) == 1 and experiment_id_for_save_path:
            single_run_id_for_path = run_data_list[0]['run_id']
            save_dir = self.get_save_dir(run_id=single_run_id_for_path, experiment_id=experiment_id_for_save_path)
            filename_prefix = f"posterior_step_{step}"
        else:
            save_dir = self.get_save_dir()
        
        filename = self.generate_filename(filename_prefix)
        save_path = self.save_figure(g.fig, filename, run_id=single_run_id_for_path, 
                                     experiment_id=experiment_id_for_save_path, dpi=dpi)
        
        if hasattr(g, 'fig') and g.fig in plt.get_fignums():
            plt.close(g.fig)
        
        return g
    
    def compare_eigs(
        self,
        var=None,
        eval_step=None,
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
        design_labels=None,
        show_ratio_to_nominal=True
    ):
        """
        Compare EIG values across multiple runs.

        Args:
            var (str or list, optional): Parameter(s) from MLflow run params to include in the label.
                                        If provided and self.run_labels is None, labels will be generated from these parameters. Otherwise run ID (first 8 chars).
            eval_step (str or int, optional): Step identifier (if omitted the most recent step is used).
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
            show_ratio_to_nominal (bool): If True, display designs as ratio to nominal design (like eig_designs). Default True.
        
        Returns:
            tuple: (fig, (ax_line, ax_heat)) matplotlib figure and axes objects.
        """
        storage_path = self.storage_path

        # Parse params if we need them for var labels or plot_input_design
        need_params = plot_input_design or (var is not None and self.run_labels is None)
        run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = self._get_run_data_list(
            parse_params=need_params
        )
        if not run_data_list:
            raise ValueError(f"No runs found in experiment {self.cosmo_exp}")

        run_ids = [run_data['run_id'] for run_data in run_data_list]
        run_id_to_exp_id = self._get_run_id_to_exp_id(run_data_list)

        all_data = []
        found_run_ids = []

        for run_id in run_ids:
            exp_id = run_id_to_exp_id.get(run_id)
            if exp_id is None:
                print(f"Warning: Run {run_id} not found in experiment {self.cosmo_exp}, skipping...")
                continue

            artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
            
            try:
                json_path, data = self.load_eig_data_file(artifacts_dir, eval_step=eval_step)
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

        # Resolve run_labels using centralized method
        run_labels = self._resolve_run_labels(run_data_list, found_run_ids, var=var)
        if len(run_labels) != len(found_run_ids):
            raise ValueError("run_labels must match the number of discovered runs")

        step_numbers = []
        for data, run_label in zip(all_data, run_labels):
            if eval_step is not None:
                selected_step = int(eval_step) if isinstance(eval_step, str) else eval_step
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

        # Check if designs are 1D or multi-dimensional
        is_1d_design = (num_dims == 1)

        reference_record = None
        global_sort_idx = None
        per_run_sort = False  # Flag to indicate per-run sorting (no sort_reference)

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
                # Sort each run independently from highest to lowest
                per_run_sort = True
                # Store per-run sort indices in each record
                for record in run_records:
                    record['sort_idx'] = np.argsort(record['eigs_avg'])[::-1]
        else:
            global_sort_idx = np.arange(num_designs)

        # Get sorted designs for heatmap (only if not per-run sorting)
        if per_run_sort:
            sorted_designs = None  # No heatmap when sorting per-run
            run_for_nominal = run_records[0]  # Use first run for nominal design
        elif reference_record is not None:
            sorted_designs = reference_record['designs'][global_sort_idx]
            run_for_nominal = reference_record
        else:
            sorted_designs = run_records[0]['designs'][global_sort_idx]
            run_for_nominal = run_records[0]
        
        # Get nominal design if needed for ratio display
        nominal_design = None
        if show_ratio_to_nominal and not is_1d_design:
            # Initialize experiment from the reference run to get nominal_design
            try:
                device = "cuda:0"
                run_data_item = next((r for r in run_data_list if r['run_id'] == run_for_nominal['run_id']), None)
                if run_data_item and run_data_item.get('run_obj') is not None:
                    run_params = run_data_item['params'].copy()
                    
                    # init_experiment is imported via 'from util import *' at the top
                    # init_experiment will automatically load prior_args from artifacts
                    experiment = init_experiment(
                        run_data_item['run_obj'],
                        run_params,
                        device=device,
                        design_args=None,
                        global_rank=0
                    )
                    if hasattr(experiment, 'nominal_design'):
                        nominal_design = experiment.nominal_design.cpu().numpy()
                        # Verify shape matches (nominal_design should be 1D with num_dims elements)
                        if nominal_design.ndim == 1:
                            if nominal_design.shape[0] != sorted_designs.shape[1]:
                                print(f"Warning: nominal_design shape {nominal_design.shape} doesn't match design dimensions {sorted_designs.shape[1]}, disabling ratio display.")
                                nominal_design = None
                        else:
                            print(f"Warning: nominal_design has unexpected shape {nominal_design.shape}, expected 1D array, disabling ratio display.")
                            nominal_design = None
            except Exception as e:
                print(f"Warning: Could not initialize experiment to get nominal_design: {e}")
                print("Will display designs as absolute values instead of ratios.")

        # Determine if we should show heatmap (only if not per-run sorting)
        show_heatmap = not per_run_sort and (not is_1d_design or sort)
        
        if is_1d_design and not sort and not per_run_sort:
            fig, ax_line = plt.subplots(figsize=figsize)
            ax_heat = None
            cbar_ax = None
            x_vals = sorted_designs[:, 0]
            x_label = design_labels[0]
        elif per_run_sort:
            # Per-run sorting: no heatmap, just line plot
            fig, ax_line = plt.subplots(figsize=figsize)
            ax_heat = None
            cbar_ax = None
            x_vals = np.arange(num_designs)
            x_label = "Design Index (sorted per run, highest to lowest)"
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
            fig.subplots_adjust(left=0.06, right=0.88)
            bbox0 = ax_line.get_position()
            bbox1 = ax_heat.get_position()
            cbar_width = 0.02
            cbar_gap = 0.01
            cbar_left = bbox1.x1 + cbar_gap
            cbar_bottom = bbox1.y0
            cbar_height = bbox0.y1 - bbox1.y0
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            plt.setp(ax_line.get_xticklabels(), visible=False)
            ax_line.tick_params(axis='x', which='both', length=0)
            
            # Set x_vals and x_label for sorted/multi-dimensional case
            x_vals = np.arange(num_designs)
            if sort_reference is not None:
                x_label_suffix = f" (sorted by {reference_record['run_label']})"
            elif sort:
                # Use first run as reference when sort_reference is not specified
                x_label_suffix = f" (sorted by {run_records[0]['run_label']})"
            else:
                x_label_suffix = ""
            x_label = f"Design Index{x_label_suffix}"

        nominal_line_drawn = False

        ax_line.margins(x=0)

        handles_for_legend = []
        nominal_handle = None

        for record in run_records:
            # Use per-run sort_idx if per_run_sort, otherwise use global_sort_idx
            if per_run_sort:
                sort_idx = record['sort_idx']
            else:
                sort_idx = global_sort_idx

            eig_vals = record['eigs_avg'][sort_idx]
            eig_std_vals = record['eigs_std'][sort_idx]
            color = record['color']

            baseline = 0.0
            if normalize:
                if record['nominal_eig'] is None:
                    raise ValueError(f"normalize=True requires nominal_eig for run {record['run_id']}")
                baseline = record['nominal_eig']

            eig_vals_plot = eig_vals - baseline

            # Determine if single point or line
            num_points = len(eig_vals_plot)
            if num_points == 1:
                line = ax_line.scatter(x_vals, eig_vals_plot, color=color, zorder=5, label=record['run_label'])
            else:
                line = ax_line.plot(x_vals, eig_vals_plot, color=color, linewidth=2, label=record['run_label'])[0]
                
            handles_for_legend.append(line)
            if show_errorbars and np.any(eig_std_vals > 0):
                if num_points == 1:
                    ax_line.errorbar(x_vals, eig_vals_plot, yerr=eig_std_vals, color=color, zorder=5, fmt='o', capsize=5, label=record['run_label'])
                else:
                    ax_line.fill_between(
                        x_vals,
                        eig_vals_plot - eig_std_vals,
                        eig_vals_plot + eig_std_vals,
                        color=color,
                        alpha=0.2
                    )

            if normalize:
                # Always show horizontal line at y=0 when normalized (represents nominal baseline)
                if not nominal_line_drawn:
                    nominal_handle = ax_line.axhline(0, linestyle='--', color='black', linewidth=1.2, label='Nominal (all runs)')
                    nominal_line_drawn = True
            elif show_nominal:
                # When not normalized, show nominal lines per run if requested
                if record['nominal_eig'] is not None:
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
        self._set_legend(ax_line, len(legend_labels), handles=legend_handles, labels=legend_labels)
        
        # Plot sorted designs (heatmap) if ax_heat exists and we're not doing per-run sorting
        if ax_heat is not None and not per_run_sort:
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
                # Format label: use as-is if already has $, otherwise wrap with $
                label0 = design_labels[0] if design_labels[0].startswith('$') else f'${design_labels[0]}$'
                ax_heat.set_yticklabels([label0])
            else:
                # Multi-dimensional designs - use ratio to nominal
                if nominal_design is None or np.any(nominal_design == 0):
                    plot_data = sorted_designs.T
                    cmap = 'viridis'
                    im = ax_heat.imshow(plot_data, aspect='auto', cmap=cmap, origin='upper')
                else:
                    plot_data = (sorted_designs / nominal_design[np.newaxis, :]).T
                    cmap = 'RdBu'
                    from matplotlib.colors import TwoSlopeNorm
                    vmin = plot_data.min()
                    vmax = plot_data.max()
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                    im = ax_heat.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm, origin='upper')
                
                ax_heat.set_xlim(-0.5, num_designs - 0.5)
                ax_line.set_xlim(-0.5, num_designs - 0.5)
                ax_heat.set_yticks(np.arange(num_dims))
                # Format labels: use as-is if already has $, otherwise wrap with $
                formatted_labels = [label if label.startswith('$') else f'${label}$' for label in design_labels]
                ax_heat.set_yticklabels(formatted_labels)

            ax_heat.set_xlabel(x_label, fontsize=12, weight='bold')
            ax_heat.set_ylabel('')
            ax_heat.tick_params(axis='x', which='both', labelsize=10)
            ax_heat.margins(x=0)

            cbar = fig.colorbar(im, cax=cbar_ax)
            if is_1d_design and not sort:
                cbar.set_label('Design Value', labelpad=15, fontsize=12, weight='bold')
            else:
                cbar.set_label('Ratio to Nominal Design' if cmap == 'RdBu' else 'Design Value', 
                              labelpad=15, fontsize=12, weight='bold')
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
            if self.mlflow_exp and actual_mlflow_exp_for_title:
                title = f'EIG per Design - {actual_mlflow_exp_for_title}'
            elif self.mlflow_exp:
                title = f'EIG per Design - {self.mlflow_exp}'
            else:
                title = 'EIG per Design Comparison'
        fig.suptitle(title, fontsize=16, y=0.95, weight='bold')

        if save_path is not None:
            # Extract filename from save_path
            filename = os.path.basename(save_path)
            # Determine run_id and experiment_id from save_path or use defaults
            # If save_path contains run_id, extract it; otherwise use None
            save_path_dir = os.path.dirname(save_path)
            run_id_for_save = None
            experiment_id_for_save = None
            # Try to extract from path structure: .../mlruns/{exp_id}/{run_id}/artifacts/plots/...
            if 'mlruns' in save_path_dir:
                parts = save_path_dir.split('mlruns/')
                if len(parts) > 1:
                    remaining = parts[1]
                    path_parts = remaining.split('/')
                    if len(path_parts) >= 2:
                        experiment_id_for_save = path_parts[0]
                        run_id_for_save = path_parts[1] if len(path_parts) > 1 else None
            # Use self.save_figure() which handles directory creation
            actual_save_path = self.save_figure(fig, filename, run_id=run_id_for_save, 
                                               experiment_id=experiment_id_for_save, dpi=dpi)
            print(f"Saved comparison plot to {actual_save_path}")

        return fig, (ax_line, ax_heat)
    
    def compare_optimal_designs(
        self,
        eval_step=None,
        top_n=1,
        save_path=None,
        figsize=(14, 8),
        dpi=400,
        colors=None,
        title=None,
        log_scale=True,
        display_mode='absolute',
        include_nominal=False,
        # Additional params for top_n > 1 (heatmap mode)
        cmap='viridis',
        nominal_design=None
    ):
        """
        Compare optimal designs (or top N best designs) across multiple runs.
        
        When top_n=1 (default), displays optimal designs as bar charts (1D) or grouped bar charts (multi-dimensional).
        When top_n>1, displays top N designs per run as a heatmap with visual separation between runs.
        
        Args:
            eval_step (str or int, optional): Which step to use. If None, finds most recent eig_data file
            top_n (int): Number of designs to plot from each run (default: 1). When top_n=1, uses bar chart. When top_n>1, uses heatmap.
            save_path (str, optional): Path to save the comparison plot. If None, doesn't save
            figsize (tuple): Figure size (width, height)
            dpi (int): Resolution for saved figure
            colors (list, optional): Colors for each run. If None, uses default color cycle (only used for top_n=1)
            title (str, optional): Custom title for the plot. If None, generates default title
            log_scale (bool): Whether to use log scale for y-axis (default: True, only used for top_n=1)
            display_mode (str): How to display design values for top_n=1. Options: 'absolute', 'fractional', or 'ratio' (default: 'absolute')
            include_nominal (bool): Whether to include nominal design bars on the left (default: False, only used for top_n=1)
            cmap (str): Colormap to use for the heatmap when top_n>1 (default: 'viridis'). If nominal_design is provided, a diverging colormap will be used.
            nominal_design (list, optional): Nominal design as a list of floats for top_n>1. If provided, colors will be plotted as ratios relative to this nominal design.
        
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        storage_path = self.storage_path

        # Get run data
        run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = self._get_run_data_list(
            parse_params=True
        )
        if not run_data_list:
            raise ValueError(f"No runs found in experiment {self.cosmo_exp}")
        
        # Extract run_ids from run_data_list
        run_ids = [run_data['run_id'] for run_data in run_data_list]
        
        # Build a map from run_id to exp_id
        run_id_to_exp_id = self._get_run_id_to_exp_id(run_data_list)
        
        # Branch based on top_n
        if top_n == 1:
            # ===================================================================
            # BAR CHART MODE (original compare_optimal_design logic)
            # ===================================================================
            
            # Validate display_mode
            valid_modes = ['absolute', 'fractional', 'ratio']
            if display_mode not in valid_modes:
                raise ValueError(f"display_mode must be one of {valid_modes}, got '{display_mode}'")
            
            # Find and load JSON files for each run
            optimal_designs = []
            optimal_eigs = []
            found_run_ids = []
            design_labels = None
            
            for run_id in run_ids:
                if run_id not in run_id_to_exp_id:
                    print(f"Warning: Run {run_id} not found in experiment {self.cosmo_exp}, skipping...")
                    continue
                
                exp_id = run_id_to_exp_id[run_id]
                artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
                
                # Load completed eig_data file
                try:
                    json_path, data = self.load_eig_data_file(artifacts_dir, eval_step=None)
                    if data is None:
                        print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                        continue
                except ValueError as e:
                    print(f"Warning: {e}, skipping run {run_id}...")
                    continue
                
                try:
                    # Determine which step to use
                    if eval_step is not None:
                        step_str = f"step_{eval_step}"
                    else:
                        # Find the highest available step key
                        step_keys = [k for k in data.keys() if k.startswith('step_')]
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
                            else:
                                step_str = step_keys[0]
                        else:
                            print(f"Warning: No step data found in {json_path}, skipping...")
                            continue
                    
                    if step_str not in data:
                        print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                        continue
                    
                    step_data = data[step_str]
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
                        
                except Exception as e:
                    print(f"Warning: Error loading {json_path}: {e}, skipping...")
                    continue
            
            if len(optimal_designs) == 0:
                raise ValueError("No valid optimal design data found to compare")
            
            # Resolve run labels using centralized method
            labels = self._resolve_run_labels(run_data_list, found_run_ids, var=None)
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
            
            colors = [convert_color(c) for c in colors]
            
            # Initialize experiments for each run to get per-run nominal designs and other metadata
            nominal_total_obs_list = []
            nominal_designs = []
            design_labels = None
            needs_experiment = (display_mode in ['absolute', 'ratio'] or design_labels is None or include_nominal)
            
            if needs_experiment:
                device = "cuda:0"
                run_id_to_run_data = {run_data['run_id']: run_data for run_data in run_data_list}
                
                for run_id in found_run_ids:
                    if run_id not in run_id_to_run_data:
                        print(f"Warning: Run {run_id} not found in run_data_list, skipping experiment initialization...")
                        nominal_total_obs_list.append(None)
                        nominal_designs.append(None)
                        continue
                    
                    run_data = run_id_to_run_data[run_id]
                    try:
                        run_params = run_data['params'].copy()  # Make a copy to avoid modifying original
                        
                        # init_experiment will automatically load prior_args from artifacts
                        experiment = init_experiment(
                            run_data['run_obj'], 
                            run_params,
                            device=device, 
                            global_rank=0
                        )
                        
                        if display_mode == 'absolute' or display_mode == 'ratio':
                            nominal_total_obs_list.append(experiment.nominal_total_obs)
                        else:
                            nominal_total_obs_list.append(None)
                        
                        if (display_mode == 'ratio' or include_nominal) and hasattr(experiment, 'nominal_design'):
                            nominal_designs.append(experiment.nominal_design.cpu().numpy())
                        else:
                            nominal_designs.append(None)
                        
                        if design_labels is None and hasattr(experiment, 'design_labels'):
                            design_labels = experiment.design_labels
                            
                    except Exception as e:
                        traceback.print_exc()
                        nominal_total_obs_list.append(None)
                        nominal_designs.append(None)
                
                if display_mode == 'absolute' and all(nto is None for nto in nominal_total_obs_list):
                    print(f"Warning: Could not get nominal_total_obs for any run. Cannot convert to absolute quantities.")
                if display_mode == 'ratio' and all(nd is None for nd in nominal_designs):
                    print(f"Warning: Could not get nominal_design for any run. Cannot compute ratios.")
                if include_nominal and all(nd is None for nd in nominal_designs):
                    print(f"Warning: Could not get nominal designs. Skipping nominal design bars.")
            
            # Use generic labels if design_labels not found
            if design_labels is None:
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
            if display_mode == 'absolute':
                converted_designs = []
                for i, design in enumerate(optimal_designs):
                    if i < len(nominal_total_obs_list) and nominal_total_obs_list[i] is not None:
                        converted_designs.append(design * nominal_total_obs_list[i])
                    else:
                        print(f"Warning: nominal_total_obs not available for run {found_run_ids[i]}, keeping fractional values.")
                        converted_designs.append(design)
                optimal_designs = converted_designs
                
                if include_nominal:
                    for i, nominal_design in enumerate(nominal_designs):
                        if nominal_design is not None and i < len(nominal_total_obs_list) and nominal_total_obs_list[i] is not None:
                            nominal_designs[i] = nominal_design * nominal_total_obs_list[i]
                            
            elif display_mode == 'ratio':
                converted_designs = []
                for i, design in enumerate(optimal_designs):
                    if i < len(nominal_designs) and nominal_designs[i] is not None:
                        nominal_design_safe = np.where(nominal_designs[i] == 0, np.nan, nominal_designs[i])
                        converted_designs.append(design / nominal_design_safe)
                    else:
                        print(f"Warning: nominal_design not available for run {found_run_ids[i]}, cannot compute ratio. Keeping fractional values.")
                        converted_designs.append(design)
                optimal_designs = converted_designs
                
                if include_nominal:
                    for i, nominal_design in enumerate(nominal_designs):
                        if nominal_design is not None:
                            nominal_designs[i] = np.ones_like(nominal_design)
            
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
                fig, ax = plt.subplots(figsize=figsize)
                
                design_values = [design[0] for design in optimal_designs]
                plot_labels = list(labels) if labels else [run_id[:8] for run_id in found_run_ids]
                plot_colors = list(colors) if colors else [plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])] for i in range(len(optimal_designs))]
                
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
                
                x = np.arange(len(design_labels))
                
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
                        nominal_colors = ['tab:blue'] * len(nominal_plot_designs)
                        plot_colors = nominal_colors + plot_colors
                
                n_designs_to_plot = len(plot_designs)
                width = 0.8 / n_designs_to_plot
                
                for i, (design, label, color) in enumerate(zip(plot_designs, plot_labels, plot_colors)):
                    offset = (i - n_designs_to_plot/2 + 0.5) * width
                    bars = ax.bar(x + offset, design, width, label=label, color=color, alpha=0.7, linewidth=1.5, zorder=2)
                
                ax.set_xlabel('Tracer Class', fontsize=12, weight='bold')
                ax.set_ylabel(ylabel, fontsize=12, weight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(design_labels, fontsize=11)
                self._set_legend(ax, len(plot_labels), fontsize=16)
                ax.grid(True, alpha=0.3, axis='y')
                if log_scale:
                    ax.set_yscale('log')
            
            # Set title
            if title is None:
                if self.mlflow_exp and actual_mlflow_exp_for_title:
                    title = f'Optimal Design Comparison - {actual_mlflow_exp_for_title}'
                elif self.mlflow_exp:
                    title = f'Optimal Design Comparison - {self.mlflow_exp}'
                else:
                    title = f'Optimal Design Comparison Across Runs'
            ax.set_title(title, fontsize=14, weight='bold', pad=20)
            
        else:
            # ===================================================================
            # HEATMAP MODE (original compare_best_designs logic)
            # ===================================================================
            
            # Find and load JSON files for each run
            all_run_top_designs = []
            found_run_ids = []
            run_boundaries = []
            
            for run_id in run_ids:
                if run_id not in run_id_to_exp_id:
                    print(f"Warning: Run {run_id} not found in experiment {self.cosmo_exp}, skipping...")
                    continue
                
                exp_id = run_id_to_exp_id[run_id]
                artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
                
                try:
                    json_path, data = self.load_eig_data_file(artifacts_dir, eval_step=None)
                    if data is None:
                        print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                        continue
                except ValueError as e:
                    print(f"Warning: {e}, skipping run {run_id}...")
                    continue
                
                try:
                    # Determine which step to use
                    if eval_step is not None:
                        step_str = f"step_{eval_step}"
                    else:
                        step_keys = [k for k in data.keys() if k.startswith('step_')]
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
                            else:
                                step_str = step_keys[0]
                        else:
                            print(f"Warning: No step data found in {json_path}, skipping...")
                            continue
                    
                    if step_str not in data:
                        print(f"Warning: Step {step_str} not found in {json_path}, skipping...")
                        continue
                    
                    step_data = data[step_str]
                    variable_data = step_data.get('variable', {})
                    
                    if 'input_designs' not in data:
                        print(f"Warning: No input_designs found in {json_path}, skipping...")
                        continue
                    
                    if 'eigs_avg' not in variable_data:
                        print(f"Warning: No eigs_avg data found in {json_path} for {step_str}, skipping...")
                        continue
                    
                    designs = np.array(data['input_designs'])
                    eigs = np.array(variable_data['eigs_avg'])
                    
                    if len(designs) != len(eigs):
                        print(f"Warning: Mismatch - {len(designs)} designs but {len(eigs)} EIG values for run {run_id}")
                    
                    # Sort designs by EIG (descending) and take top N
                    sorted_indices = np.argsort(eigs)[::-1]
                    top_n_indices = sorted_indices[:min(top_n, len(designs))]
                    top_designs = designs[top_n_indices]
                    
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
            
            # Resolve run labels using centralized method
            labels = self._resolve_run_labels(run_data_list, found_run_ids, var=None)
            if len(labels) != len(all_run_top_designs):
                labels = labels[:len(all_run_top_designs)]
            
            # Determine design labels and nominal design from first run's experiment
            design_labels = None
            experiment_nominal_design = None
            try:
                first_run_data = run_data_list[0]
                device = "cuda:0"
                run_params = first_run_data['params'].copy()  # Make a copy to avoid modifying original

                # init_experiment will automatically load prior_args from artifacts
                experiment = init_experiment(
                    first_run_data['run_obj'],
                    run_params,
                    device=device,
                    global_rank=0
                )
                if hasattr(experiment, 'design_labels'):
                    design_labels = experiment.design_labels
                if hasattr(experiment, 'nominal_design'):
                    experiment_nominal_design = experiment.nominal_design.cpu().numpy()
            except Exception as e:
                print(f"Warning: Could not initialize experiment to get design labels/nominal: {e}")
            
            # Use generic labels if still None
            if design_labels is None:
                n_dims = combined_designs.shape[1]
                design_labels = [f'$f_{i}$' for i in range(n_dims)]
            
            if len(design_labels) != combined_designs.shape[1]:
                raise ValueError(f"design_labels length ({len(design_labels)}) must match number of design dimensions ({combined_designs.shape[1]})")
            
            # Resolve nominal design: use parameter if provided, else fall back to experiment's nominal
            if nominal_design is not None:
                nominal_design = np.array(nominal_design)
            elif experiment_nominal_design is not None:
                nominal_design = experiment_nominal_design

            # Compute ratios relative to nominal if available
            plot_data = combined_designs.copy()
            use_relative_colors = False
            if nominal_design is not None:
                if len(nominal_design) != combined_designs.shape[1]:
                    raise ValueError(f"nominal_design length ({len(nominal_design)}) must match number of design dimensions ({combined_designs.shape[1]})")

                if np.any(nominal_design == 0):
                    raise ValueError("nominal_design cannot contain zero values (division by zero)")

                plot_data = combined_designs / nominal_design[np.newaxis, :]
                use_relative_colors = True
                cmap = 'RdBu'
            
            # Create figure with space for colorbar
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(left=0.06, right=0.99)
            
            ax = fig.add_subplot(111)
            
            # Plot heatmap using imshow
            if use_relative_colors:
                from matplotlib.colors import TwoSlopeNorm
                vmin = plot_data.min()
                vmax = plot_data.max()
                norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap, norm=norm)
            else:
                im = ax.imshow(plot_data.T, aspect='auto', cmap=cmap)
            
            # Add vertical lines to separate runs
            for start_idx, end_idx in run_boundaries:
                if start_idx > 0:
                    ax.axvline(start_idx - 0.5, color='white', linewidth=2, linestyle='-', alpha=0.8)
            
            # Set y-axis labels
            ax.set_yticks(np.arange(len(design_labels)))
            ax.set_yticklabels(design_labels, fontsize=14)
            ax.set_ylabel('Design Dimension', fontsize=12, weight='bold')
            
            # Set x-axis ticks and labels for run groups
            x_tick_positions = []
            x_tick_labels = []
            for i, (start_idx, end_idx) in enumerate(run_boundaries):
                tick_pos = (start_idx + end_idx - 1) / 2
                x_tick_positions.append(tick_pos)
                x_tick_labels.append(labels[i] if i < len(labels) else found_run_ids[i][:8])
            
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
            ax.set_xlim(-0.5, len(combined_designs) - 0.5)
            
            # Add colorbar
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
                if self.mlflow_exp and actual_mlflow_exp_for_title:
                    title = f'Top {top_n} Designs Comparison - {actual_mlflow_exp_for_title}'
                elif self.mlflow_exp:
                    title = f'Top {top_n} Designs Comparison - {self.mlflow_exp}'
                else:
                    title = f'Top {top_n} Designs Comparison Across Runs'
            ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure (auto-save if save_path not provided)
        if save_path is not None:
            # Extract filename from save_path
            filename = os.path.basename(save_path)
            # Determine run_id and experiment_id from save_path or use defaults
            save_path_dir = os.path.dirname(save_path)
            run_id_for_save = None
            experiment_id_for_save = None
            # Try to extract from path structure: .../mlruns/{exp_id}/{run_id}/artifacts/plots/...
            if 'mlruns' in save_path_dir:
                parts = save_path_dir.split('mlruns/')
                if len(parts) > 1:
                    remaining = parts[1]
                    path_parts = remaining.split('/')
                    if len(path_parts) >= 2:
                        experiment_id_for_save = path_parts[0]
                        run_id_for_save = path_parts[1] if len(path_parts) > 1 else None
            # Use self.save_figure() which handles directory creation
            actual_save_path = self.save_figure(fig, filename, run_id=run_id_for_save, 
                                               experiment_id=experiment_id_for_save, dpi=dpi)
            print(f"Saved comparison plot to {actual_save_path}")
        else:
            # Auto-save with generated filename to experiment's plots directory
            filename = self.generate_filename("optimal_designs_comparison")
            self.save_figure(fig, filename, run_id=None, experiment_id=experiment_id_for_save_path, dpi=dpi, close_fig=False, display_fig=False)
        
        return fig, ax
    
    def compare_training(
        self,
        var=None,
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

        Args:
            var (str or list): Parameter(s) from MLflow run params to include in the label.
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
        storage_path = self.storage_path
        mlflow.set_tracking_uri(storage_path + "/mlruns")
        client = MlflowClient()

        run_data_list, experiment_id_for_save_path, _ = self._get_run_data_list(
            parse_params=True
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

            # Resolve run_labels using centralized method
            resolved_labels = self._resolve_run_labels(run_data_list, [run_id_iter], var=var)
            base_label = resolved_labels[0]

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
        num_runs = len(valid_runs_processed_for_metrics)

        # Configure ax1 (Loss)
        ax1.set_ylabel("Loss")
        ax1.tick_params(axis='y')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        self._set_legend(ax1, num_runs)
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
        title = f"Training History - {self.mlflow_exp}" if self.mlflow_exp else f"Training History - {len(run_data_list)} Run(s)"
        if param_pair:
            title += f" - Area: {param_pair}"
        fig.set_constrained_layout(True)
        fig.suptitle(title, fontsize=16)

        # Determine save path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = None
        run_id_for_save = None
        experiment_id_for_save = experiment_id_for_save_path

        if self.mlflow_exp and experiment_id_for_save:
            filename = f"training_comparison_{timestamp}.png"
        elif len(run_data_list) == 1:
            single_run_id = run_data_list[0]['run_id']
            run_id_for_save = single_run_id
            if not experiment_id_for_save:
                try:
                    single_run_data_item = run_data_list[0]
                    if 'run_obj' in single_run_data_item and single_run_data_item['run_obj'] is not None:
                        experiment_id_for_save = single_run_data_item['run_obj'].info.experiment_id
                    else:
                        fallback_run_obj = client.get_run(single_run_id)
                        experiment_id_for_save = fallback_run_obj.info.experiment_id
                except Exception as e:
                    print(f"COMPARE_TRAINING: Warning: Could not determine experiment ID for run {single_run_id} for save_dir: {e}. Defaulting.")
            
            filename = f"training_{timestamp}.png"
        else:
            filename = f"training_comparison_{timestamp}.png"

        # Use self.save_figure() which handles directory creation
        save_path = self.save_figure(fig, filename, run_id=run_id_for_save, 
                                     experiment_id=experiment_id_for_save, dpi=dpi)
        plt.close(fig)

    def plot_design_dim_by_eig(
        self,
        run_labels=None,
        eval_step=None,
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
            run_labels (list, optional): Labels for each run. If None, uses run IDs (first 8 chars)
            eval_step (str or int, optional): Which step to use. If None, finds most recent eig_data file
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
        
        # Use instance storage_path
        storage_path = self.storage_path
        
        # Get run data using class method
        run_data_list, experiment_id_for_save_path, actual_mlflow_exp_for_title = self._get_run_data_list(
            parse_params=True
        )
        if not run_data_list:
            raise ValueError(f"No runs found in experiment {self.cosmo_exp}")
        
        # Extract run_ids from run_data_list
        run_ids = [run_data['run_id'] for run_data in run_data_list]
        
        # Build a map from run_id to exp_id
        run_id_to_exp_id = self._get_run_id_to_exp_id(run_data_list)
        
        # Find and load JSON files for each run
        all_run_data = []  # List of dicts: each dict contains designs, eigs, and run_id for that run
        found_run_ids = []
        
        for run_id in run_ids:
            if run_id not in run_id_to_exp_id:
                print(f"Warning: Run {run_id} not found in experiment {self.cosmo_exp}, skipping...")
                continue
            
            exp_id = run_id_to_exp_id[run_id]
            artifacts_dir = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts"
            
            # Load completed eig_data file using class method
            try:
                json_path, data = self.load_eig_data_file(artifacts_dir, eval_step=None)
                if data is None:
                    print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                    continue
            except ValueError as e:
                print(f"Warning: {e}, skipping run {run_id}...")
                continue
            
            try:
                
                # Determine which step to use
                if eval_step is not None:
                    step_str = f"step_{eval_step}"
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
                    device=device, 
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
            fig.savefig(save_path, dpi=dpi)
            print(f"Saved plot to {save_path}")
        
        return fig, axes
    

# ============================================================================
# Standalone Plotting Functions (for backward compatibility and utilities)
# ============================================================================

def compare_increasing_design(
        mlflow_exp=None,
        run_ids=None,
        excluded_runs=[],
        cosmo_exp='num_tracers',
        labels=None,
        eval_step=None,
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
        eval_step (str or int, optional): Which step to use. If None, finds most recent eig_data file
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
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
    
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
                        ref_json_path, ref_data = load_eig_data_file(ref_artifacts_dir, eval_step=None)
                        if ref_data is not None:
                            print(f"Loaded reference EIG data from {ref_json_path}")
                            
                            # Determine which step to use
                            if eval_step is not None:
                                step_str = f"step_{eval_step}"
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
            json_path, data = load_eig_data_file(artifacts_dir, eval_step=None)
            if data is None:
                print(f"Warning: No completed eig_data file found for run {run_id}, skipping...")
                continue
            print(f"Loaded EIG data from {json_path}")
        except ValueError as e:
            print(f"Warning: {e}, skipping run {run_id}...")
            continue
        
        try:
            
            # Determine which step to use
            if eval_step is not None:
                step_str = f"step_{eval_step}"
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
                device=device, 
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
        fig.savefig(save_path, dpi=dpi)
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
            fig2.savefig(scatter_save_path, dpi=dpi)
            print(f"Saved scatter plot to {scatter_save_path}")
    else:
        fig2 = None
        ax2 = None
    
    return fig, ax
    
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
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
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
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
    os.makedirs(f"{storage_path}/mlruns/{exp_id}/plots", exist_ok=True)
    save_path = f"{storage_path}/mlruns/{exp_id}/plots/loss_area_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.gcf().savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

def plot_lr_schedule(initial_lr, gamma, gamma_freq, steps=100000):
    steps = np.arange(0, steps, 1)
    lr = initial_lr * gamma ** (steps / gamma_freq)
    legend_label = f'initial_lr={initial_lr}, gamma={gamma}, gamma_freq={gamma_freq}'
    plt.plot(steps, lr, label=legend_label)

    return lr[-1]



def plot_2d_eig(
    run_id,
    eval_step=None,
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
        eval_step (str or int, optional): Step key to filter eig_data files (e.g., 50000 or 'last').
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
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
    
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
    json_path, eig_data = load_eig_data_file(artifacts_dir, eval_step=None)
    if eig_data is None:
        raise ValueError(f"No completed eig_data file found for run {run_id}")
    
    # Determine which step to use
    if eval_step is not None:
        step_str = f"step_{eval_step}"
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
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved plot to {save_path}")
    
    return fig
