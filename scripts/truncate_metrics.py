#!/usr/bin/env python3
"""
Script to truncate MLflow metrics files to remove rows beyond a specified resume step.

This script is designed to work with the bedcosmo MLflow tracking structure where metrics
are stored in individual files per metric name, with each line containing:
timestamp value step

The script searches backwards from the end of each file for optimal performance when
the resume step is near the end (the common case when resuming from large step counts).

Usage:
    python truncate_metrics.py --run_id <run_id> --resume_step <step> [--dry_run]
    
Example:
    python truncate_metrics.py --run_id dcc7aa8be3ce43b1809dd3adb7772823 --resume_step 41000
"""

import os
import argparse
import glob
from pathlib import Path
import shutil

scratch_dir = os.environ["SCRATCH"]

def find_metrics_directory(run_id, cosmo_exp=None):
    """
    Find the metrics directory for a given run_id by searching through the mlruns structure.
    
    Args:
        run_id (str): The MLflow run ID to search for
        cosmo_exp (str, optional): Cosmological experiment name (e.g., 'num_tracers', 'variable_redshift')
        
    Returns:
        str: Path to the metrics directory, or None if not found
    """
    # If cosmo_exp is specified, search only in that directory
    if cosmo_exp:
        search_paths = [
            f"{cosmo_exp}/mlruns",
            f"../{cosmo_exp}/mlruns",
            f"../../{cosmo_exp}/mlruns",
            os.path.join(scratch_dir, f"bedcosmo/{cosmo_exp}/mlruns")
        ]
    else:
        # Search in all known locations
        search_paths = [
            "num_tracers/mlruns",
            "../num_tracers/mlruns", 
            "../../num_tracers/mlruns",
            os.path.join(scratch_dir, "bedcosmo/num_tracers/mlruns"),
            "variable_redshift/mlruns",
            "../variable_redshift/mlruns",
            "../../variable_redshift/mlruns",
            os.path.join(scratch_dir, "bedcosmo/variable_redshift/mlruns")
        ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            # Look for the run_id in experiment directories
            for exp_dir in glob.glob(os.path.join(search_path, "*")):
                if os.path.isdir(exp_dir) and os.path.basename(exp_dir).isdigit():
                    run_dir = os.path.join(exp_dir, run_id)
                    if os.path.exists(run_dir):
                        metrics_dir = os.path.join(run_dir, "metrics")
                        if os.path.exists(metrics_dir):
                            return metrics_dir
    
    return None


def truncate_metrics_file(file_path, resume_step, dry_run=False):
    """
    Truncate a metrics file to keep only rows up to the resume_step.
    
    Searches backwards from the end of the file for optimal performance when
    the resume_step is near the end (the typical case).
    
    Args:
        file_path (str): Path to the metrics file
        resume_step (int): Step number to truncate at (inclusive)
        dry_run (bool): If True, only show what would be done without making changes
        
    Returns:
        tuple: (original_line_count, new_line_count, truncated)
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return 0, 0, False
    
    # Read all lines and find the cutoff point
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    original_count = len(lines)
    cutoff_index = -1
    
    # Search backwards from the end since truncate_step is almost always near the end
    # This is much faster when resuming from large step counts
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                step = int(parts[2])
                if step <= resume_step:
                    cutoff_index = i
                    break  # Found the last valid line, stop searching
            except ValueError:
                # Skip lines that don't have a valid step number
                continue
    
    if cutoff_index == -1:
        print(f"Warning: No valid steps found in {file_path}")
        return original_count, 0, False
    
    new_count = cutoff_index + 1
    
    if dry_run:
        print(f"  Would truncate {file_path}: {original_count} -> {new_count} lines")
        return original_count, new_count, True
    
    # Write truncated content
    with open(file_path, 'w') as f:
        f.writelines(lines[:new_count])
    
    return original_count, new_count, True


def main():
    parser = argparse.ArgumentParser(
        description="Truncate MLflow metrics files to remove rows beyond a specified resume step"
    )
    parser.add_argument("--run_id", required=True, help="MLflow run ID")
    parser.add_argument("--resume_step", type=int, required=True, help="Resume step number (inclusive)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--metrics_dir", help="Direct path to metrics directory (optional)")
    parser.add_argument("--cosmo_exp", help="Cosmological experiment name (e.g., 'num_tracers', 'variable_redshift')")
    
    args = parser.parse_args()
    
    print(f"Truncating metrics for run_id: {args.run_id}")
    print(f"Resume step: {args.resume_step}")
    print(f"Dry run: {args.dry_run}")
    if args.cosmo_exp:
        print(f"Cosmo experiment: {args.cosmo_exp}")
    
    # Find metrics directory
    if args.metrics_dir:
        metrics_dir = args.metrics_dir
        if not os.path.exists(metrics_dir):
            print(f"Error: Specified metrics directory does not exist: {metrics_dir}")
            return 1
    else:
        metrics_dir = find_metrics_directory(args.run_id, args.cosmo_exp)
        if not metrics_dir:
            print(f"Error: Could not find metrics directory for run_id: {args.run_id}")
            print("Searched in:")
            if args.cosmo_exp:
                search_paths = [
                    f"{args.cosmo_exp}/mlruns",
                    f"../{args.cosmo_exp}/mlruns",
                    f"../../{args.cosmo_exp}/mlruns",
                    os.path.join(scratch_dir, f"bedcosmo/{args.cosmo_exp}/mlruns")
                ]
            else:
                search_paths = [
                    "num_tracers/mlruns", "../num_tracers/mlruns", "../../num_tracers/mlruns",
                    "variable_redshift/mlruns", "../variable_redshift/mlruns", "../../variable_redshift/mlruns"
                ]
            for path in search_paths:
                print(f"  {path}")
            print("You can specify the metrics directory directly with --metrics_dir")
            return 1
    
    print(f"Found metrics directory: {metrics_dir}")
    
    # Get all metric files
    metric_files = glob.glob(os.path.join(metrics_dir, "*"))
    metric_files = [f for f in metric_files if os.path.isfile(f)]
    
    
    if not metric_files:
        print("Warning: No metric files found (excluding backup files)")
        return 0
    
    print(f"Found {len(metric_files)} metric files (excluding backup files)")
    
    # Process each metric file
    total_original = 0
    total_new = 0
    total_truncated = 0
    
    for metric_file in sorted(metric_files):
        original, new, truncated = truncate_metrics_file(
            metric_file, args.resume_step, args.dry_run
        )
        total_original += original
        total_new += new
        if truncated:
            total_truncated += 1
    
    print(f"\nSummary:")
    print(f"  Files processed: {len(metric_files)}")
    print(f"  Files truncated: {total_truncated}")
    print(f"  Total lines before: {total_original}")
    print(f"  Total lines after: {total_new}")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")
        print("Run without --dry_run to actually truncate the files.")
    
    return 0


if __name__ == "__main__":
    exit(main())
