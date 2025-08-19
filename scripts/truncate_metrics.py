#!/usr/bin/env python3
"""
Script to truncate MLflow metrics files to remove rows beyond a specified resume step.

This script is designed to work with the BED_cosmo MLflow tracking structure where metrics
are stored in individual files per metric name, with each line containing:
timestamp value step

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


def find_metrics_directory(run_id):
    """
    Find the metrics directory for a given run_id by searching through the mlruns structure.
    
    Args:
        run_id (str): The MLflow run ID to search for
        
    Returns:
        str: Path to the metrics directory, or None if not found
    """
    # Search in the current working directory and common locations
    search_paths = [
        "num_tracers/mlruns",
        "../num_tracers/mlruns", 
        "../../num_tracers/mlruns",
        "/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/mlruns"
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
    
    # Find the last line that has step <= resume_step
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                step = int(parts[2])
                if step <= resume_step:
                    cutoff_index = i
                else:
                    break
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
    
    args = parser.parse_args()
    
    print(f"Truncating metrics for run_id: {args.run_id}")
    print(f"Resume step: {args.resume_step}")
    print(f"Dry run: {args.dry_run}")
    
    # Find metrics directory
    if args.metrics_dir:
        metrics_dir = args.metrics_dir
        if not os.path.exists(metrics_dir):
            print(f"Error: Specified metrics directory does not exist: {metrics_dir}")
            return 1
    else:
        metrics_dir = find_metrics_directory(args.run_id)
        if not metrics_dir:
            print(f"Error: Could not find metrics directory for run_id: {args.run_id}")
            print("Searched in:")
            for path in ["num_tracers/mlruns", "../num_tracers/mlruns", "../../num_tracers/mlruns"]:
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
