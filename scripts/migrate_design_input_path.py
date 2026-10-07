#!/usr/bin/env python3
"""
One-off migration: rename ``input_designs_path`` -> ``input_path`` in the design_args.yaml
artifacts of existing MLflow runs. Optional: old keys are still read as ``input_path`` at load
time; this just brings saved artifacts in line with the current name.

Dry run by default (lists the files it would change); pass --apply to rewrite them.

Usage:
    python scripts/migrate_design_input_path.py            # dry run
    python scripts/migrate_design_input_path.py --apply
"""

import argparse
import glob
import os
import re

OLD_KEY = re.compile(r"\binput_designs_path\b")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="Rewrite files (default: dry run)")
    args = parser.parse_args()

    pattern = os.path.join(os.environ["SCRATCH"], "bedcosmo", "*", "mlruns", "*", "*", "artifacts", "design_args.yaml")
    changed = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            text = f.read()
        if not OLD_KEY.search(text):
            continue
        changed.append(path)
        if args.apply:
            with open(path, "w") as f:
                f.write(OLD_KEY.sub("input_path", text))

    for path in changed:
        print(path)
    print(f"{'Rewrote' if args.apply else 'Would rewrite'} {len(changed)} design_args.yaml files")


if __name__ == "__main__":
    main()
