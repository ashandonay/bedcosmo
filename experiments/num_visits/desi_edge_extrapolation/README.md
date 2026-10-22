# Archived DESI edge-extrapolation studies

The individual-spectrum, endpoint-quality, regularized-slope and early
fixed-width template studies are archived in Git at commit `4c04491`.
They are research snapshots, not part of the current extrapolation workflow.

To recover that snapshot in a separate directory without changing this checkout:

```bash
git archive 4c04491 experiments/num_visits/desi_edge_extrapolation | tar -x -C /path/to/archive
```

Use a checkout of `4c04491` to reproduce its old scripts with their matching
package implementation. The old input coadds and training matrices remain
external data; Git contains scripts, protocols, measurements and figures.

Current template extrapolation and response-based evaluation live in
`src/bedcosmo/num_visits/empirical/desi/extrapolation.py`. See the
[DESI build guide](../../../src/bedcosmo/num_visits/empirical/desi/README.md)
for `extend`, `response-compare`, and `response-plot`. Production remains
measured-only NMF followed by constant template extension by default.
