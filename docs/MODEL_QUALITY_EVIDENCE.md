# Model Quality Evidence

`experiments.model_quality` provides repeatable checks for the failure modes
that are easy to miss in a performance-only benchmark:

- parameter integrity before inference, plus deterministic weight perturbations;
- finite forward outputs, scalar losses, and finite gradients after backward;
- CPU/CUDA output agreement with explicit tolerances;
- a training step followed by checkpoint save/load, `torch.export` (with a
  TorchScript fallback), and a deployment-style inference probe.

The lifecycle result names are `train`, `save`, `export`, `load`, and `deploy`,
so a failed release probe identifies the broken stage rather than collapsing
the entire chain into one status.

## Running the checks

From the repository root:

```bash
python -m pytest tests/test_model_quality.py -q
```

The CPU/CUDA test does not pretend that a CPU-only environment proves device
equivalence. When CUDA is unavailable, the result is marked as a passed,
explicitly skipped comparison (`skipped=1`). To collect real CUDA evidence,
run the same test with a CUDA-enabled PyTorch installation and record the
reported `max_abs_diff`, `atol`, and `rtol` values.

## Evidence boundary

These are smoke and robustness probes, not a claim that arbitrary corrupted
weights remain accurate. A perturbation check establishes that the model still
produces finite values; task quality must be assessed separately with a
representative validation set. Export and deployment probes cover the local
TorchScript path only; a serving runtime should add its own endpoint and
serialization checks.
