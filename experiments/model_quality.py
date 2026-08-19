"""Model quality probes for weights, autograd, devices, and lifecycle checks.

The probes operate on ordinary ``torch.nn.Module`` instances and return
structured results so callers can persist evidence or fail a CI/deployment
gate without parsing exception text.
"""

from __future__ import annotations

import copy
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn


@dataclass
class CheckResult:
    """One named quality check and its optional measurements."""

    name: str
    passed: bool
    details: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def _finite_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value.detach()).all().item())


def inspect_weights(model: nn.Module) -> CheckResult:
    """Detect missing, empty, non-finite, or invalid model parameters."""

    parameters = list(model.named_parameters())
    if not parameters:
        return CheckResult("weight_integrity", False, "model has no parameters")
    bad: list[str] = []
    maximum = 0.0
    for name, parameter in parameters:
        if parameter.numel() == 0:
            bad.append(f"{name}: empty")
        elif not _finite_tensor(parameter):
            bad.append(f"{name}: non-finite")
        else:
            maximum = max(maximum, float(parameter.detach().abs().max().item()))
    if not math.isfinite(maximum):
        bad.append("invalid parameter magnitude")
    return CheckResult(
        "weight_integrity",
        not bad,
        "; ".join(bad),
        {"parameter_count": float(sum(p.numel() for _, p in parameters)), "max_abs": maximum},
    )


def perturb_weights(
    model: nn.Module,
    *,
    fraction: float = 0.01,
    noise_scale: float = 0.1,
    seed: int = 0,
) -> nn.Module:
    """Return a deterministic deep copy with a fraction of weights perturbed."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    candidate = copy.deepcopy(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in candidate.parameters():
            if parameter.numel() == 0 or fraction == 0:
                continue
            mask = torch.rand(parameter.shape, generator=generator) < fraction
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(noise.to(parameter.device, parameter.dtype) * noise_scale * mask.to(parameter.device, parameter.dtype))
    return candidate


def weight_robustness_probe(
    model: nn.Module,
    inputs: Any,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    perturbations: Sequence[tuple[float, float]] = ((0.01, 0.1), (0.1, 0.5)),
) -> list[CheckResult]:
    """Check baseline and perturbed models for finite inference and loss."""

    def evaluate(candidate: nn.Module) -> torch.Tensor:
        was_training = candidate.training
        candidate.eval()
        try:
            with torch.no_grad():
                output = candidate(inputs)
                return loss_fn(output) if loss_fn is not None else output
        finally:
            candidate.train(was_training)

    results = [inspect_weights(model)]
    try:
        baseline = evaluate(model)
        ok = _finite_tensor(baseline)
        results.append(CheckResult("baseline_inference", ok, "" if ok else "non-finite or non-tensor output"))
    except Exception as exc:  # pragma: no cover - model-specific exceptions
        results.append(CheckResult("baseline_inference", False, f"{type(exc).__name__}: {exc}"))
    for index, (fraction, scale) in enumerate(perturbations):
        candidate = perturb_weights(model, fraction=fraction, noise_scale=scale, seed=index)
        try:
            value = evaluate(candidate)
            ok = _finite_tensor(value)
            results.append(CheckResult(f"weight_perturbation_{index}", ok, "" if ok else "non-finite output", {"fraction": fraction, "noise_scale": scale}))
        except Exception as exc:  # pragma: no cover - model-specific exceptions
            results.append(CheckResult(f"weight_perturbation_{index}", False, f"{type(exc).__name__}: {exc}"))
    return results


def forward_backward_probe(
    model: nn.Module,
    inputs: Any,
    loss_fn: Callable[[Any], torch.Tensor],
) -> CheckResult:
    """Run a real forward/backward pass and inspect outputs, loss, and grads."""

    model.zero_grad(set_to_none=True)
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            output = model(inputs)
            if not _finite_tensor(output):
                return CheckResult("forward_backward", False, "forward output is non-finite or not a tensor")
            loss = loss_fn(output)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not _finite_tensor(loss):
                return CheckResult("forward_backward", False, "loss is non-finite or not scalar")
            loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        missing = sum(gradient is None for gradient in gradients)
        bad = [gradient for gradient in gradients if gradient is not None and not _finite_tensor(gradient)]
        passed = not bad and missing == 0
        return CheckResult(
            "forward_backward",
            passed,
            "" if passed else f"missing_gradients={missing}, bad_gradients={len(bad)}",
            {"loss": float(loss.detach().item()), "gradient_count": float(len(gradients) - missing)},
        )
    except Exception as exc:  # pragma: no cover - model-specific exceptions
        return CheckResult("forward_backward", False, f"{type(exc).__name__}: {exc}")


def cpu_cuda_probe(
    model_factory: Callable[[], nn.Module],
    inputs: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> CheckResult:
    """Compare identical weights and inputs on CPU and CUDA when available."""

    if not torch.cuda.is_available():
        return CheckResult("cpu_cuda", True, "CUDA unavailable; comparison skipped", {"skipped": 1.0})
    try:
        cpu_model = model_factory().cpu().eval()
        cuda_model = model_factory().cuda().eval()
        cuda_model.load_state_dict(cpu_model.state_dict())
        with torch.no_grad():
            cpu_output = cpu_model(inputs.cpu())
            cuda_output = cuda_model(inputs.cuda()).cpu()
        difference = float((cpu_output - cuda_output).abs().max().item())
        passed = bool(torch.allclose(cpu_output, cuda_output, atol=atol, rtol=rtol))
        return CheckResult("cpu_cuda", passed, "" if passed else "outputs differ beyond tolerance", {"max_abs_diff": difference, "atol": atol, "rtol": rtol})
    except Exception as exc:  # pragma: no cover - depends on CUDA/runtime
        return CheckResult("cpu_cuda", False, f"{type(exc).__name__}: {exc}")


def lifecycle_probe(
    model_factory: Callable[[], nn.Module],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    export: bool = True,
) -> list[CheckResult]:
    """Exercise training, save/load, export, and deployment inference."""

    results: list[CheckResult] = []
    model = model_factory().cpu()
    try:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        finite_loss = bool(torch.isfinite(loss).item())
        results.append(CheckResult("train", finite_loss, "" if finite_loss else "non-finite training loss", {"loss": float(loss.detach().item())}))
    except Exception as exc:
        results.append(CheckResult("train", False, f"{type(exc).__name__}: {exc}"))
        return results
    with tempfile.TemporaryDirectory(prefix="model-quality-") as directory:
        checkpoint = Path(directory) / "model.pt"
        export_path = Path(directory) / "model.pt2"
        export_format = ""
        try:
            torch.save(model.state_dict(), checkpoint)
            saved = checkpoint.is_file() and checkpoint.stat().st_size > 0
            results.append(CheckResult("save", saved, "" if saved else "checkpoint is empty"))
        except Exception as exc:
            results.append(CheckResult("save", False, f"{type(exc).__name__}: {exc}"))
            return results
        if export:
            try:
                if hasattr(torch, "export") and hasattr(torch.export, "save"):
                    exported = torch.export.export(model.eval(), (inputs,))
                    torch.export.save(exported, export_path)
                    export_format = "torch.export"
                else:  # pragma: no cover - compatibility with older PyTorch
                    export_path = Path(directory) / "model.ts"
                    exported = torch.jit.trace(model.eval(), inputs)
                    exported.save(str(export_path))
                    export_format = "torch.jit"
                saved = export_path.is_file() and export_path.stat().st_size > 0
                results.append(CheckResult("export", saved, export_format if saved else "export artifact is empty"))
            except Exception as exc:
                results.append(CheckResult("export", False, f"{type(exc).__name__}: {exc}"))
                return results
        try:
            loaded = model_factory().cpu().eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            loaded.load_state_dict(state)
            with torch.no_grad():
                loaded_output = loaded(inputs)
            ok = _finite_tensor(loaded_output)
            results.append(CheckResult("load", ok, "" if ok else "loaded model produced non-finite output"))
        except Exception as exc:
            results.append(CheckResult("load", False, f"{type(exc).__name__}: {exc}"))
            return results
        if export:
            try:
                if export_format == "torch.export":
                    deployed = torch.export.load(export_path).module()
                else:  # pragma: no cover - compatibility with older PyTorch
                    deployed = torch.jit.load(str(export_path), map_location="cpu").eval()
                with torch.no_grad():
                    deployment_output = deployed(inputs)
                difference = float((loaded_output - deployment_output).abs().max().item())
                ok = _finite_tensor(deployment_output) and torch.allclose(loaded_output, deployment_output)
                results.append(CheckResult("deploy", bool(ok), "" if ok else "deployment output mismatch", {"max_abs_diff": difference}))
            except Exception as exc:
                results.append(CheckResult("deploy", False, f"{type(exc).__name__}: {exc}"))
    return results


def all_passed(results: Iterable[CheckResult]) -> bool:
    """Return whether every supplied check passed."""

    return all(result.passed for result in results)
