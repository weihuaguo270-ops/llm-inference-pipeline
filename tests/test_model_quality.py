import pytest

torch = pytest.importorskip("torch")
from torch import nn

from experiments.model_quality import (
    all_passed,
    cpu_cuda_probe,
    forward_backward_probe,
    inspect_weights,
    lifecycle_probe,
    perturb_weights,
    weight_robustness_probe,
)


def _model():
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def test_weight_integrity_detects_corruption_and_perturbation_is_copy():
    model = _model()
    assert inspect_weights(model).passed
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    assert not inspect_weights(model).passed

    original = _model()
    changed = perturb_weights(original, fraction=1.0, noise_scale=0.25, seed=3)
    assert all(torch.equal(left, right) for left, right in zip(original.parameters(), _model().parameters()))
    assert any(not torch.equal(left, right) for left, right in zip(original.parameters(), changed.parameters()))


def test_weight_robustness_probe_checks_baseline_and_corruption():
    model = _model()
    inputs = torch.randn(3, 4)
    results = weight_robustness_probe(model, inputs)
    assert all_passed(results)
    with torch.no_grad():
        next(model.parameters()).fill_(float("inf"))
    assert not all_passed(weight_robustness_probe(model, inputs))


def test_forward_backward_probe_rejects_non_finite_gradients():
    model = _model()
    inputs = torch.randn(3, 4)
    result = forward_backward_probe(model, inputs, lambda output: output.square().mean())
    assert result.passed

    broken = _model()
    with torch.no_grad():
        next(broken.parameters()).fill_(float("nan"))
    assert not forward_backward_probe(broken, inputs, lambda output: output.mean()).passed

    class NanGradient(torch.autograd.Function):
        @staticmethod
        def forward(_ctx, value):
            return value.clone()

        @staticmethod
        def backward(_ctx, gradient):
            return torch.full_like(gradient, float("nan"))

    class BrokenBackward(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, value):
            return NanGradient.apply(self.linear(value))

    assert not forward_backward_probe(BrokenBackward(), inputs, lambda output: output.mean()).passed


def test_cpu_cuda_probe_is_explicit_when_cuda_is_unavailable():
    result = cpu_cuda_probe(_model, torch.randn(2, 4))
    assert result.passed
    if not torch.cuda.is_available():
        assert result.metrics["skipped"] == 1
    else:
        assert "max_abs_diff" in result.metrics


def test_lifecycle_probe_covers_train_save_load_export_deploy():
    inputs = torch.randn(4, 4)
    targets = torch.randn(4, 2)
    results = lifecycle_probe(_model, inputs, targets, nn.MSELoss())
    assert [result.name for result in results] == ["train", "save", "export", "load", "deploy"]
    assert all_passed(results)
