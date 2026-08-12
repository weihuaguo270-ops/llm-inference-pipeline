import pytest

from experiments.device_policy import resolve_device


def test_cuda_request_can_fall_back_for_development():
    assert resolve_device("cuda", cuda_available=False, require_cuda=False) == "cpu"


def test_required_cuda_fails_closed():
    with pytest.raises(RuntimeError, match="CUDA was required"):
        resolve_device("cuda", cuda_available=False, require_cuda=True)


def test_available_cuda_is_preserved():
    assert resolve_device("cuda", cuda_available=True, require_cuda=True) == "cuda"
