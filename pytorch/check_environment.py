"""Report CUDA, SDPA and torch.compile prerequisites."""

import importlib.util
import platform
import sys

import torch


def main():
    print("PyTorch inference environment")
    print(f"  Python:              {sys.version.split()[0]}")
    print(f"  Platform:            {platform.platform()}")
    print(f"  PyTorch:             {torch.__version__}")
    print(f"  CUDA runtime:        {torch.version.cuda or 'not compiled'}")
    print(f"  CUDA available:      {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability:  {torch.cuda.get_device_capability(0)}")
        print(
            f"  Flash compiled:      "
            f"{torch.backends.cuda.is_flash_attention_available()}"
        )
        print(
            f"  Efficient SDPA:      "
            f"{torch.backends.cuda.mem_efficient_sdp_enabled()}"
        )

    triton_available = importlib.util.find_spec("triton") is not None
    print(f"  Triton available:    {triton_available}")

    if platform.system() == "Windows":
        from .windows_toolchain import ensure_msvc_environment

        compiler = ensure_msvc_environment()
        print(f"  MSVC compiler:       {compiler or 'not found'}")
        print(f"  Python UTF-8 mode:   {bool(sys.flags.utf8_mode)}")

    if torch.cuda.is_available() and not triton_available:
        print(
            "\nCUDA execution is ready, but CUDA torch.compile is unavailable "
            "without a compatible Triton build. Use WSL2/Linux for the "
            "official toolchain, or validate a matching triton-windows build."
        )


if __name__ == "__main__":
    main()
