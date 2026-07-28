"""Discover and activate the Visual C++ toolchain for torch.compile."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def _find_vswhere():
    candidate = shutil.which("vswhere.exe")
    if candidate:
        return Path(candidate)
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if not program_files_x86:
        return None
    candidate = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return candidate if candidate.exists() else None


def _find_vcvars64():
    vswhere = _find_vswhere()
    if vswhere is not None:
        result = subprocess.run(
            [
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        installation = result.stdout.strip()
        if installation:
            candidate = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.exists():
                return candidate

    roots = [
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
    ]
    for root in filter(None, roots):
        visual_studio = Path(root) / "Microsoft Visual Studio"
        if not visual_studio.exists():
            continue
        for candidate in visual_studio.glob("**/VC/Auxiliary/Build/vcvars64.bat"):
            return candidate
    return None


def ensure_msvc_environment():
    """Load vcvars64 into this process and return the resolved cl.exe path."""
    compiler = shutil.which("cl.exe")
    if compiler:
        os.environ["VSLANG"] = "1033"
        return compiler

    vcvars64 = _find_vcvars64()
    if vcvars64 is None:
        return None
    command = f'call "{vcvars64}" >nul 2>&1 && set'
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        encoding="mbcs",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value
    compiler = shutil.which("cl.exe")
    if compiler:
        os.environ["VSLANG"] = "1033"
        return compiler

    # Some host processes retain their original PATH casing/value even though
    # vcvars exported the remaining VC variables. Resolve the compiler from
    # VCToolsInstallDir and prepend its directory explicitly.
    tools_dir = os.environ.get("VCToolsInstallDir")
    if tools_dir:
        candidate = Path(tools_dir) / "bin" / "Hostx64" / "x64" / "cl.exe"
        if candidate.exists():
            os.environ["PATH"] = f"{candidate.parent};{os.environ.get('PATH', '')}"
            # Keep Inductor's compiler probes decodable on non-English Windows.
            os.environ["VSLANG"] = "1033"
            return str(candidate)
    return None


def reexec_with_utf8_for_compile():
    """Restart a Windows CLI in UTF-8 mode when MSVC output requires it."""
    if os.name != "nt" or sys.flags.utf8_mode:
        return None
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    return subprocess.call([sys.executable, *sys.argv], env=environment)
