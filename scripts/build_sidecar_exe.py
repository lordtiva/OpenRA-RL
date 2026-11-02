"""build_sidecar_exe.py — Compiles the RL skirmish sidecar into a standalone executable.

Usage:
    python scripts/build_sidecar_exe.py

Requirements:
    pip install pyinstaller

Output:
    dist/rl_sidecar.exe
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    print("=== Building Standalone RL Sidecar Binary ===")
    if not check_pyinstaller():
        print("PyInstaller is not installed in the active environment.")
        print("To build a standalone executable for end users without Python, install it via:")
        print("    pip install pyinstaller")
        print("\nThen re-run this script.")
        return 1

    entry_script = ROOT_DIR / "ra_rl" / "play_skirmish.py"
    dist_dir = ROOT_DIR / "dist"
    build_dir = ROOT_DIR / "build_pyinstaller"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "rl_sidecar",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--paths",
        str(ROOT_DIR),
        "--hidden-import",
        "torch",
        "--hidden-import",
        "numpy",
        "--hidden-import",
        "grpc",
        "--hidden-import",
        "ra_env",
        "--hidden-import",
        "ra_rl",
        str(entry_script),
    ]

    print(f"Running PyInstaller command:\n{' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        exe_path = dist_dir / "rl_sidecar.exe"
        print(f"\n[SUCCESS] Standalone executable created: {exe_path}")
        print("Run `python scripts/package_patcher.py` to bundle it into the installer package.")
    else:
        print(f"\n[ERROR] PyInstaller failed with exit code {res.returncode}")

    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
