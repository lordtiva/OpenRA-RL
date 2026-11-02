"""package_rl_edition.py — Package standalone portable OpenRA - RL Edition (Windows x64).

Creates dist/OpenRA-RLEdition-winportable:
- Self-contained .NET 8 runtime (OpenRA.exe, coreclr.dll, all BCL DLLs)
- RL-enabled OpenRA.Mods.Common.dll with gRPC & ONNX runtime integration
- Standalone zero-dependency rl_sidecar.exe (compiled with PyInstaller + ONNX)
- Exported ONNX models (commander_item.onnx, commander_army.onnx)
- Mods (ra, common, cnc, d2k) and GLSL shaders
- Desktop shortcut creator (Create-Desktop-Shortcut.ps1) and RedAlert-RL.bat
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
TARGET_DIR = DIST_DIR / "OpenRA-RLEdition-winportable"
PUBLISH_DIR = ROOT / "OpenRA" / "publish_win64"
BIN_DIR = ROOT / "OpenRA" / "bin"
MODS_DIR = ROOT / "OpenRA" / "mods"
GLSL_DIR = ROOT / "OpenRA" / "glsl"
PROGRAM_FILES_OPENRA = Path(r"C:\Program Files\OpenRA")


def clean_target():
    print(f"Cleaning {TARGET_DIR}...")
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)


def copy_runtime_and_binaries():
    print("Copying self-contained .NET 8 OpenRA binaries...")
    if not PUBLISH_DIR.exists():
        raise FileNotFoundError(f"Published directory not found: {PUBLISH_DIR}")

    # 1. Copy publish_win64 base files
    for item in PUBLISH_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, TARGET_DIR / item.name)

    # 2. Overlay freshly built OpenRA assemblies & dependencies from bin
    print("Overlaying freshly compiled OpenRA assemblies and dependencies from bin/...")
    skip_files = {"OpenRA.runtimeconfig.json", "OpenRA.deps.json", "OpenRA.exe", "OpenRA.dll"}
    for ext in ("*.dll", "*.pdb", "*.deps.json", "*.runtimeconfig.json", "*.config"):
        for item in BIN_DIR.glob(ext):
            if item.name not in skip_files:
                shutil.copy2(item, TARGET_DIR / item.name)

    # 3. Copy native Windows dependencies from Program Files if missing
    native_extras = [
        "lua51.dll",
        "ra.ico",
        "cnc.ico",
        "d2k.ico",
        "IP2LOCATION-LITE-DB1.IPV6.BIN.ZIP",
    ]
    if PROGRAM_FILES_OPENRA.exists():
        for name in native_extras:
            src = PROGRAM_FILES_OPENRA / name
            dst = TARGET_DIR / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    # 4. Global mix database
    gmix = ROOT / "OpenRA" / "global mix database.dat"
    if gmix.exists():
        shutil.copy2(gmix, TARGET_DIR / "global mix database.dat")
    elif (PROGRAM_FILES_OPENRA / "global mix database.dat").exists():
        shutil.copy2(PROGRAM_FILES_OPENRA / "global mix database.dat", TARGET_DIR / "global mix database.dat")

    # 5. GLSL shaders
    if GLSL_DIR.exists():
        shutil.copytree(GLSL_DIR, TARGET_DIR / "glsl", dirs_exist_ok=True)


def copy_mods():
    print("Copying mods (ra, common, cnc, d2k)...")
    target_mods = TARGET_DIR / "mods"
    target_mods.mkdir(parents=True, exist_ok=True)

    for mod_name in ["ra", "common", "cnc", "d2k"]:
        src_mod = MODS_DIR / mod_name
        if src_mod.exists():
            print(f"  Copying mod: {mod_name}")
            shutil.copytree(src_mod, target_mods / mod_name, dirs_exist_ok=True)


def copy_rl_sidecar_and_models():
    print("Copying standalone rl_sidecar.exe and ONNX models...")
    sidecar_src = DIST_DIR / "rl_sidecar.exe"
    if not sidecar_src.exists():
        raise FileNotFoundError(f"rl_sidecar.exe not found at {sidecar_src}. Please compile it first.")

    # Copy to root and to rl_agent/ folder
    shutil.copy2(sidecar_src, TARGET_DIR / "rl_sidecar.exe")
    rl_agent_dir = TARGET_DIR / "rl_agent"
    rl_agent_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar_src, rl_agent_dir / "rl_sidecar.exe")

    # Copy ONNX models
    onnx_src = ROOT / "onnx_models"
    if onnx_src.exists():
        shutil.copytree(onnx_src, TARGET_DIR / "onnx_models", dirs_exist_ok=True)
        shutil.copytree(onnx_src, rl_agent_dir / "onnx_models", dirs_exist_ok=True)


def create_launchers():
    print("Generating launcher scripts and shortcut installers...")

    # RedAlert-RL.bat
    bat_content = """@echo off
title OpenRA - Red Alert (RL Edition)
cd /d "%~dp0"
start "" "%~dp0OpenRA.exe" Game.Mod=ra %*
"""
    (TARGET_DIR / "RedAlert-RL.bat").write_text(bat_content, encoding="utf-8")

    # Create-Desktop-Shortcut.ps1
    ps1_content = """# PowerShell script to create Desktop shortcut for OpenRA - RL Edition
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$ShortcutPath = Join-Path $Desktop "OpenRA - Red Alert (RL Edition).lnk"

$TargetExe = Join-Path $PSScriptRoot "OpenRA.exe"
$IconPath = Join-Path $PSScriptRoot "ra.ico"
if (-not (Test-Path $IconPath)) { $IconPath = $TargetExe }

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetExe
$Shortcut.Arguments = "Game.Mod=ra"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.IconLocation = "$IconPath,0"
$Shortcut.Description = "OpenRA Red Alert - RL Edition (Zero Python/Docker Dependency)"
$Shortcut.Save()

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Acceso directo creado exitosamente en tu Escritorio:" -ForegroundColor Green
Write-Host "   $ShortcutPath" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
"""
    (TARGET_DIR / "Create-Desktop-Shortcut.ps1").write_text(ps1_content, encoding="utf-8")

    # LEEME.txt
    readme_content = """===============================================================
       OpenRA - Red Alert (RL Edition) - Version Portable
===============================================================

Esta versión portátil incluye el agente de Aprendizaje por Refuerzo (RL Bot / PPO)
totalmente integrado y optimizado para ejecutarse en Windows de forma nativa.

CARACTERISTICAS:
- Cero dependencias: NO requiere Docker, NO requiere instalar Python ni CUDA.
- Inferencia neuronal mediante ONNX Runtime integrada.
- Totalmente independiente: No modifica ni rompe tu instalación oficial de OpenRA.
- Detección automática de contenido de audio y mapas existentes en %APPDATA%\\OpenRA.

COMO JUGAR:
1. Para crear un acceso directo en tu Escritorio, haz clic derecho sobre
   'Create-Desktop-Shortcut.ps1' y selecciona 'Ejecutar con PowerShell'.
2. O simplemente ejecuta 'RedAlert-RL.bat' o 'OpenRA.exe Game.Mod=ra'.
3. En el menú principal, entra a 'Escaramuza' (Skirmish).
4. En cualquier slot de IA, selecciona 'PPO Agent' (o Bot RL).
5. ¡Inicia la partida! El agente RL jugará en tiempo real contra ti o contra otras IAs.

Disfruta de la partida.
"""
    (TARGET_DIR / "LEEME.txt").write_text(readme_content, encoding="utf-8")


def create_zip_archive():
    zip_path = DIST_DIR / "OpenRA-RLEdition-release-20250330-winportable.zip"
    print(f"Creating archive {zip_path.name}...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(TARGET_DIR):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(DIST_DIR)
                zf.write(full_path, rel_path)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Archive created: {zip_path} ({size_mb:.1f} MB)")


def main():
    print("==========================================================")
    print(" Packaging OpenRA - RL Edition (Windows x64 Portable)")
    print("==========================================================")
    clean_target()
    copy_runtime_and_binaries()
    copy_mods()
    copy_rl_sidecar_and_models()
    create_launchers()
    create_zip_archive()
    print("Packaging complete!")


if __name__ == "__main__":
    main()
