"""package_patcher.py — Builds the distribution patcher package for OpenRA release-20250330.

This script aggregates:
1. Patched C# assemblies (OpenRA.Mods.Common.dll, Grpc, Protobuf)
2. Mod YAML rules and localization (rl-bot.yaml, mod.yaml, fluent rules.ftl)
3. The RL agent runtime (play_skirmish.py or standalone executable, + checkpoint_best.pt)
4. Automated PowerShell installer and uninstaller scripts with backup support
5. Inno Setup configuration for generating an .exe installer
"""

import shutil
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OPENRA_DIR = ROOT_DIR / "OpenRA"
DIST_DIR = ROOT_DIR / "dist" / "OpenRA-RL-Patcher"
ZIP_OUTPUT = ROOT_DIR / "dist" / "OpenRA-RL-Patcher-release-20250330.zip"

BIN_FILES = [
    "OpenRA.Mods.Common.dll",
    "OpenRA.Game.dll",
    "Google.Protobuf.dll",
    "Grpc.AspNetCore.Server.ClientFactory.dll",
    "Grpc.AspNetCore.Server.dll",
    "Grpc.Core.Api.dll",
    "Grpc.Net.Client.dll",
    "Grpc.Net.ClientFactory.dll",
    "Grpc.Net.Common.dll",
    "System.IO.Pipelines.dll",
    "System.Threading.Channels.dll",
]


def clean_dist():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for item in DIST_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except Exception:
            pass
    (DIST_DIR / "bin").mkdir(exist_ok=True)
    (DIST_DIR / "mods" / "ra" / "rules").mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "mods" / "ra" / "fluent").mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "rl_agent").mkdir(parents=True, exist_ok=True)


def copy_binaries():
    print("[1/5] Copying compiled assemblies...")
    bin_src = OPENRA_DIR / "bin"
    for b in BIN_FILES:
        src = bin_src / b
        if src.exists():
            shutil.copy2(src, DIST_DIR / "bin" / b)
            print(f"  + {b}")
        else:
            print(f"  ! Warning: {b} not found in {bin_src}")


def copy_mod_files():
    print("[2/5] Copying mod rules and localization...")
    # rl-bot.yaml
    src_rl_bot = OPENRA_DIR / "mods" / "ra" / "rules" / "rl-bot.yaml"
    if src_rl_bot.exists():
        shutil.copy2(src_rl_bot, DIST_DIR / "mods" / "ra" / "rules" / "rl-bot.yaml")
        print("  + mods/ra/rules/rl-bot.yaml")

    # mod.yaml
    src_mod_yaml = OPENRA_DIR / "mods" / "ra" / "mod.yaml"
    if src_mod_yaml.exists():
        shutil.copy2(src_mod_yaml, DIST_DIR / "mods" / "ra" / "mod.yaml")
        print("  + mods/ra/mod.yaml")

    # fluent rules.ftl
    src_fluent = OPENRA_DIR / "mods" / "ra" / "fluent" / "rules.ftl"
    if src_fluent.exists():
        shutil.copy2(src_fluent, DIST_DIR / "mods" / "ra" / "fluent" / "rules.ftl")
        print("  + mods/ra/fluent/rules.ftl")


def copy_rl_agent():
    print("[3/5] Copying RL sidecar & checkpoint...")
    # Best checkpoint from selfplay
    ckpt_src = ROOT_DIR / "runs" / "v4" / "selfplay" / "checkpoint_best.pt"
    if not ckpt_src.exists():
        ckpt_src = ROOT_DIR / "runs" / "v4" / "selfplay" / "checkpoint_latest.pt"

    if ckpt_src.exists():
        shutil.copy2(ckpt_src, DIST_DIR / "rl_agent" / "checkpoint_best.pt")
        print(f"  + Checkpoint: {ckpt_src.name} -> rl_agent/checkpoint_best.pt ({ckpt_src.stat().st_size // 1024} KB)")

    # Play skirmish script
    shutil.copy2(ROOT_DIR / "ra_rl" / "play_skirmish.py", DIST_DIR / "rl_agent" / "play_skirmish.py")
    print("  + rl_agent/play_skirmish.py")

    # Bundle self-contained ra_env and ra_rl modules
    for mod_name in ["ra_env", "ra_rl"]:
        src_mod = ROOT_DIR / mod_name
        dest_mod = DIST_DIR / "rl_agent" / mod_name
        if dest_mod.exists():
            shutil.rmtree(dest_mod)
        shutil.copytree(
            src_mod,
            dest_mod,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        )
        print(f"  + Bundled module: rl_agent/{mod_name}")

    # Check for standalone compiled sidecar if exists
    sidecar_exe = ROOT_DIR / "dist" / "rl_sidecar.exe"
    if sidecar_exe.exists():
        shutil.copy2(sidecar_exe, DIST_DIR / "rl_agent" / "rl_sidecar.exe")
        print("  + rl_agent/rl_sidecar.exe (Standalone binary)")


def create_installer_scripts():
    print("[4/5] Generating patcher & uninstaller scripts...")
    # patch_openra.ps1
    patcher_ps1 = """# OpenRA RL Patcher Installer (PowerShell)
# Compatible with OpenRA release-20250330
param(
    [string]$OpenRAPath = ""
)

$ErrorActionPreference = "Stop"
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " OpenRA RL Bot Patcher (release-20250330)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Detect OpenRA installation via Registry first, then standard paths
if ([string]::IsNullOrEmpty($OpenRAPath)) {
    try {
        $reg = Get-ItemProperty "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*", "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*" -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -eq "OpenRA" -and $_.InstallLocation } | Select-Object -First 1
        if ($reg -and (Test-Path $reg.InstallLocation)) {
            $OpenRAPath = $reg.InstallLocation
        }
    } catch { }
}

if ([string]::IsNullOrEmpty($OpenRAPath)) {
    $searchPaths = @(
        "$env:LOCALAPPDATA\\OpenRA",
        "${env:ProgramFiles(x86)}\\OpenRA",
        "$env:ProgramFiles\\OpenRA",
        "C:\\Games\\OpenRA",
        "D:\\Games\\OpenRA"
    )
    foreach ($p in $searchPaths) {
        if ((Test-Path (Join-Path $p "OpenRA.Game.exe")) -or (Test-Path (Join-Path $p "OpenRA.exe")) -or (Test-Path (Join-Path $p "RedAlert.exe")) -or (Test-Path (Join-Path $p "OpenRA.Game.dll"))) {
            $OpenRAPath = $p
            break
        }
    }
}

if ([string]::IsNullOrEmpty($OpenRAPath) -or !(Test-Path $OpenRAPath)) {
    Write-Host "No se detectó la instalación de OpenRA automáticamente." -ForegroundColor Yellow
    $OpenRAPath = Read-Host "Por favor ingresa la ruta de instalación de OpenRA (ej. C:\\Program Files\\OpenRA)"
}

if (!(Test-Path (Join-Path $OpenRAPath "mods\\ra"))) {
    Write-Host "Error: No se encontró la carpeta 'mods\\ra' en $OpenRAPath." -ForegroundColor Red
    exit 1
}

# Check administrator elevation if targeting Program Files or protected locations
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$isProtected = ($OpenRAPath -like "*Program Files*") -or ($OpenRAPath -like "*Program Files (x86)*")
if (-not $isAdmin -and $isProtected) {
    Write-Host "Se requieren privilegios de Administrador para modificar: $OpenRAPath" -ForegroundColor Yellow
    Write-Host "Solicitando elevación de permisos (UAC)..." -ForegroundColor Yellow
    $psExe = (Get-Process -Id $PID).Path
    Start-Process $psExe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -OpenRAPath `"$OpenRAPath`"" -Verb RunAs
    exit
}

Write-Host "Instalación detectada en: $OpenRAPath" -ForegroundColor Green

$srcDir = $PSScriptRoot

# 2. Backup existing files
Write-Host "`nCreando copias de respaldo (.bak)..." -ForegroundColor Cyan
$backupFiles = @(
    "OpenRA.Mods.Common.dll",
    "mods\\ra\\mod.yaml",
    "mods\\ra\\fluent\\rules.ftl"
)
foreach ($bf in $backupFiles) {
    $targetFile = Join-Path $OpenRAPath $bf
    $bakFile = "$targetFile.bak"
    if ((Test-Path $targetFile) -and !(Test-Path $bakFile)) {
        Copy-Item $targetFile $bakFile -Force
        Write-Host "  Respaldo: $bf -> $bf.bak" -ForegroundColor DarkGray
    }
}

# 3. Copy binaries & dependencies
Write-Host "`nInstalando archivos parcheados y librerías..." -ForegroundColor Cyan
$binFiles = Get-ChildItem (Join-Path $srcDir "bin")
foreach ($b in $binFiles) {
    Copy-Item $b.FullName (Join-Path $OpenRAPath $b.Name) -Force
    Write-Host "  + $($b.Name)" -ForegroundColor DarkGreen
}

# 4. Copy mod files
Write-Host "`nActualizando reglas del mod Red Alert..." -ForegroundColor Cyan
Copy-Item (Join-Path $srcDir "mods\\ra\\rules\\rl-bot.yaml") (Join-Path $OpenRAPath "mods\\ra\\rules\\rl-bot.yaml") -Force
Copy-Item (Join-Path $srcDir "mods\\ra\\mod.yaml") (Join-Path $OpenRAPath "mods\\ra\\mod.yaml") -Force
Copy-Item (Join-Path $srcDir "mods\\ra\\fluent\\rules.ftl") (Join-Path $OpenRAPath "mods\\ra\\fluent\\rules.ftl") -Force
Write-Host "  + mods\\ra\\rules\\rl-bot.yaml" -ForegroundColor DarkGreen
Write-Host "  + mods\\ra\\mod.yaml" -ForegroundColor DarkGreen
Write-Host "  + mods\\ra\\fluent\\rules.ftl" -ForegroundColor DarkGreen

# 5. Copy RL Agent
Write-Host "`nInstalando agente RL (Sidecar & Checkpoint)..." -ForegroundColor Cyan
$targetAgentDir = Join-Path $OpenRAPath "rl_agent"
if (!(Test-Path $targetAgentDir)) {
    New-Item -ItemType Directory -Path $targetAgentDir | Out-Null
}
Copy-Item (Join-Path $srcDir "rl_agent\\*") $targetAgentDir -Recurse -Force
Write-Host "  + rl_agent copiado exitosamente." -ForegroundColor DarkGreen

Write-Host "`n¡Parche aplicado con éxito!" -ForegroundColor Green
Write-Host "Ahora puedes abrir OpenRA, ir a Escaramuza (Skirmish), y seleccionar 'PPO Agent' como oponente." -ForegroundColor Yellow
Write-Host "Para revertir los cambios, ejecuta: .\\uninstall_openra.ps1 -OpenRAPath `"$OpenRAPath`"" -ForegroundColor Gray
"""
    (DIST_DIR / "patch_openra.ps1").write_text(patcher_ps1, encoding="utf-8-sig")

    # uninstall_openra.ps1
    uninstaller_ps1 = """# OpenRA RL Patcher Uninstaller (PowerShell)
param(
    [string]$OpenRAPath = ""
)

$ErrorActionPreference = "Stop"
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Desinstalador OpenRA RL Bot" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

if ([string]::IsNullOrEmpty($OpenRAPath)) {
    try {
        $reg = Get-ItemProperty "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*", "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*" -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -eq "OpenRA" -and $_.InstallLocation } | Select-Object -First 1
        if ($reg -and (Test-Path $reg.InstallLocation)) {
            $OpenRAPath = $reg.InstallLocation
        }
    } catch { }
}

if ([string]::IsNullOrEmpty($OpenRAPath)) {
    $searchPaths = @(
        "$env:LOCALAPPDATA\\OpenRA",
        "${env:ProgramFiles(x86)}\\OpenRA",
        "$env:ProgramFiles\\OpenRA",
        "C:\\Games\\OpenRA",
        "D:\\Games\\OpenRA"
    )
    foreach ($p in $searchPaths) {
        if ((Test-Path (Join-Path $p "OpenRA.Game.exe")) -or (Test-Path (Join-Path $p "OpenRA.exe")) -or (Test-Path (Join-Path $p "RedAlert.exe")) -or (Test-Path (Join-Path $p "OpenRA.Game.dll"))) {
            $OpenRAPath = $p
            break
        }
    }
}

if ([string]::IsNullOrEmpty($OpenRAPath) -or !(Test-Path $OpenRAPath)) {
    $OpenRAPath = Read-Host "Ingresa la ruta de instalación de OpenRA"
}

# Check administrator elevation if targeting Program Files or protected locations
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$isProtected = ($OpenRAPath -like "*Program Files*") -or ($OpenRAPath -like "*Program Files (x86)*")
if (-not $isAdmin -and $isProtected) {
    Write-Host "Se requieren privilegios de Administrador para restaurar: $OpenRAPath" -ForegroundColor Yellow
    Write-Host "Solicitando elevación de permisos (UAC)..." -ForegroundColor Yellow
    $psExe = (Get-Process -Id $PID).Path
    Start-Process $psExe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -OpenRAPath `"$OpenRAPath`"" -Verb RunAs
    exit
}

Write-Host "Restaurando archivos originales en: $OpenRAPath..." -ForegroundColor Cyan

$backupFiles = @(
    "OpenRA.Mods.Common.dll",
    "mods\\ra\\mod.yaml",
    "mods\\ra\\fluent\\rules.ftl"
)

foreach ($bf in $backupFiles) {
    $targetFile = Join-Path $OpenRAPath $bf
    $bakFile = "$targetFile.bak"
    if (Test-Path $bakFile) {
        Move-Item $bakFile $targetFile -Force
        Write-Host "  Restaurado: $bf" -ForegroundColor Green
    }
}

# Remove added files
$cleanFiles = @(
    "mods\\ra\\rules\\rl-bot.yaml",
    "rl_agent"
)
foreach ($cf in $cleanFiles) {
    $t = Join-Path $OpenRAPath $cf
    if (Test-Path $t) {
        Remove-Item $t -Recurse -Force
        Write-Host "  Eliminado: $cf" -ForegroundColor DarkGray
    }
}

Write-Host "`nOpenRA ha sido restaurado a su versión original." -ForegroundColor Green
"""
    (DIST_DIR / "uninstall_openra.ps1").write_text(uninstaller_ps1, encoding="utf-8-sig")

    # Inno Setup script
    inno_script = """; Inno Setup Script for OpenRA RL Bot Patcher
#define MyAppName "OpenRA RL Bot Patcher"
#define MyAppVersion "release-20250330-v4"
#define MyAppPublisher "OpenRA RL Team"
#define MyAppURL "https://github.com/lordtiva/OpenRA"

[Setup]
AppId={{9F82B7D2-55E8-4B21-9CE1-789D42C8B561}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\\OpenRA
DirExistsWarning=no
EnableDirDoesntExistWarning=no
OutputDir=..\\installer_bin
OutputBaseFilename=OpenRA-RL-Patcher-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "bin\\*"; DestDir: "{app}"; Flags: ignoreversion
Source: "mods\\ra\\rules\\rl-bot.yaml"; DestDir: "{app}\\mods\\ra\\rules"; Flags: ignoreversion
Source: "mods\\ra\\mod.yaml"; DestDir: "{app}\\mods\\ra"; Flags: ignoreversion
Source: "mods\\ra\\fluent\\rules.ftl"; DestDir: "{app}\\mods\\ra\\fluent"; Flags: ignoreversion
Source: "rl_agent\\*"; DestDir: "{app}\\rl_agent"; Flags: ignoreversion recursesubdirs
Source: "uninstall_openra.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\\OpenRA with RL Bot"; Filename: "{app}\\RedAlert.exe"
"""
    (DIST_DIR / "installer.iss").write_text(inno_script, encoding="utf-8")

    # README_INSTALL.md
    readme_text = """# OpenRA RL Bot Patcher (release-20250330)

Este paquete parchea tu instalación oficial de **OpenRA release-20250330** para permitirte jugar partidas en Escaramuza (*Skirmish*) contra el agente de Aprendizaje por Refuerzo (**PPO Agent / Commander v4**).

---

## 🚀 Instalación Rápida (1 Clic)

1. Abre **PowerShell** en esta carpeta.
2. Ejecuta el script de instalación:
   ```powershell
   .\\patch_openra.ps1
   ```
   *El script detectará automáticamente tu instalación de OpenRA, creará copias de seguridad (.bak) de los archivos originales e instalará el agente.*

3. Abre OpenRA normalmente:
   - Ve a **Single Player** -> **Skirmish** (o **Escaramuza**).
   - En la lista de oponentes de IA, selecciona **PPO Agent**.
   - Haz clic en **Start Game**. ¡El bot RL jugará automáticamente en tiempo real!

---

## 🔄 Desinstalación / Restauración

Para revertir todos los cambios y volver al OpenRA vanilla original:
```powershell
.\\uninstall_openra.ps1
```
Tus archivos originales respaldados serán restaurados automáticamente.

---

## 🛠️ Contenido del Paquete

- `bin/`: DLLs parcheadas del motor OpenRA (con soporte para `ExternalBotBridge` y gRPC).
- `mods/ra/`: Reglas YAML y traducciones para registrar al "PPO Agent" en el lobby.
- `rl_agent/`: Runtime del agente de inferencia y pesos del modelo (`checkpoint_best.pt`).
- `installer.iss`: Script de Inno Setup si prefieres compilar un instalador ejecutable `.exe`.
"""
    (DIST_DIR / "README_INSTALL.md").write_text(readme_text, encoding="utf-8")


def create_zip():
    print("[5/5] Creating distribution ZIP archive...")
    ZIP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in DIST_DIR.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(DIST_DIR)
                zf.write(file, arcname)
    print(f"  -> Generated: {ZIP_OUTPUT} ({ZIP_OUTPUT.stat().st_size // 1024} KB)")


def main():
    print("=== Packaging OpenRA RL Patcher ===")
    clean_dist()
    copy_binaries()
    copy_mod_files()
    copy_rl_agent()
    create_installer_scripts()
    create_zip()
    print("=== Packaging Complete ===")


if __name__ == "__main__":
    main()
