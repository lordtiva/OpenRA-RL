# Lanza el OpenRA de escritorio + el sidecar PPO (por si desactivaste el autostart).
# Uso (desde la raiz del repo):
#   .\play_skirmish.ps1
#   .\play_skirmish.ps1 -Ckpt rl\ckpts\latest.pt -Greedy
param(
    [string]$Ckpt = "rl\ckpts\best.pt",
    [switch]$Greedy
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = ""
$env:OPENRA_RL_AUTOSTART = "0"
$env:OPENRA_RL_ROOT = $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "No está $py — el sidecar PPO necesita el venv del repo."
}

$game = Join-Path $PSScriptRoot "OpenRA\launch-game.cmd"
if (-not (Test-Path $game)) {
    Write-Error "No está OpenRA\launch-game.cmd"
}

$bin = Join-Path $PSScriptRoot "OpenRA\bin\OpenRA.dll"
if (-not (Test-Path $bin)) {
    Write-Host "OpenRA no está compilado. Desde OpenRA\: .\make.cmd all" -ForegroundColor Yellow
}

$pyArgs = @("-m", "rl.play_skirmish", "--attach", "--port", "10001", "--ckpt", $Ckpt)
if ($Greedy) { $pyArgs += "--greedy" }

Write-Host "Sidecar PPO: $py $($pyArgs -join ' ')"
Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $PSScriptRoot -WindowStyle Minimized

Start-Sleep -Seconds 1
Write-Host "Abriendo OpenRA RA. Skirmish → oponente 'PPO Agent'."
Set-Location (Join-Path $PSScriptRoot "OpenRA")
& .\launch-game.cmd Game.Mod=ra
