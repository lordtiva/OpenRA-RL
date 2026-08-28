# Live canvas vs latest.pt. Mismos params que auto_train, visor en :8786.
# Uso: .\watch_live.ps1
Set-Location $PSScriptRoot
$env:PYTHONPATH = ""
& .\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live @args
