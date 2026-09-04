# Live canvas vs latest.pt. Mismos params que auto_train (Run 43: sin war nudge), visor en :8786.
# Uso: .\watch_live.ps1
#      .\watch_live.ps1 --bot-type easy --no-greedy
Set-Location $PSScriptRoot
$env:PYTHONPATH = ""
& .\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live --no-war-nudge @args
