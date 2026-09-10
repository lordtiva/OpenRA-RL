# Live canvas vs latest.pt. Visor en :8786.
# Training defaults (RandomAllies / Random / a_short) stay unless you opt in.
# Uso:
#   .\watch_live.ps1
#   .\watch_live.ps1 --bot-type easy --no-greedy
#   .\watch_live.ps1 --enemy-faction russia --spawn ne --episodes 1
#   .\watch_live.ps1 --enemy-faction Random --spawn random
# Flags live-only: --enemy-faction, --spawn sw|ne|random, --player-faction (Allies only).
# UI selectors on http://localhost:8786/ POST /api/config for the next episode.
Set-Location $PSScriptRoot
$env:PYTHONPATH = ""
& .\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live --no-war-nudge --no-greedy @args
