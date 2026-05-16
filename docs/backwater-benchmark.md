# Backwater Benchmark Workflow

This guide runs the `backwater-hanxin` benchmark with a real OpenRA-RL LLM agent.
The benchmark uses:

- Map: `backwater-battle-hanxin`
- Player: LLM-controlled Han side
- Enemy: OpenRA `beginner` bot, the easiest/simple bot
- Time limit: 5 minutes (`agent.max_time_s: 300`)
- Scenario structure: Han has a bait/main force near the mountain pass and a
  hidden rear detachment near Zhao camp. Zhao's main army starts outside camp,
  already pulled into the pass. Han should pull the bait force back into the
  mountain ambush corridor around `(28,39)-(31,42)`, raid Zhao camp buildings,
  then hit Zhao from front and rear inside the mountains.
- Example model: OpenRouter `inception/mercury-2`

## 1. Create and Activate the Local Python Environment

From the repository root:

```powershell
cd C:\Users\huixu3\code\openrarl\OpenRA-RL

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

Verify:

```powershell
openra-rl --version
```

When you change local Python code, this `.venv` picks it up because the package
is installed editable. The Docker image does not: rebuild `backwater-local` after
agent, server, map, or Dockerfile changes.

## 2. Configure OpenRouter

Set the OpenRouter API key in the same PowerShell session:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-your-real-key"
```

Then pass it explicitly when running:

```powershell
--api-key "$env:OPENROUTER_API_KEY"
```

This avoids accidentally using an old key stored in:

```text
C:\Users\huixu3\.openra-rl\config.yaml
```

To update the saved key instead:

```powershell
openra-rl config
```

Choose OpenRouter and set the model to:

```text
inception/mercury-2
```

## 3. Build a Local Docker Image with the Benchmark Map

If you run the published image, it may not include local map changes. Build a local
image from this repository and use a custom tag:

```powershell
docker build -t ghcr.io/yxc20089/openra-rl:backwater-local .
```

The CLI uses a local image if the requested tag exists. It only pulls from the
internet when the tag is missing locally.

Rebuild this image after any local change that should affect the benchmark,
including:

- `openra_env/agent.py` LLM/tool-call handling fixes
- `openra_env/server/` reset or bridge diagnostics
- `OpenRA/mods/ra/maps/backwater-battle-hanxin/` map files
- `OpenRA/` engine, trait, or protobuf changes
- `Dockerfile` or `.dockerignore`

Check local tags:

```powershell
docker images ghcr.io/yxc20089/openra-rl
```

Expected output should include the local tag:

```text
ghcr.io/yxc20089/openra-rl   backwater-local   ...
```

This repository's Dockerfile builds from the checked-out `OpenRA/` directory, so
local maps and engine changes are included. If the Dockerfile clones a remote
OpenRA branch instead, the build can fail on stale generated protobuf C# stubs or
produce an image that does not include the benchmark map.

The `.dockerignore` file must also allow `OpenRA/` into the build context. It can
still ignore generated folders like `OpenRA/**/bin/`, `OpenRA/**/obj/`, and
`OpenRA/.git/`.

If the build succeeds, continue with `--version backwater-local`. This is the key
flag that makes `openra-rl play` use the local image instead of `latest`.

Map authoring note: multiplayer slot names must stay as `Multi0`, `Multi1`,
etc. In `map.yaml`, the `PlayerReference@Multi0` and `PlayerReference@Multi1`
blocks should also use `Name: Multi0` and `Name: Multi1`. The session launcher
assigns bots by these slot names (`Multi1:rl-agent,Multi0:beginner`); custom
names such as `Zhao AI` or `Han LLM` prevent the slots from being found.

If the slot names are wrong, the OpenRA `rl-bridge.log` will contain messages
like:

```text
Slot 'Multi1' not found in map, skipping bot
Session ... ExternalBotBridge not found
```

## 4. Run the Benchmark

Use the local Docker image tag:

```powershell
openra-rl play `
  --provider openrouter `
  --api-key "$env:OPENROUTER_API_KEY" `
  --model inception/mercury-2 `
  --benchmark backwater-hanxin `
  --version backwater-local `
  --verbose
```

Do not add `--server-url` unless you already started the server yourself. Without
`--server-url`, `openra-rl play` starts/reuses the Docker game server automatically.

You should see the server start line reference the local tag:

```text
Starting game server on port 8000 (ghcr.io/yxc20089/openra-rl:backwater-local)...
```

## 5. Confirm the Correct Map Was Used

After a successful run, OpenRA-RL writes a run artifact under:

```text
C:\Users\huixu3\.openra-rl\runs\
```

Check the latest run:

```powershell
Get-ChildItem "$env:USERPROFILE\.openra-rl\runs" -Filter *.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  Get-Content |
  Select-String '"map_name"'
```

Expected:

```text
"map_name": "backwater-battle-hanxin"
```

If you see `singles.oramap`, the run used the default map. Rebuild the local Docker
image and rerun with `--version backwater-local`.

## 6. Result and Replay Locations

Run artifacts:

```text
C:\Users\huixu3\.openra-rl\runs\
```

Benchmark exports:

```text
C:\Users\huixu3\.openra-rl\bench-exports\
```

For Backwater runs, the bench export includes:

```json
{
  "benchmark": "backwater-hanxin",
  "backwater_score": 42.5,
  "backwater_rubric": {
    "score": 42.5,
    "components": {
      "primary_result": 0.0,
      "lure_and_pincer": 8.2,
      "camp_raid": 9.0,
      "combat_efficiency": 13.3,
      "pincer_pressure": 4.7,
      "han_survival": 5.1,
      "scouting": 2.5,
      "operational_quality": 5.0
    }
  }
}
```

The score is positive even for a loss if Han survives, trades efficiently, scouts,
attacks, or destroys Zhao buildings. A win can reach the highest scores, with a
faster win receiving extra primary-result credit.

Replays:

```text
C:\Users\huixu3\.openra-rl\replays\
```

List replays:

```powershell
openra-rl replay list
```

Copy replays from a running Docker server:

```powershell
openra-rl replay copy
```

## 7. Backwater Rubric

The Backwater score is a 0-100 scenario rubric. It is calculated in OpenRA-RL and
stored in the run artifact plus the bench export. OpenRA-Bench consumes
`backwater_score` when `benchmark` is `backwater-hanxin`; otherwise it uses the
generic composite score.

Historically, Han Xin did not win by matching Zhao in direct frontline strength.
The OpenRA scenario follows the same idea: Han first lures Zhao out into the
mountain pass, then the hidden rear detachment acts like the "2000 cavalry" and
raids Zhao's empty camp. The bait force should withdraw into the mountain ambush
corridor, not all the way back to base. After the camp is disrupted, Han should
collapse Zhao from both sides inside the pass.

- Primary result: up to 40 points for winning against Beginner Zhao AI, with
  faster wins receiving more credit. Losses can receive small survival-time credit.
- Lure and pincer: up to 15 points for initiating attack/lure behavior, pulling
  the fight into the pass, destroying camp buildings, and creating pressure.
- Camp raid: up to 15 points for destroying Zhao camp buildings (`weap`, `fact`,
  `proc`, `powr`, `barr/tent`).
- Combat efficiency: up to 20 points based on K/D cost ratio. `kd_ratio > 1.0`
  is good; `> 1.5` gets full credit for this component.
- Pincer pressure: up to 10 points for Zhao cost damage after the camp raid.
- Han survival: up to 10 points for ending with positive assets, units, and
  buildings.
- Scouting: up to 5 points for explored map percentage.
- Operational quality: up to 5 points for avoiding invalid tool results.

Penalties apply for repeated irrelevant naval-yard/refinery-style loops and for
agent runtime errors. The current telemetry uses proxies for the historical
sequence: attack/lure orders, Zhao building destruction, kills cost, scouting,
and surviving assets.

## 8. Watch a Replay with OpenRA Directly

Use OpenRA directly instead of the browser/VNC replay viewer:

```powershell
.\OpenRA\bin\OpenRA.exe `
  Engine.EngineDir="C:\Users\huixu3\code\openrarl\OpenRA-RL\OpenRA" `
  Game.Mod=ra `
  Game.Platform=Default `
  "Launch.Replay=C:\Users\huixu3\.openra-rl\replays\ra-2026-05-14T182819Z.orarep"
```

Or:

```powershell
dotnet .\OpenRA\bin\OpenRA.dll `
  Engine.EngineDir="C:\Users\huixu3\code\openrarl\OpenRA-RL\OpenRA" `
  Game.Mod=ra `
  Game.Platform=Default `
  "Launch.Replay=C:\Users\huixu3\.openra-rl\replays\ra-2026-05-14T182819Z.orarep"
```

Important: `Engine.EngineDir` must be absolute, or relative to `OpenRA\bin`.
If you use `Engine.EngineDir=.\OpenRA` while launching `OpenRA\bin\OpenRA.exe`,
OpenRA resolves it as `OpenRA\bin\.\OpenRA`, which does not exist.

## Troubleshooting

### 401 Authentication Failed

Pass the key explicitly:

```powershell
openra-rl play `
  --provider openrouter `
  --api-key "$env:OPENROUTER_API_KEY" `
  --model inception/mercury-2 `
  --benchmark backwater-hanxin `
  --version backwater-local `
  --verbose
```

If this works but the command without `--api-key` fails, update the saved config:

```powershell
openra-rl config
```

### Connection Refused on `localhost:8000`

Do not use `--server-url` unless a server is already running. Let `play` start it:

```powershell
openra-rl play `
  --provider openrouter `
  --api-key "$env:OPENROUTER_API_KEY" `
  --model inception/mercury-2 `
  --benchmark backwater-hanxin `
  --version backwater-local `
  --verbose
```

Or start it manually:

```powershell
openra-rl server start --port 8000 --difficulty easy
```

Then connect with:

```powershell
openra-rl play `
  --provider openrouter `
  --api-key "$env:OPENROUTER_API_KEY" `
  --model inception/mercury-2 `
  --benchmark backwater-hanxin `
  --server-url http://localhost:8000 `
  --verbose
```

### Malformed Tool Call Crash

If you see:

```text
'NoneType' object is not subscriptable
```

update to a version that normalizes LLM responses in `openra_env/agent.py`.
The agent should convert malformed tool calls into feedback for the model and
handle `content: null` responses instead of crashing the game loop.

This can happen in either the planning loop or the gameplay loop when a model
returns a malformed tool call such as `function: null` or a message with
`content: null`. With `--verbose`, unexpected agent exceptions print a Python
traceback to make any remaining response-shape issue easier to locate.

### Rebuild After Local Fixes

If you changed local source and are running with Docker, always rebuild before
rerunning the benchmark:

```powershell
docker build -t ghcr.io/yxc20089/openra-rl:backwater-local .
```

Then run with:

```powershell
--version backwater-local
```

Without the rebuild, `openra-rl play` may still use an older image that does not
contain your latest map or Python agent fix.
