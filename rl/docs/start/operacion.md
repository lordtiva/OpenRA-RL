# Operacion — Comandos y reglas de run limpio

> **Actualizado 2026-09-09.** Fuente de verdad de flags del train: `rl/auto_train.py` -> `TRAIN_ARGS`.
> Este doc describe **como operar**. Contrato Aliados: [`../contract/ra-aliados.md`](../contract/ra-aliados.md). Diarios de runs viejos: [`../_archive/runs/`](../_archive/runs/).

Todo se ejecuta desde `C:\Users\lordc\Desktop\OpenRA-RL` con `PYTHONPATH` limpio (el desktop inyecta el suyo y rompe venvs):

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
```

---

## Los comandos del dia a dia

### 1) Contenedor — levantar limpio

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
docker compose down
docker compose up -d --build openra-rl
curl.exe -f http://localhost:8000/health
```

* Esperar `200` antes de lanzar train. Si apuras, `bridge_client` falla el handshake.
* **Aliados** ([`../contract/ra-aliados.md`](../contract/ra-aliados.md)): el lock
  `RandomAllies`, Capture/Infiltrate, patrol, SUPPORT_POWER y Chrono Tank viven
  en C#. `--build` es obligatorio tras pull de este corte; un hot-patch de
  Python no cambia el lobby ni esas órdenes. Type-head creció (`patrol` +
  `support_power`): resume land = partial load.
* Logs: `docker compose logs -f openra-rl`
* Por que recrear: el daemon .NET acumula sesiones; un `down`/`up` resetea el heap.
* Segundo daemon (recomendado en 5600X):  
  `docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d`  
  -> `:8000` + `:8010` (`openra-rl-2`). `auto_train` detecta los que respondan `/health`.

### 2) Train — canónico = `auto_train` (vos lo lanzas)

No hace falta (ni conviene) invocar `rl.train` a mano: el launcher arma URLs, resume, watchdog de cuelgue y (opcional) colapso.

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.venv\Scripts\python.exe rl\auto_train.py
```

Flags del **launcher** (no se reenvian a `rl.train`):

| Flag | Efecto |
|------|--------|
| *(default)* | Resume `rl/ckpts/latest.pt` si existe; si no, seed/`iter*.pt`. |
| `--scratch` | Pesos random; ignora latest/seed (`FORCE_SCRATCH=1`). |
| `--onboard` | Curriculum A→B→C para un clone **sin** `.pt`. Primera vez: `--scratch --onboard`. B trae `--bc --bc-teacher-bot beginner`. Doc: [`onboard.md`](onboard.md). |
| `--onboard-rewind N` | En B o C: `latest` y `best` ← iterN, trunca metrics/race. Desde C vuelve a B. `λ_bc` en el piso. Una vez. |
| `--collapse` | *(default)* Watchdog politica muerta + sequia wr20 -> copia `best.pt` -> `latest.pt` + `--reset-opt`. En `--onboard`, A lo ignora (siempre off); B y C lo usan. |
| `--no-collapse` | Apaga solo ese watchdog (B/C). Siguen cuelgue GPU/daemon y relanzos por crash. |

Ejemplo onboarding (no uses el `TRAIN_ARGS` de PFSP/easy; el overlay de fase lo saca):

```powershell
.\.venv\Scripts\python.exe rl\auto_train.py --scratch --onboard
```

Ejemplos:

```powershell
# Scratch sin restore de best (util al arrancar era nueva)
.\.venv\Scripts\python.exe rl\auto_train.py --scratch --no-collapse

# Resume normal overnight
.\.venv\Scripts\python.exe rl\auto_train.py
```

Log: `rl/auto_train.log`. Al arrancar imprime `collapse_watch=ON|OFF` y `scratch=yes|no`.

**Que hace el watchdog (ademas del colapso):**

* Metrics sin avanzar ~300s + GPU baja -> mata/relanza train (cuelgue Python).
* Markers `Session failed to become ready` en cascada -> recrea el contenedor Docker (cuelgue daemon).
* Train exit ≠ completo -> relanza desde `latest.pt`.
* Train llega a `--iters` del `TRAIN_ARGS` -> sale limpio.

### 3) `TRAIN_ARGS` — regimen vivo

Editar **solo** `TRAIN_ARGS` en `rl/auto_train.py`. Snapshot tipico (2026-09 — verificar el archivo):

* Escenario `a_short`, preset **`eradicate_v4`**
* Macro **50** ticks / **1000** max-steps / gamma **0.995**
* PPO: LR `1e-4`, epochs `1`, clip `0.15`, max-grad-norm `1.0`
* Arch: `--burn-in 8`, `--xf-topk 16`, `--qsa-topk 8 --qsa-block 8`
* `--auto-support` + **`--no-war-nudge`** (APM basico on; guerra la manda PPO)
* Oponente: mirar `TRAIN_ARGS` (`--bot-type`, `--pfsp` / `--pfsp-rl` si estan)
* Capa 1: `--bc` / `--sil` / `--roles-vocab` segun el experimento (comentar/descomentar ahi)
* `--iters` = **ultima iter inclusive** (absoluto). Scratch `--iters 100` -> imprime `1..100`. Resume desde 1141 con `--iters 1161` -> 20 updates.

`--iters` en el help de `rl.train`: ultima iter inclusive, no “cuantas mas” salvo que el start sea 0.

### 4) Dashboard

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
.\.venv\Scripts\python.exe -m http.server 8501
# http://localhost:8501/dashboard.html
```

* Lee `rl/ckpts_v2/metrics.jsonl` por default (`?dir=rl/ckpts` para v1.1). Append-only mientras corre el train.
* `auto_train` escribe en la **raiz** del ckpt-dir para que el dash vea el run vivo; al cerrar una era, archivar con `rl/archive_run.py`.
* **Mix de acciones:** `attack*` suma `attack` + `attack_move` + `army/infantry/vehicle_attack_move`. Eco incluye `harvesters_move`. La linea gris es **solo `no_op`**.
* **Alertas / ultimo collapse:** muestra HH:MM:SS + iter desde `last_collapse.json` (escrito por `auto_train` al restaurar best) o, en vivo sin reiniciar, parseando `rl/auto_train.log` (COLAPSO / SEQUIA wr20).

### 5) Visor live (headless, canvas)

Politica vs bot / vs ckpt, con WebM opcional:

```powershell
.\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live `
  --ckpt rl\ckpts\latest.pt --bot-type easy --no-greedy --no-war-nudge
```

* Device CUDA si hay. Ckpts **pre arch v2** cargan tronco con `strict=False` + soft-adapt XF/fusion/spatial/feats/type (`adapt_v2_state_dict` + capa2c/scalar). Adam fresco si hay mismatch. **A/B:** train v2 en `rl/ckpts_v2/` (no mezclar con v1.1 en `rl/ckpts/`).
* Grabaciones: `rl/ckpts/live_recordings/{episode_id}.webm` cuando el pipeline de MediaRecorder esta activo.

Script helper (si existe en el repo): `rl/watch_live.ps1` — alinear flags a `TRAIN_ARGS` (`--no-war-nudge`, etc.).

### 6) Skirmish humano vs PPO (cliente Windows)

Docker = train headless. Para jugar vos:

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL\OpenRA
.\launch-game.cmd Game.Mod=ra
```

* Lobby Skirmish -> oponente **PPO Agent**. gRPC lobby **:10001** (train/Docker en **:9999** — no pisan).
* Ckpt: `OPENRA_RL_CKPT` (default `best.pt`). Mapa de train: `Singles` / `a_short`.
* PPO entrenado Allies / spawn SW; pone al agente en SW la primera vez.
* Detalle facciones/roles: [`../contract/facciones-mods-roles.md`](../contract/facciones-mods-roles.md).

---

## Arquitectura / obs (snapshot Sep 2026)

### AlphaLiteNet v2 vs v1.1 (**completa**)

* **v1.1:** XF 2 capas d=64 FF=128; `unit_vec` = proj(`own_mean || own_max || ene_mean`); cell head Sequential 1x1->SiLU->3x3; U-Net full ch=96. ~3.0M params. Ckpts en `rl/ckpts/`.
* **v2 completa** (phase 1+2+3):
  1. **Entity XF** 3×4h d=96 FF=256 + pools **Friendly / Enemy / Global** → Fusion MLP → GRU.
  2. **Fog `last_seen`:** `EnemyBeliefStore` por episodio; ghosts con visible/conf/time_since_seen (`UNIT_FEAT_DIM=14`); entran al pool enemigo del XF / F/E/G.
  3. **Multi-select macros:** `infantry_attack_move` / `vehicle_attack_move` / `harvesters_move` (+ `army_attack_move`); adapter emite N× MOVE/ATTACK_MOVE. Single-unit AR intacto para BUILD/PLACE/etc.
  4. **U-Net shrink:** mid=64 en enc/bott/dec1; fmap out sigue **96** (cell/QSA/scatter). Spatial ~1.0M (antes ~1.5M). Total net ~3.0–3.3M.
* **A/B:** train v2 en **`rl/ckpts_v2/`** (`--ckpt-dir rl/ckpts_v2`) vs v1.1 en `rl/ckpts/`. Load: soft-pad XF/fusion/feats/spatial/type-head; Adam fresco si mismatch.

| Pieza | Estado |
|-------|--------|
| `SCALAR_DIM` | **33** (P4: + `own/ene` naval+air; pad Net2Net en load). |
| `UNIT_FEAT_DIM` | **14** (11 + visible/conf/time_since_seen). Ghosts en slots enemigo ≤32. |
| Force edge | Reward chico en `eradicate_v4` (`w_force_edge`) si Strong y combate lejos de base. Modulo `rl/force_estimate.py`. |
| Arch v2 completa | XF 3×96 FF=256; F/E/G fusion; fog ghosts; group macros; U-Net mid64→fmap96. A/B: `rl/ckpts_v2/`. |
| Entity XF | 128 tokens, **3 layers / d=96 / FF=256** (v2); top-k sparse (`--xf-topk`). |
| Map QSA | Bloques 8×8, top-8 (`--qsa-topk` / `--qsa-block`). |
| Burn-in | `--burn-in 8` (GRU sin loss antes del BPTT). |
| Roles | `--roles-vocab` siembra ids fijos de produccion; embedding de entidad ya fijo en `ROLE_VOCAB`. |

Ckpts viejos (cell `Conv 296->1`, SCALAR 21): cargan con missing keys; **scratch** si queres baseline limpia.

---

## Reglas de run limpio

1. **Un cambio de regimen por vez** (reward *o* red *o* vocab *o* oponente). No mezclar “arch v1.1 + PFSP nuevo + BC” sin haber medido cada uno.
2. **Archivar antes de reseedar** la raiz `rl/ckpts/` (`python rl/archive_run.py …`). El dash solo mira la raiz.
3. **`auto_train` lo lanzas vos** (log visible al volver). El asistente no debe lanzarlo en background si queres ver la consola.
4. **No declarar fracaso con pocas iters** — smoke 20 para “no revienta”; juicio de wr con decenas/cientos.
5. **Metrica norte:** `wr20` / era WR vs el ancla (`bot-type` / PFSP anchor). Componentes de reward = diagnostico.
6. **Colapso:** con pesos maduros deja `--collapse`. En scratch temprano suele convenir `--no-collapse` (best@1 con iwr=1.0 pisa aprendizaje; ver [`../design/rl-vs-rl.md`](../design/rl-vs-rl.md)).
7. **BC:** `--bc-start-iter` nunca `0` (train lo trata como unset y en resume reinicia warmup). Usar `1` en scratch.
8. **Onboard:** no mezclar con PFSP/hard. No `--scratch --onboard` a mitad de B/C. A no sale a las 20 iters: hace falta wr20 del *alumno*. Un 4/4 no promociona. Collapse **off en A**, on en B/C. Redo A desde SFT: `--onboard --onboard-rewind 20`. Undo C: `--onboard-rewind N` (vuelve a B). Detalle: [`onboard.md`](onboard.md). Gaps RA completo: [`../contract/ra-completo.md`](../contract/ra-completo.md).

---

## Currículum (alto nivel)

Detalle historico de cortes: [`../_archive/runs/`](../_archive/runs/). Regla practica ahora:

| Senal | Accion |
|-------|--------|
| Smoke 20: H sana, sin NaN, wr no a 0 | Seguir el run |
| wr20 ancla decente y incomplete bajando | Subir dificultad / abrir PFSP pool |
| Sequia wr20 con politica **viva** (H ok, no spam) | Revisar si `--collapse` esta matando el run -> `--no-collapse` o endurecer sequia |
| Otro mod (`cnc`/`d2k`) | Otro ckpt; no mezclar con `ra`. Ver [`../contract/facciones-mods-roles.md`](../contract/facciones-mods-roles.md) |
| Soviet en `ra` | Mismo ckpt + roles; beacon por **slot**, no por bando |

---

## Archivos que toca cada comando

| Comando | Escribe | Lee |
|---------|---------|-----|
| `docker compose up` | daemon `:8000` (gRPC 9999) | compose, `OpenRA/…`, proto |
| `rl/auto_train.py` | log; relanza `rl.train` | `TRAIN_ARGS`, metrics, docker |
| `rl.train` | `rl/ckpts/metrics.jsonl`, `economy_race.jsonl`, `latest.pt` / `iter*.pt`, `best.pt` | obs, network, reward, ckpt |
| `http.server` + dash | — | `metrics.jsonl` -> `dashboard.html` |
| `play_vs_checkpoint_live` | tapes / `live_recordings/` | ckpt, daemon |

---

## Referencias rapidas

* Aliados: [`../contract/ra-aliados.md`](../contract/ra-aliados.md)
* RL-vs-RL: [`../design/rl-vs-rl.md`](../design/rl-vs-rl.md)
* Facciones / roles: [`../contract/facciones-mods-roles.md`](../contract/facciones-mods-roles.md)
* Filosofia / pilares: [`../design/filosofia-rl.md`](../design/filosofia-rl.md)
* Reward: `rl/reward_shaping.py` (`eradicate_v4`)
* Red: `rl/network.py` (arch v2; v1.1 = XF 2x64 + U-Net full-96)
* Diarios de runs: [`../_archive/runs/`](../_archive/runs/)

## P3 map pools

- Catalog: rl/map_catalog.py (official OpenRA titles / display_name).
- Default train / auto_train: a_short (lakes OK; navy gated). Long `fase2_a` archived.
- Small curated pool: `--map-pool official_2p_small` or
  `--onboard-map-pool official_2p_small` (a_short + Doughnut / Bombardment Islands /
  Tournament Island / X-Lake).
- Also: land | mixed | water (water = naval-viable mixed land-water, not pure-water).
- Onboard persists `map_pool` in curriculum.json; unset resume stays a_short.
- Details: [`../contract/ra-completo.md`](../contract/ra-completo.md).
