# Operacion — Comandos y reglas de run limpio

> **Actualizado 2026-09-04.** Fuente de verdad de flags del train: `rl/auto_train.py` -> `TRAIN_ARGS`.
> Este doc describe **como operar**; el detalle de cada run esta en `rl/docs/16-…` … `21-…`.

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
| `--onboard` | Curriculum A→B→C para un clone **sin** `.pt`. Primera vez: `--scratch --onboard`. B trae `--bc --bc-teacher-bot beginner`. Doc: `22-onboard.md`. |
| `--onboard-rewind N` | En B: `latest` ← best/iterN, trunca metrics/race, pinnea BC start. Una vez. |
| `--collapse` | *(default)* Watchdog politica muerta + sequia wr20 -> copia `best.pt` -> `latest.pt` + `--reset-opt`. |
| `--no-collapse` | Apaga solo ese watchdog. Siguen cuelgue GPU/daemon y relanzos por crash. |

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

* Lee `rl/ckpts/metrics.jsonl` (append-only mientras corre el train).
* `auto_train` escribe en la **raiz** `rl/ckpts/` para que el dash vea el run vivo; al cerrar una era, archivar con `rl/archive_run.py`.

### 5) Visor live (headless, canvas)

Politica vs bot / vs ckpt, con WebM opcional:

```powershell
.\.venv\Scripts\python.exe -m rl.play_vs_checkpoint_live `
  --ckpt rl\ckpts\latest.pt --bot-type easy --no-greedy --no-war-nudge
```

* Device CUDA si hay. Ckpts **pre arch v1.1** cargan tronco con `strict=False` (cell head Sequential + `unit_pool_proj` nacen frescos; Adam fresco si hay mismatch).
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
* Detalle facciones/roles: `15-facciones-mods-roles.md`.

---

## Arquitectura / obs (snapshot Sep 2026)

| Pieza | Estado |
|-------|--------|
| `SCALAR_DIM` | **25** (21 clasicos + AOA `rel_power/health/speed/strong`). Pad Net2Net en load. |
| Force edge | Reward chico en `eradicate_v4` (`w_force_edge`) si Strong y combate lejos de base. Modulo `rl/force_estimate.py`. |
| Arch v1.1 | Cell head `Conv 296->64 -> SiLU -> Conv3×3->1`; GRU `unit_vec` = proj(`own_mean ‖ own_max ‖ ene_mean`). ~3.0M params. |
| Entity XF | 128 tokens, top-k sparse (`--xf-topk`). |
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
6. **Colapso:** con pesos maduros deja `--collapse`. En scratch temprano suele convenir `--no-collapse` (best@1 con iwr=1.0 pisa aprendizaje; ver Run 42 / doc 16).
7. **BC:** `--bc-start-iter` nunca `0` (train lo trata como unset y en resume reinicia warmup). Usar `1` en scratch.
8. **Onboard:** no mezclar con PFSP/hard. No `--scratch --onboard` a mitad de B/C. A no sale a las 20 iters: hace falta wr20 del *alumno*. Un 4/4 no promociona. Collapse off en A y B. Redo A desde SFT: `--onboard --onboard-rewind 20`. Detalle: `22-onboard.md`.

---

## Currículum (alto nivel)

Detalle historico de cortes: docs `13`–`21`. Regla practica ahora:

| Senal | Accion |
|-------|--------|
| Smoke 20: H sana, sin NaN, wr no a 0 | Seguir el run |
| wr20 ancla decente y incomplete bajando | Subir dificultad / abrir PFSP pool |
| Sequia wr20 con politica **viva** (H ok, no spam) | Revisar si `--collapse` esta matando el run -> `--no-collapse` o endurecer sequia |
| Otro mod (`cnc`/`d2k`) | Otro ckpt; no mezclar con `ra`. Ver `15-facciones-mods-roles.md` |
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

* Runs recientes: `16-rl-vs-rl-run42.md` … `21-run47-map-qsa.md`
* Facciones / roles: `15-facciones-mods-roles.md`
* Filosofia / pilares: `06-filosofia-rl.md`
* Reward: `rl/reward_shaping.py` (`eradicate_v4`)
* Red: `rl/network.py` (arch v1.1)
