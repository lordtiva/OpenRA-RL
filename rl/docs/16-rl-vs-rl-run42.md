# 16 — RL-vs-RL (dual bridge) + Run 42

> **Fecha:** 2026-09-03 · **Branch:** `exp/rl-2026-08-28-grok` · **Seed:** `best.pt` iter 1141 (easy, wr~33.8%, r20 0.5)

## Qué es

Hasta Run 41 el slot enemigo era un bot de `ai.yaml`. **RL-vs-RL** pone dos `ExternalBotBridge` en el mismo `World`:

| Slot | Rol |
|------|-----|
| `Multi1` | Learner (primary). FastAdvance / metrics / PPO |
| `Multi0` | Oponente: bot scripted **o** segundo `rl-agent` con política frozen |

## Bridge (C# / proto)

Archivos en el submodule `OpenRA/`:

- `ExternalBotBridge.cs` — `PlayerSessions[sessionId|player]`; `Sessions[id]` = Multi1; `EnqueueCommandsOnly`; `GetObservation`
- `rl_bridge.proto` — `peer_commands` / `peer_slot` en `FastAdvanceRequest`; RPC `GetObservation`; `player_slot` en obs
- `RLBridgeService.cs` — enruta peer + GetObservation
- `RLSessionManager.cs` — cleanup de `PlayerSessions` al crash

Un solo FastAdvance tiquea el mundo; ambos `ITick` drenan órdenes. Deactivate: solo el primary marca `SessionDone`.

**Requiere imagen Docker nueva** (`Dockerfile.local` copia el submodule y regenera stubs gRPC). Hot-patch de Python en contenedores vivos no alcanza para el C#.

## Python

- `bot_type=rl` → bots `Multi1:rl-agent,Multi0:rl-agent`
- `OpenRAAction.peer_commands` + `OpenRAObservation.peer` (dict fog de Multi0; no confiar solo en `metadata`)
- `rl/peer_obs.py` — `peer_obs_from_metadata`
- `rl/rollout.py` — `opponent_net` actúa desde la obs peer (no entra al traj del learner)
- `rl/train.py` — `--pfsp-rl`; si sample=`rl`, carga frozen `latest|prev20|best` y pasa `opponent_net`
- `rl/pfsp.py` — pool puede incluir `rl`; `pick_rl_ckpt()`

## Run 42 (era en blanco)

Archivo previo: `rl/ckpts/Run 41 (a_short pfsp-bots 1142-1152)/`.

| Knob | Valor |
|------|--------|
| Seed pesos | `latest.pt` ← best-1141 con `iteration=0` → train arranca en **iter 1** |
| `--iters` | 400 |
| PFSP | `--pfsp --pfsp-rl --pfsp-pool medium,rl --pfsp-anchor-prob 0.5` (easy es ancla 50%, no entra al PFSP; beginner afuera) |
| Ancla / north star | `easy` |
| Macro | 50 ticks / max_steps 1000 / γ 0.995 |
| Support | fog scout + fast-2proc + RAID_HOME (flags previos) |
| BC | **off** (política madura; `bc-start-iter=1` reiniciaría λ=1 ~80 iters) |
| SIL | on, λ=0.5 |
| `best.json` | era en blanco (primer batch vs easy → `reason=first`); `best.pt` 1141 queda para restore |

### Promoción `best.pt`

Sigue `rl/best_ckpt.is_strictly_better`: `iter_winrate` → `winrate` era → `wr20` → `viability`.

Con PFSP, el row de métricas / `maybe_update_best` usa solo episodios vs ancla **`easy`**. Vs `rl` / beginner / medium van a `pfsp_stats.json`, no promocionan `best.pt`.

### Lanzar

```
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
.\.venv\Scripts\python.exe rl\auto_train.py
```

(auto_train resume `latest.pt`; no lo lance el asistente si hace falta ver el log.)

### Rebuild Docker (cuando cambie C#)

```
docker compose -f docker-compose.yaml build openra-rl
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d --force-recreate openra-rl openra-rl-2
```

Smoke: reset con `bot_type=rl` + `fase2_a_short.oramap` (map_data); step con `peer_commands`; `observation.peer` no vacío.

## Relación con el plan

Cierra el bloqueo de **Capa 3** en `12-plan-4-capas-siguiente-nivel.md` (“sesiones RL vs RL en el bridge”). PFSP de bots (Run 41) queda como ancla; el pool `rl` es self-play chico de checkpoints.

## Watchdog sequía (post-mortem Run 42)

`auto_train` disparó **SEQUIA wr20** a iters 15, 27 y 87 y restauró `best.pt` (congelado en iter 1, iwr=1.0) sobre `latest.pt`. En esas ventanas la política **seguía viva** (H≈1.2–1.7, no_op bajo): el restore de 87 borró aprendizaje post wins vs medium (57/58/68).

Fix: `drought_should_restore` = sequía wr20 **y** racha sin `policy_still_alive`. Colapso `dead_policy` (attack-spam / no_op-spam / entropy-crash) sigue restaurando solo.

También: `sync_env_into_containers` ahora copia `models.py` + pb2 (evita `VALIDATION_ERROR` tras `force-recreate` con imagen sin campo `peer`).
