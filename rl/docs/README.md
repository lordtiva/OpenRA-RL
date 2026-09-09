# RL — documentación

Fuente de verdad del agente PPO (AlphaLiteNet v2) para OpenRA mod **RA Aliados**.

`docs/` en la raíz del repo son **imágenes** del proyecto original (`banner.png`, `architecture.png`). Todo el texto técnico vive acá.

Historia de runs, auditorías v1 e informes de revisión: [`_archive/`](_archive/README.md). No son el hello-world.

---

## Empezá acá

| Querés | Doc |
|--------|-----|
| Clonaste el repo y no tenés pesos | [`start/onboard.md`](start/onboard.md) |
| Docker, dash, skirmish, reglas de run | [`start/operacion.md`](start/operacion.md) |
| El agente es RA **Aliados** (lock, catálogo, superpoderes) | [`contract/ra-aliados.md`](contract/ra-aliados.md) |
| Un train por país / soviet / otro mod | [`contract/facciones-mods-roles.md`](contract/facciones-mods-roles.md) |
| Qué hay de naval / air / buildings (P0–P4) | [`contract/ra-completo.md`](contract/ra-completo.md) |

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL
$env:PYTHONPATH=""
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d --build
.\.venv\Scripts\python.exe rl\auto_train.py --scratch --onboard
```

Rebuild C# / Docker es obligatorio después del corte Aliados (lobby + órdenes nuevas). Python solo no cambia el bando.

---

## Contratos (código)

| Doc | Qué fija |
|-----|----------|
| [`contract/ra-aliados.md`](contract/ra-aliados.md) | Lobby `RandomAllies`, catálogo, APC, capture/infiltrate, Chronosphere, patrol, Chrono Tank, `IDENTITY_ITEMS` |
| [`contract/facciones-mods-roles.md`](contract/facciones-mods-roles.md) | Roles agnósticos a país. Un ckpt `ra` Aliados cubre england/france/germany. Soviet = mismo ckpt más adelante. `cnc`/`d2k` = otro ckpt |
| [`contract/ra-completo.md`](contract/ra-completo.md) | P0–P4 **DONE** (macros navy/air, building slots, micro, mapas, obs/dash). Leftovers = tapes / multi-spawn, no chasis |

---

## Diseño (sigue vigente)

| Doc | Qué cubre |
|-----|-----------|
| [`design/filosofia-rl.md`](design/filosofia-rl.md) | Traducción bot→RL: 4 pilares. Reward no es el producto |
| [`design/plan-4-capas.md`](design/plan-4-capas.md) | Roadmap 6–12 meses en 2070+5600X: entorno → BC/SIL → red → self-play |
| [`design/capa2c-identidad-matchup.md`](design/capa2c-identidad-matchup.md) | Capa 2c: A+B shipped. Pointer 2c-C revertido |
| [`design/advance-macro.md`](design/advance-macro.md) | `advance()` con interrupciones |
| [`design/rl-vs-rl.md`](design/rl-vs-rl.md) | Dual bridge + PFSP-RL (`--pfsp-rl`). El log de Run 42 queda como ejemplo |

Diario empírico de Capa 0 (Runs 8–36, 49 kB): [`_archive/runs/13-capa0-status-post-run8.md`](_archive/runs/13-capa0-status-post-run8.md).

---

## Cómo leer (casos)

1. **Train Aliados:** `start/onboard.md` + `start/operacion.md` + `contract/ra-aliados.md`.
2. **Tocar reward:** `design/filosofia-rl.md` + `rl/reward_shaping.py` (fuente). El preset vivo es `eradicate_v4`.
3. **Tocar red / action set:** `design/capa2c-identidad-matchup.md` + `design/plan-4-capas.md`. Un régimen por vez. No crecer `ENABLED_TYPES` el mismo corte que la arch.
4. **Self-play:** `design/rl-vs-rl.md`.
5. **Otro mod:** `contract/facciones-mods-roles.md` — no es este run.

---

## Mapa código ↔ doc

| Tema | Código | Proto / config |
|------|--------|----------------|
| `advance()` | `openra_env/server/bridge_client.py`, `openra_environment.py` | `proto/rl_bridge.proto` |
| Red | `rl/network.py` `AlphaLiteNet` | `rl/obs_encoding.py` `SCALAR_DIM=33` `MAX_UNITS=96` |
| Acciones | `rl/action_adapter.py` | `ActionType` en proto + `openra_env/models.py` |
| Roles / uniques | `rl/roles.py` `IDENTITY_ITEMS`, `rl/allies.py` | — |
| Reward `eradicate_v4` | `rl/reward_shaping.py` + `rl/auto_support.py` | — |
| Lobby Aliados | `RLSessionManager.ResolvePlayerFaction` | `CreateSessionRequest.player_faction` |
| RL-vs-RL | `ExternalBotBridge` + `rl/peer_obs.py` + `rl/pfsp.py` | `peer_commands` / `GetObservation` |

---

## Convenciones

- **Un cambio de régimen por vez** (reward *o* red *o* vocab *o* oponente). Incluye estado latente: no mezclar en el mismo resume.
- **`SCALAR_DIM = 33`** (P4: + naval/air). Ckpts land `in=29` padan. Los de 21/19/16 son otra era — scratch o adapt explícito.
- **Métrica norte:** `wr20` vs el ancla (`beginner` / `easy` / PFSP). Reward medio y `P(win)` son diagnóstico.
- **Type-head append-only:** `patrol` + `support_power` están al final. Resume land = partial load (`adapt_v2_state_dict`).
- **Escala:** antes de tocar red, verificar que el cuello no sea señal/entorno (lección de la era económica, archivada).
