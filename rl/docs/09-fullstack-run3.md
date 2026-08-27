# Full-stack Run3 — Coloso con remate (SCALAR 21 + autonomía + w_raze/w_timeout)

> **Hito:** infraestructura definitiva congelada antes del run largo.
> Unifica el plan "full-stack + 3 ajustes" (2026-08-27) sobre la base del Run2 (`08-avance-run2.md`).
> **Un solo cambio de régimen:** `eradicate_v3 → eradicate_v4` + `SCALAR 19→21`.

## 1. Qué cambia respecto a Run2

| Pilar | Cambio | Archivo | Valor |
|-------|--------|---------|-------|
| **C — Lanchester** | `military_ratio = own_army_value / max(visible_est,300) /3` | `rl/obs_encoding.py` `scalar_features()` | `[0,1]` continuo, `visible_est = 400·n_enemies + 1000·n_bldgs` |
| **C — Tech** | `tech_tier = tier/4` (`barr=1, weap=2, atek=3, stek=4`) | `rl/obs_encoding.py` | `[0,1]` |
| **B — Autonomía** | `support_commands()` → `repair hp<35% && cash>500` (máx 2/bloque) + `power_down` si `drained>provided` | `rl/auto_support.py` + `rl/rollout.py:145` + `rl/train.py --auto-support` | 0 decisiones, gratis para PPO |
| **Reward raze** | `w_raze 1.0 → 2.0` (`/2000 valor`) | `rl/reward_shaping.py` `eradicate_v4` | 1 raze 2k = +2.0 (era +1.0 = 25k farmeo) |
| **Reward timeout** | `w_timeout 0.0 → 1.0` | `rl/reward_shaping.py` `finalize()` | `win +8 > incomplete -1 > lose -2.5` |
| **Beacon** | Añadidos `fase2_a` / `fase2_amin160` / `fase2_a_minus` sin sufijo | `rl/obs_encoding.py` `BEACON_BY_MAP` | Evita `beacon=None` a ciegas |

`SCALAR_DIM 19→21` — **última rotura** de `scalar_mlp`. Ckpts con 19 son incompatibles (mismo `size mismatch` que 16→19). Congelado acá.

## 2. Por qué calibrado así (no 3.0 / 2.0)

* `w_raze 2.0` duplica el incentivo sin tapar `w_mining_rate 0.04/1000 + w_garrison 0.005`. `3.0` haría `1 raze = 75k farmeo` y reintroduce el rush 10:1 que se corrigió a 3:1.
* `w_timeout 1.0` rankea sin empatar `incomplete` con `lose`. Con `2.0`, `incomplete -2.0 ≈ lose -2.5` → pesimismo aprendido de vuelta. GAE además lo anula si todos truncan (solo rankea cuando algún episodio gana).

## 3. Cómo se inyecta la autonomía (sin tocar log_prob)

```python
# rollout.py — después de index_to_command_effective()
action, eff = index_to_command_effective(obs, ...)
if auto_support:
    for cmd in support_commands(obs):  # repair/power_down
        action.commands.append(cmd)    # no cambia eff/log_prob del buffer
pending_cmd = action  # 1 strategic + N soporte en el mismo step
```

`support_commands` no escribe en `last_components` ni en el buffer de PPO — solo evita `defense_loss -3.0` / `hold_zero -0.06`.

## 4. Comando del Run3

```powershell
# Archivar Run2 (obligatorio por SCALAR 21)
copy rl\ckpts\metrics.jsonl rl\ckpts\archivo\metrics_run2_$(Get-Date -Format yyyyMMdd_HHmm).jsonl
copy rl\ckpts\latest.pt rl\ckpts\archivo\latest_run2_$(Get-Date -Format yyyyMMdd_HHmm).pt

# Levantar limpio
docker compose down; docker compose up -d --build openra-rl; curl.exe -f http://localhost:8000/health

# Run3 — coloso con remate (SIN --resume, SCALAR 21 es incompatible)
.\.venv\Scripts\python.exe -m rl.train --url http://localhost:8000 --iters 100 --concurrency 12 --max-steps 624 --macro-ticks 80 --lr 1.5e-4 --batch-size 128 --shaper-preset eradicate_v4 --auto-support

# Dashboard
.\.venv\Scripts\python.exe -m http.server 8501  # http://localhost:8501/dashboard.html
```

Currículum: `beginner` hasta `winrate_rolling20 >0.3` y `raze>0` sostenidos → `--bot-type easy` (mismo comando + `--bot-type easy --resume rl/ckpts/latest.pt`).

## 5. Qué mirar en el dashboard (primeras 30 iters)

| Señal | Esperado Run3 | Alarma |
|-------|---------------|--------|
| `action_hist.army_attack_move` | >5% y creciendo | <2% → beacon sigue `None` (revisar `map_name`) |
| `reward_components.raze` | >0 en algún episodio | 0.0 tras 30 iters → `w_raze` insuficiente |
| `reward_components.timeout` | -1.0 en incompletes | 0 → flag no activo |
| `n_buildings.enemy` | baja de ~12 a <10 cuando razea | plano en 12 → no llega a base |
| `scalars military_ratio` | visible en `metrics.jsonl` | — |

## 6. Congelado

`SCALAR_DIM=21` es definitivo. Próximos cambios solo por `w_*` (sin romper `scalar_mlp`) o por `--bot-type`. Ver `08-avance-run2.md` §3 para el diagnóstico que motiva este preset.
