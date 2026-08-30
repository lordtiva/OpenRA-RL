# Operación — Comandos y reglas de run limpio

Fuente de los 3 comandos PowerShell validados el 2026-08-27. Todo se ejecuta desde `C:\Users\lordc\Desktop\OpenRA-RL` con `PYTHONPATH=` limpio (el desktop inyecta el suyo y rompe venvs).

## Los 3 comandos

### 1) Contenedor — levantar limpio (acumula basura si no se recrea)

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL; docker compose down; docker compose up -d --build openra-rl; curl.exe -f http://localhost:8000/health
```

* Esperar a `200` antes de lanzar el train. Si apurás, `bridge_client` falla el handshake.
* Logs: `docker compose logs -f openra-rl`
* Por qué recrear: el daemon .NET acumula sesiones y GC sin `DestroySession` limpio (ver `fix-endgame-multisesion.md`). Un `down`/`up` resetea el heap.
* Escala opcional (3 daemons): `docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d` → `:8000/:8010/:8020`

### 2) Train — comando canónico del run actual

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL; .\.venv\Scripts\python.exe -m rl.train --url http://localhost:8000 --iters 100 --concurrency 12 --max-steps 624 --macro-ticks 80 --lr 1.5e-4 --batch-size 128 --shaper-preset eradicate_v3
```

Notas:

* `--max-steps 624` es el horizonte medido en `sonda-horizonte.md` (ninguna declaración cabe en 104; 624 ≈ 8 min de juego). No bajar.
* `--macro-ticks 80` → 1 decisión cada 80 ticks (~3.2s a 25 tps). Régimen 2-B.
* `--shaper-preset eradicate_v3` trae `w_refinery_early 2.0 (target 6000)`, `w_first_ore 1.5`, `w_garrison 0.005`, `w_mining_rate 0.04`, `w_cancel 0.15`. Ver `parche-grande-2026-08.md` §5 y `reward_shaping.py`.
* **SCALAR_DIM 19** (desde 2026-08-27): `has_refinery`, `can_afford_proc`, `garrison_ratio` en `obs_encoding.py`. Ckpts con 16 son **incompatibles** (`size mismatch [256,16] vs [256,19]` incluso con `strict=False`). Primer run con 19 debe ir **sin `--resume`**.
* Para archivar el run previo: `copy rl\ckpts\metrics.jsonl rl\ckpts\archivo\metrics_pre19_$(Get-Date -Format yyyyMMdd_HHmm).jsonl`

Variantes:

```powershell
# Resume (solo si el ckpt ya es 19)
.\.venv\Scripts\python.exe -m rl.train --url http://localhost:8000 --iters 200 --resume rl/ckpts/latest.pt --shaper-preset eradicate_v3

# Currículum (promoción por winrate, ver 06-filosofia-rl.md)
.\.venv\Scripts\python.exe -m rl.train --url http://localhost:8000 --iters 100 --bot-type easy --shaper-preset eradicate_v3
.\.venv\Scripts\python.exe -m rl.train --url http://localhost:8000 --iters 100 --bot-type medium --shaper-preset eradicate_v3
```

### 3) Dashboard

```powershell
cd C:\Users\lordc\Desktop\OpenRA-RL; .\.venv\Scripts\python.exe -m http.server 8501
# abrir http://localhost:8501/dashboard.html  (dashboard.html:130)
```

* Lee `rl/ckpts/metrics.jsonl` cada 2s. No tocar `rl/ckpts/metrics.jsonl` mientras corre el train (append-only).
* Si ves `interrupts: {}` eternamente, es bug F6 previo a 2026-08-24; hoy debe venir poblado.

## Reglas de run limpio

1. **Un cambio de régimen por vez** (reward *o* red *o* vocab, nunca juntos). Incluye estado latente: cambiar reward y hacer resume con vocab viejo invalida el experimento (lección era económica).
2. **Todo reinicio pausa el watchdog** (`cron`) y verifica primera iteración con métricas antes de reactivarlo.
3. **No declarar fracaso con pocas iters** — esperar 100-150 antes de juzgar (ruido de PPO).
4. **Métrica norte:** `winrate` vs `beginner` en Escenario A. Reward y `P(win)` heurístico son diagnóstico.
5. **Escala:** no ensanchar red (2M params) antes de winrate >30% en A; la escala amplifica señal, no la crea.

## Criterios de promoción (currículum)

| Condición | Acción |
|-----------|--------|
| `winrate_rolling20 >0.6` en `beginner` (A) sostenido ~200 eps | Pasar a `--bot-type easy`. **Hecho ~900** (wr20≥0.50 ×43). Easy 901–935 = **0/140** (Run 11). Volver a beginner + dest credit; no Capa 2 todavía. Archivar con `rl/archive_run.py` antes de relanzar. |
| wr20 ≥0.40 vs beginner y incomplete <40% (visor: blob llega al NE, no mill mid-map) | Capa 2 (transformer + scatter + `celda\|unidad`). Un PR. No easy hasta entonces. Dest-credit Run 12: crédito OK, incomplete 70% → re-asalto post-recall. |
| Re-asalto no baja incomplete / blob sigue idle en el tent | Rally+stance+sell **hechos corte 923** (con dest pasable). Si el visor no muestra `set_rally_point`, es bug, no “falta Capa 2”. |
| Capa 2 pointer ok (20 iters, `last_push` sigue al sujeto) y wr20 ≥0.40 | Recién ahí `enter_transport`/`unload` (y `guard` de política). No antes. |
| `winrate >0.4` en `easy` + `harvest_edge` positivo | Pasar a `--bot-type medium` |
| `winrate >0.3` en `medium` + `raze` y `n_buildings.enemy` bajando | Evaluar `hard` (=`normal`) |
| Meseta sin `w_mass`/`w_tech` | Considerar Pilar C/D de `06-filosofia-rl.md` (escalar `military_ratio`, PBRS por Tier) |

## Archivos que toca cada comando

| Comando | Escribe | Lee |
|---------|---------|-----|
| `docker compose up` | daemon `:8000` (gRPC 9999) | `OpenRA/mods/ra/rules/ai.yaml`, `proto/rl_bridge.proto` |
| `rl.train` | `rl/ckpts/metrics.jsonl`, `rl/ckpts/economy_race.jsonl`, `rl/ckpts/*.pt` | `rl/obs_encoding.py`, `rl/network.py`, `rl/reward_shaping.py` |
| `http.server` | stderr del server | `rl/ckpts/metrics.jsonl` → `dashboard.html` |
