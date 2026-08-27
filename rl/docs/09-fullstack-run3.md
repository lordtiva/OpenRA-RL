# Full-stack Run3 — Coloso con remate (SCALAR 21 + autonomía + w_raze/w_timeout)

> **Hito:** infraestructura definitiva congelada antes del run largo.
> Unifica el plan "full-stack + 3 ajustes" (2026-08-27) sobre la base del Run2 (`08-avance-run2.md`).
> **Un solo cambio de régimen:** `eradicate_v3 → eradicate_v4` + `SCALAR 19→21`.
> **Actualizado post-ejecución:** resultados empíricos Run1 vs Run2 vs Run3 (2026-08-27/28) confirman el salto militar.

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
Fix post-auditoría Run3: `_raze_by_value` incluye `eradicate_v4` y `closing=True` en `rollout.py:304` (commit `85600e1`).

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

## 6. Resultados empíricos: Run1 vs Run2 vs Run3

> Fuente: análisis comparativo entregado 2026-08-28 sobre `metrics.jsonl` de los tres runs. Confirma el salto cualitativo del full-stack.

### 6.1 Tabla comparativa de evolución

| Métrica / Comportamiento | Run1 (Original) | Run2 (Económico / BPTT) | **Run3 (Full Stack / Asalto)** |
|---|---|---|---|
| **Recolección (`own_harvest_per_1k`)** | `0.0 $/kt` (Muerte total) | `150 – 550 $/kt` (Despertar económico) | **`250 – 600 $/kt` (Economía sólida y continua)** |
| **Explotación `cancel_production`** | Masiva (30-40% acciones) | Erradicada (<1% acciones) | **Erradicada (0-3 por iteración)** |
| **Destrucción edificios (`raze`)** | `0.0` (Inexistente) | `0.0 – 0.15` (Esporádico) | **`+0.5 a +2.4` 💥 (Asedio constante)** |
| **Acciones Combate (`attack + move`)** | <30 o espasmos ciegos | 50 – 150 | **`400 – 900+` ⚔️ (Presión militar activa)** |
| **Patrimonio en partidas largas (`own`)** | Colapso (<$4k) | $30k – $60k (Turtling masivo) | **`$50k – $81k` 🏰 (Dominio total)** |
| **Retorno Medio Episodio** | Negativo severo (-6 a -9) | Estable (-4 a -1) | **Positivo en picos (+1.1 a +4.2) 📈** |
| **Resultado Formal** | Derrotas rápidas (tick 8k) | `lose` / `incomplete` | `lose` / `incomplete` (con ventaja abrumadora) |

### 6.2 Run2 — El despertar económico (coloso pacifista)

* Solucionado: `w_cancel` + `obs_encoding` liquidaron el bucle parásito de cancelaciones.
* `first_ore +0.75 a +1.5` + refinería dispararon la recolección a `300-540 $/kt`.
* **Nuevo mínimo local de Run2:** sin morir de hambre y con reward por guarnición/minería, el agente aprendió a construir bases gigantes ($50k-60k en iter 61 y 72) pero se quedó en **"SimCity Defensivo"** — casi no atacaba (`raze: 0.0`), esperando el límite `51,792 ticks` (`incomplete`).

### 6.3 Run3 — El despertar militar y la ofensiva real

Combinación `auto_support` + `SCALAR 21` (Lanchester + tech tier) + `w_raze 2.0` produjo **el salto cualitativo más grande del proyecto**:

**A. La red aprendió a demoler la base rival (`raze` explotó):**
* Iter 69: `raze: +2.40` (pico histórico, combate `+0.41`, retorno `+4.22`)
* Iter 73: `raze: +1.58`
* Iter 80: `raze: +2.22` (con 602 `attack_move` + 181 `attack`)
* Iter 85: `raze: +1.58`
* Iter 87: `raze: +1.25`

**B. Densidad militar y asalto agresivo:**
* Iter 69: 384 `attack_move` + 101 `attack`
* Iter 80: 602 `attack_move` + 181 `attack`
* Iter 87: 615 `attack_move` + 186 `attack`
* Iter 96: **863 `attack_move` + 130 `attack`** (¡casi 1,000 órdenes de ataque por tanda!)

**C. Dominación absoluta del mapa (`sup_exact: true`, espectador):**
* Iter 80 Ep4: Propio **`$81,608` vs Rival `$13,463`** (lead ratio `+1.0`)
* Iter 83 Ep4: Propio **`$68,800` vs Rival `$7,100`**
* Iter 93 Ep3/4: Propio **`$44,350 / $47,975` vs Rival `$14,200 / $15,133`**

Ver `08-avance-run2.md` §2 para la progresión económica y `12-plan-4-capas-siguiente-nivel.md` Capa 0 para el contexto de victoria.

### 6.4 Diagnóstico fundamental: ¿por qué sigue `winrate 0.0`?

> **Esto confirma al 100% la tesis de la Capa 0 del Documento 12:** el cuello ya no es el aprendizaje, es el **criterio de finalización del entorno**.

1.  Regla de victoria del motor: OpenRA solo declara `win` si **absolutamente todos** los actores enemigos (incluido un perro escondido en niebla o un silo en la esquina) son destruidos.
2.  El agente barre la base principal pero no "barre el mapa": tras demoler la base bot, el rival deja de ser amenaza pero la red no tiene motivo para rastrear restos dispersos en 51k ticks → `incomplete` por tiempo.
3.  El término `w_win = +8.0` sigue sin cobrarse: a pesar de dominar militarmente, la red nunca recibe el premio terminal porque la partida no se cierra.

## 7. Próximo paso obligatorio (Capa 0 del plan)

El entrenamiento demostró que la red **sabe cosechar, sabe guarnecerse y sabe atacar en masa**. Para convertir palizas `$80k vs $7k` en victorias declaradas (`winrate >80%`):

1.  **Declaración temprana de victoria en C#/Bridge:** si el enemigo no tiene edificios de producción vivos (`n_buildings_production == 0`) o su patrimonio cae por debajo del 10% del tuyo durante 500 ticks → declarar `win` inmediato.
2.  **Barrido de asalto sostenido:** cuando el `beacon` principal sea destruido, redirigir `army_attack_move` a esquinas/celdas no exploradas para limpiar restos.

Una vez habilitado el cierre de partida en el motor, este mismo agente debería pasar de **0% a 60-70% de victorias** sin tocar hiperparámetros ni red. Ver `12-plan-4-capas-siguiente-nivel.md` Capa 0 para el detalle.

## 8. Congelado

`SCALAR_DIM=21` es definitivo. Próximos cambios solo por `w_*` (sin romper `scalar_mlp`) o por `--bot-type`. Ver `08-avance-run2.md` §3 para el diagnóstico que motiva este preset y `12-plan-4-capas-siguiente-nivel.md` para el roadmap post-Run3.
