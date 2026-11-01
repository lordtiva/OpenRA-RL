# Plan y Arquitectura OpenRA-RL v4 (`Commander v4` + Self-Play Híbrido)

`v3-final` (`runs/phase4/league_v2/checkpoint_best.pt`, congelado en `main` / tag `v3-final`)
alcanzó **80.0% vs `hard`**, **74.0% vs `brutal`** y **85.0% vs `turtle`** (media armónica
**0.794** en semillas de test `10000–10099`), topándose con tres techos estructurales del
entorno `v3`: un único blob terrestre (`MAIN`) gobernado por retirada difusa frente a bases
fortificadas, ceguera a unidades móviles/cosechadoras y entrenamiento únicamente contra
bots C# estáticos en facciones Aliadas.

`v4` (**Eje B**) levanta esos techos combinando un comandante táctico expresivo con
**self-play real en liga híbrida**.

---

## Fases del pipeline (`scripts/run_v4_pipeline.py`)

### Fase 4.0 — Motor C#, Lock-Step Dual y `SelfPlaySession`
- Sincronización `PostWorldTick` al cierre de cada `WorldTick` en `ExternalBotBridge` y
  `RLSessionManager` (`Multi1:rl-agent,Multi0:rl-agent` en un único `FastAdvance` RPC).
- `Squad.Fearless` (`choice="all_in"`), escuadrón independiente `HARASS` (hasta 4 raiders),
  control externo de `AIR` (`AirIdleState`) y poderes de soporte `DECIDE_SUPPORT`.
- Candidatos `mpspawn` agnósticos al mapa + memoria corta (`300` ticks) de `enemy_harvester`
  y `enemy_mobile` (`MaxTargets = 16`).
- Verificado con `tests/test_selfplay_daemon.py` y `tests/test_daemon.py` (`8/8` verdes).

### Fase 4.1 — `FEATURE_VERSION = 3`, `CommanderNet v4` e Imitación Bilateral + DAgger
- `GLOBAL_DIM` incorpora facción propia (`england`, `france`, `germany`, `russia`, `ukraine`),
  inferencia honesta a la niebla del bando rival (`allies`/`soviet`/`unknown`), estado y
  memoria por escuadrón (`MAIN`, `HARASS`, `AIR`), amenaza antiaérea y `placement_failures`.
- `CommanderNet` añade `tactical_emb` sobre los 11 canales tácticos (`MAIN`, `HARASS`, `AIR`
  + 7 `SUPPORT_POWERS` + fallback) y doble puntuación por candidato (`0..15` ataque normal,
  `16..31` asalto `all_in`, `32 = HOLD`, `33 = KEEP`).
- `ScriptedPolicy` v4 opera bilateralmente con árboles Aliado y Soviético, regla estricta de
  rearme aéreo (`1 aeronave : 1 plataforma`), raiders `HARASS`, asalto `all_in` con masa
  decisiva y disparo contextual de poderes de soporte.
- Grabación de `data/scripted_v4` (75% vs `hard,brutal,turtle` + 25% espejo `RL-vs-RL`) +
  1 ronda DAgger (`data/dagger1_v4`) -> modelo ancla supervisado `runs/v4/bc_v4/model.pt`.

### Fase 4.2 — Warm-Start Multi-Bot PFSP (`ra_rl.league`)
- Calentamiento del crítico asimétrico CTDE `V(g, priv)` y ajuste fino de 30 iteraciones
  contra `{hard, brutal, turtle}` con facciones bilaterales (`Random`), SMDP GAE conjunto y
  PBRS de suma cero (`runs/v4/league_warmstart`).

### Fase 4.3 — Liga Híbrida de Self-Play (`ra_rl.selfplay`)
Distribución de rollouts por iteración (`--pool-shares 0.30,0.20,0.35,0.15`):
1. **30% Espejo en vivo (`π_θ vs π_θ`)**: `SelfPlaySession` recolectando trayectorias
   on-policy de **ambos** bandos (`Multi1` y `Multi0`, doble eficiencia muestral por tick).
2. **20% Hall of Fame (`π_θ vs π_snap`)**: snapshots históricos muestreados con PFSP
   (`(1 - WR)^1.5`) alternando slots `Multi1`/`Multi0` para anular sesgo de spawn.
3. **35% Exploiters C# (`hard`, `brutal`, `turtle`)**: muestreo PFSP para impedir olvido
   cíclico piedra-papel-tijera contra rush, macro o defensas estáticas.
4. **15% Ancla supervisada (`π_θ vs bc_v4`)**: alternando `Multi1`/`Multi0`.

Estabilizadores algorítmicos:
- **Normalización estratificada de ventajas por pool** (`bot_idx` separado para `selfplay`,
  `snapshot`, `bc`, `hard`, `brutal`, `turtle`).
- **Decaimiento coseno dual del ancla KL**: `kl_prod` (`0.04 -> 0.005`) conserva la
  economía/árbol tecnológico mientras `kl_army` (`0.04 -> 0.002`) libera flanqueos
  multi-escuadrón, timings de `all_in` y uso de soporte.
- **Seguimiento Elo en vivo** (anclado en `bc_v4 = 1200`) y selección de `checkpoint_best.pt`
  por media armónica compuesta sobre `(hard, brutal, turtle, vs_bc)`.

---

## Ejecución

```powershell
# Pipeline completo resumible (Fase 4.1 -> Fase 4.2 -> Fase 4.3 -> Eval oficial 10000-10099):
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage all

# O por etapas individuales:
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_1
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_2
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage phase4_3
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage eval
```
