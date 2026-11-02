# Plan y Arquitectura OpenRA-RL v5 (`Autonomous General & Creative Meta`)

`v4-champion` (`runs/v4/selfplay/checkpoint_best.pt`) consolidó el primer comandante con auto-juego real en OpenRA: alcanzó **1418.4 Elo**, un **Composite Score de 0.817**, y un win rate del **100% vs Hard**, **88.3% vs Brutal** y **98.3% vs Turtle** en `singles.oramap`.

Sin embargo, la auditoría del agente reveló un **techo de cristal** marcado por cuatro limitaciones estructurales:
1. **Barrera de KL Inversa permanente ($D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$):** La penalización matemática hacia el maestro supervisado actúa como barrera infinita cuando $\pi_{\text{ref}}(a) \approx 0$, castigando al agente si intenta construir unidades de Tier 3 (`ctnk`, `qtnk`, `mslo`, `pdox`) no utilizadas por el bot scripted.
2. **Ceguera de habilidades especiales y macro operativo:** El contrato gRPC v4 carecía de mecanismos para ordenar saltos de Chrono Tank (`ctnk`), detonaciones de Demolition Truck (`dtrk`), o captura selectiva de edificios con Ingenieros (`e6`), mezclando especialistas en el *blob* general de `MAIN`.
3. **Sobreajuste espacial a mapa único:** El entrenamiento exclusivo en `singles.oramap` fijó los kernels convolucionales/atencionales a distancias cortas (15–20 celdas), colapsando en mapas abiertos con recursos dispersos.
4. **Lista negra estática `FORBIDDEN_TRAIN_ITEMS`:** Impedía categóricamente el entrenamiento de 14 unidades avanzadas en lugar de condicionarlas a la salud económica.

`v5` (**Autonomous General**) levanta estos techos dotando al agente de **atención espacial basada en Transformer**, dos cabezas de decisión dedicadas (`DECIDE_ABILITY` y `DECIDE_MACRO_OPS`), destacamento C# `SPECOPS`, regularización con **Clipped Reverse KL**, suavizado de etiquetas (**Label Smoothing** $\epsilon = 0.04$), **Discovery Exploiters** al estilo AlphaStar y un pool de mapas dual (`singles.oramap` + `doubles.oramap`).

---

## Fases del Pipeline v5 (`scripts/run_v5_pipeline.py`)

El pipeline de la v5 es completamente modular, reanudable (*resumable*) e instrumentado con telemetría en tiempo real:

```powershell
# Ejecución end-to-end completa:
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage all

# O etapas individuales:
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage warmstart
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage record
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage bc
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage ppo
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage selfplay
.\.venv\Scripts\python.exe scripts/run_v5_pipeline.py --stage eval
```

---

### Fase 5.0 — Warm-Start Transfer Dimensional (`--stage warmstart` / `phase5_0`)
- Transfiere 27 tensores de pesos del Campeón v4 (`runs/v4/selfplay/checkpoint_best.pt`) hacia la nueva arquitectura `CommanderNet v5`.
- Aplica *zero-padding* en la nueva dimensión de entrada del trunk (`304 -> 305`, agregando `ally_building`) y del crítico asimétrico (`325 -> 326`).
- Inicializa con pesos pre-entrenados la cabeza de producción (`prod_head`) y ejército (`army_head`), dejando listas las nuevas cabezas `ability_head` y `macro_ops_head`.
- Salida guardada en `runs/v5/warmstart/model.pt`.

---

### Fase 5.1 — Dataset Supervisado Dual y Entrenamiento `bc_v5` (`--stage record`, `--stage bc` / `phase5_1`)
1. **Grabación DAgger Multi-Mapa (`ra_rl.record`):**
   - Recolección de 1.200 partidas divididas 50/50 entre `singles.oramap` (combate cercano) y `doubles.oramap` (macro-expansión territorial).
   - Alternancia de oponentes: 75% bots `{hard, brutal, turtle}` + 25% auto-juego espejo bilateral RL-vs-RL (`Random` Aliados + Soviéticos).
   - Inyección de $\epsilon = 0.12$ en macro y $\epsilon = 0.08$ en micro.
   - Partidas guardadas en `data/v5_bc/g*.npz` de forma atómica y reanudable.
2. **Entrenamiento `bc_v5` con Label Smoothing (`ra_rl.bc`):**
   - Entrenamiento supervisado durante 25 épocas con `OneCycleLR` ($10^{-3}$) sobre los datos combinados.
   - **Label Smoothing ($\epsilon = 0.04$):** Distribuye masa de probabilidad uniforme sobre todas las acciones legales enmascaradas, evitando que unidades no construidas por el bot salgan con logits de $-12.0$ y permitiendo que PPO las explore al decaer el ancla KL.
   - Modelo óptimo guardado en `runs/v5/bc_v5/model.pt`.

---

### Fase 5.2 — Warm-Start PPO con Clipped Reverse KL (`--stage ppo` / `phase5_2`)
- 20 iteraciones de SMDP-PPO con crítico asimétrico CTDE $V(g, priv)$ y reward shaping PBRS exacto ($\Phi(s') - \gamma \Phi(s)$).
- **Clipped Reverse KL:** Trunca el log-probabilidad de referencia a $-5.0$ para evitar divergencias infinitas:
  $$D_{KL}^{\text{clip}}(\pi_\theta \parallel \pi_{\text{ref}}) = \sum_a \pi_\theta(a) \left( \ln \pi_\theta(a) - \max(\ln \pi_{\text{ref}}(a), -5.0) \right)$$
- Slack temporal que decae $\beta_{KL}(t) \to 0$ tras el tick 2.500, liberando la creatividad táctica del agente en el juego medio y tardío.

---

### Fase 5.3 — Liga de Auto-Juego Híbrida con Discovery Exploiters (`--stage selfplay` / `phase5_3`)
Distribución de rollouts (48 partidas por iteración, 80–100 iteraciones):
1. **30% Espejo en Vivo (`π_θ vs π_θ`):** Recolección lock-step simultánea de `Multi1` y `Multi0` en `singles.oramap` y `doubles.oramap`.
2. **20% Hall of Fame Histórico (`π_θ vs π_snap`):** Muestreo PFSP ($(1 - WR)^{1.5}$) de los últimos 20 snapshots del aprendiz.
3. **25% Exploiters de Bots C# (`hard`, `brutal`, `turtle`):** Prevención de olvido catastrófico contra arquetipos puros de rush, asedio o macro.
4. **15% Ancla Supervisada (`π_θ vs bc_v5`):** Mantiene la solidez en aperturas estándar.
5. **10% Discovery Exploiters ($\pi_{\text{disc}}$):**
   - Agentes especializados con bonus intrínseco (+0.05 a +0.10) condicionado a impacto táctico real (detonación de `qtnk` cerca de valor enemigo, salto de `ctnk`, captura de edificios con `e6`).
   - Cuando un *Exploiter* descubre una variante tecnológica viable que gana partidas, ingresa al Hall of Fame y el Main Agent aprende a dominarla por suma cero sin sufrir reward hacking.

---

### Fase 5.4 — Torneo de Evaluación y Auditoría Tecnológica (`--stage eval` / `phase5_4`)
- **100 partidas H2H vs `v4-champion`** en pool dual (`singles` + `doubles`), midiendo tasa de victoria directa y dominancia macro.
- **40 partidas vs cada bot C#** (`brutal`, `hard`, `turtle`) en ambos mapas.
- **Auditoría de Diversidad Tecnológica:** Verificación del despliegue de unidades Tier 3 (`ctnk`, `qtnk`, `mslo`, `atek`, `stek`) y acciones de `DECIDE_ABILITY` y `DECIDE_MACRO_OPS`.
- Informe consolidado en `runs/v5/tournament/v5_eval_summary.json`.

---

## Telemetría, Monitoreo y TensorBoard

El pipeline v5 cuenta con:
1. **Monitor de Progreso en Vivo (`StageProgressMonitor`):** Muestra barra interactiva, partidas/iteraciones, porcentaje, tiempo transcurrido, tiempo restante estimado (ETA) y hora proyectada de reloj en cada fase.
2. **Stop y Resume Seguro:** Interrupción suave con `Ctrl+C` y reanudación automática idempotente en todas las fases.
3. **Métricas Detalladas en TensorBoard (`runs/v5/selfplay/tb` y `runs/v5/bc_v5/tb`):**
   - `selfplay/elo`: Elo dinámico del aprendiz.
   - `loss/value`, `loss/pi_prod`, `loss/pi_army`: Pérdidas de política y crítico.
   - `entropy/prod`, `entropy/army`, `entropy/coef`: Entropía por canal.
   - `kl/prod`, `kl/army`, `kl/ref_prod`, `kl/ref_army`: Divergencia KL hacia el ancla.
   - `pool_win_rate/*`: Tasa de victorias desglosada por pool (mirror, snapshot, bots, anchor).
   - `faction_win_rate/*`: Balance de victorias por facción (Aliados vs Soviéticos).
   - `behavior/*`: Uso de tanques, artillería, aviación, ataques `all_in`, cosechadoras y habilidades.
   - `eval/*`: Puntuación armónica, win rate vs `bc_v5` y vs cada bot.
