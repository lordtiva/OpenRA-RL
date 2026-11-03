# Informe Final de Implementación y Evaluación — OpenRA-RL v5 (Autonomous General)

> **Fecha:** 3 de Noviembre de 2026  
> **Rama Git:** `v5` (Repositorio raíz y submódulo `OpenRA/`)  
> **Pool de Mapas Oficial:** Dual 50/50 (`singles.oramap` + `doubles.oramap`)  
> **Semillas Reservadas de Torneo Oficial:** `50000–50099` (nunca vistas en entrenamiento, BC ni RL)  
> **Checkpoint Campeón:** [`runs/v5/selfplay/checkpoint_best.pt`](file:///c:/Users/lordc/Desktop/OpenRA-RL/runs/v5/selfplay/checkpoint_best.pt) (Iteración 80, **1612.4 Elo**)  
> **Resúmenes Reproducibles:** [`docs/v5-evals.jsonl`](v5-evals.jsonl) y [`runs/v5/tournament/v5_eval_summary.json`](file:///c:/Users/lordc/Desktop/OpenRA-RL/runs/v5/tournament/v5_eval_summary.json)  
> **Telemetría TensorBoard:** [`runs/v5/selfplay/tb/`](file:///c:/Users/lordc/Desktop/OpenRA-RL/runs/v5/selfplay/tb/) y [`runs/v5/bc_v5/tb/`](file:///c:/Users/lordc/Desktop/OpenRA-RL/runs/v5/bc_v5/tb/)

---

## 1. Resumen Ejecutivo de Hitos

La versión **v5 (Autonomous General)** representa el mayor salto cualitativo en la historia del proyecto OpenRA-RL. Al romper la ceguera de habilidades tácticas mediante el contrato gRPC v5 (`DECIDE_ABILITY`), incorporar micro-macro operativo (`DECIDE_MACRO_OPS`), el destacamento especializado `SPECOPS` en el motor C#, la atención espacial Cross-Attention de 17 tokens invariante a mapas, y regularización adaptativa de KL Inversa truncada con Discovery Exploiters, **v5 demolió al Campeón v4 con un 92.0% de Win Rate** y escaló hasta **1612.4 Elo**.

| Criterio de Éxito / Métrica | Objetivo Plan v5 | Resultado Oficial (`50000–50099`) | Estado |
|---|---|---|:---:|
| **Liga Self-Play Elo** | $\ge 1550.0$ Elo | **`1612.4 Elo`** (+194.0 Elo vs v4) | ✅ **SUPERADO** |
| **Head-to-Head vs v4 Campeón (100 partidas)** | $\ge 75.0\%$ | **`92.0%`** (92W / 6L / 2T) | ✅ **SUPERADO** |
| **Intervalo Wilson 95% (H2H vs v4)** | $\ge 68.0\%$ | **`[85.0%, 95.9%]`** | ✅ **SUPERADO** |
| **Generalización Espacial en Pool Dual** | Pool 50/50 sin colapso | **88.0% singles / 96.0% doubles** | ✅ **SUPERADO** |
| **Win Rate vs `bc_v5` (Anchor Supervisado)** | $\ge 85.0\%$ | **`100.0%`** (40W / 0L / 0T) | ✅ **SUPERADO** |
| **Win Rate vs Bot `hard` (40 partidas)** | $\ge 75.0\%$ | **`82.5%`** (33W / 6L / 1T) | ✅ **SUPERADO** |
| **Win Rate vs Bot `turtle` (40 partidas)** | $\ge 65.0\%$ | **`70.0%`** (28W / 2L / 10T) | ✅ **SUPERADO** |
| **Win Rate vs Bot `brutal` (40 partidas)** | $\ge 30.0\%$ | **`42.5%`** (17W / 22L / 1T) | ✅ **SUPERADO** |
| **Varianza Explicada del Crítico ($R^2$)** | $\ge 0.900$ | **`0.9701`** (Pico 0.9776) | ✅ **SUPERADO** |
| **Tasa de Incompletas en H2H Neural** | $\le 5.0\%$ | **`2.0%`** (2 / 100 partidas) | ✅ **CONTROLADO** |
| **Estabilidad y Determinismo** | 0 fallos de daemon | **0 crash / 0 errores** en >6.000 partidas | ✅ **SUPERADO** |

---

## 2. Inspección Detallada de TensorBoard (`runs/v5/selfplay/tb/`)

La telemetría de TensorBoard documenta los 80 ciclos de auto-juego (3.840 partidas colectadas + 320 partidas de validación por checkpoints) y muestra un comportamiento de optimización estable, sin colapsos de divergencia ni saturación de gradientes.

### 2.1 Trayectoria de Elo y Coeficientes de Optimización

```
Elo 1650 ┤                                      ╭────╮
Elo 1600 ┤                                 ╭────╯    ╰────────── (1612.4)
Elo 1550 ┤                            ╭────╯
Elo 1500 ┤                       ╭────╯
Elo 1400 ┤                 ╭─────╯
Elo 1300 ┤          ╭──────╯
Elo 1200 ┤────╮╭────╯
Elo 1130 ┤    ╰╯
         └────┬──────────────┬──────────────┬──────────────┬─────────
            Iter 1         Iter 20        Iter 40        Iter 60    Iter 80
```

* **Evolución del Elo (`selfplay/elo`):**
  * Iteración 1: `1226.4`
  * Mínimo transitorio (Iteración 4): `1135.6` (fase de adaptación inicial a la duplicación de tokens espaciales en `doubles.oramap`).
  * Iteración 40: `1528.9` (cruce del umbral histórico de v4).
  * Iteración 60 (Pico): `1657.6`.
  * Iteración 80 (Consolidación final): **`1612.4`**.
* **Decaimiento de Regularización KL (`selfplay/kl_coef_*`):**
  * `kl_coef_prod`: Iniciado en `0.0400` y decaído linealmente hasta `0.0050`.
  * `kl_coef_army`: Iniciado en `0.0400` y decaído hasta `0.0020`.
* **Deriva Controlada del Maestro (`kl/ref_*`):**
  * `kl/ref_prod` subió suavemente de `0.0000` a `0.0803` (la macroeconomía se mantiene disciplinada y cercana al canon óptimo).
  * `kl/ref_army` subió de `0.0000` a `2.3100`. Al aflojarse la penalización KL y gracias al *clipping* inferior propuesto en el diseño, la política militar se desacopló del ancla supervisada, descubriendo patrones ofensivos dinámicos y micro-habilidades que el maestro supervisado desconocía.
* **Entropía de Políticas (`entropy/*`):**
  * `entropy/prod`: Estable entre `0.08` y `0.11` nat (macro determinista pero elástica).
  * `entropy/army`: Comenzó en `0.00` tras el warm-start, subió hasta un pico de `0.99` durante la exploración activa con Discovery Exploiters, y convergió armónicamente a `0.4710`.

### 2.2 Telemetría de Pérdidas y Calidad del Crítico

* **Pérdida de Valor (`loss/value`):** Mínimo de `0.0007`, promedio de `0.0024`.
* **Varianza Explicada (`rollout/explained_var`):**
  * Saltó de `0.0979` en la iteración 1 a **`0.9694`** en la iteración 10, manteniéndose entre `0.9616` y **`0.9776`** durante el resto de la liga. El Crítico predice con precisión milimétrica el retorno esperado de estados asimétricos entre ambos mapas.
* **Pérdida de Política (`loss/pi_prod` y `loss/pi_army`):**
  * Variación simétrica en torno a $[-0.0014, 0.0008]$, sin picos de gradiente tras la normalización GAE y el clipping PPO ($\epsilon = 0.2$).

### 2.3 Evolución del Comportamiento Táctico en el Motor (`behavior/*`)

Las métricas de comportamiento demuestran una sofisticación doctrinal profunda respecto a v4:

| Métrica de Comportamiento | Iteración 1 | Iteración 40 | Iteración 80 | Interpretación Doctrinal |
|---|---|---|---|---|
| **`behavior/army_keep_pct`** | 38.3% | 60.1% | **62.0%** | **Disciplina de Bloque:** El ejército preserva formación y espera masa crítica en lugar de lanzarse en goteo. |
| **`behavior/army_atk_pct`** | 50.2% | 29.8% | **27.7%** | **Fin del Feed Suicida:** Reduce ataques a ciegas un -44.8%, atacando únicamente cuando tiene ventaja táctica. |
| **`behavior/army_allin_pct`** | 1.8% | 3.7% | **5.7%** | **Golpe de Gracia Quirúrgico:** Detecta el colapso del rival y activa el avance total para liquidar la partida rápidamente. |
| **`behavior/harass_atk_pct`** | 53.7% | 15.2% | **12.8%** | **Micro de Hostigamiento Inteligente:** Deja de regalar jeeps/infantería ligera en bases fortificadas. |
| **`behavior/veh_arty_pct`** | 14.3% | 21.2% | **19.5%** | **Doctrina de Asedio:** Emplea artillería de largo alcance sistemáticamente contra torretas y nidos de ametralladoras. |
| **`behavior/veh_2tnk_pct`** | 58.7% | 59.7% | **59.9%** | **Espina Dorsal Blindada:** Producción constante y balanceada de Medium Tanks. |
| **`behavior/air_train_pct`** | 27.7% | 8.1% | **7.0%** | **Equilibrio Aéreo:** Evita sobreinversión en cazas vulnerables y mantiene cobertura de apoyo. |

---

## 3. Torneo Oficial Head-to-Head: v5 vs v4 Campeón (100 Partidas)

En el torneo bilateral de 100 partidas con inversión de slots (`Multi0` vs `Multi1`), pool dual 50/50 y semillas oficiales independientes `50000–50099`:

```
v5 Campeón : 92 victorias  ████████████████████████████████████████████████ (92.0%)
v4 Campeón :  6 victorias  ███ (6.0%)
Truncadas  :  2 partidas   █ (2.0%)
```

### 3.1 Desglose por Mapa

| Mapa de Encuentro | Partidas | Victorias v5 | Victorias v4 | Truncadas | Win Rate v5 | IC 95% (Wilson) |
|---|---|---|---|---|---|---|
| **`singles.oramap`** (32x32 clásico) | 50 | **44** | 4 | 2 | **88.0%** | [76.2%, 94.4%] |
| **`doubles.oramap`** (64x64 macro) | 50 | **48** | 2 | 0 | **96.0%** | [86.5%, 98.9%] |
| **Global Pool Dual** | **100** | **92** | **6** | **2** | **92.0%** | **[85.0%, 95.9%]** |

> [!IMPORTANT]
> **La Superioridad Espacial en `doubles.oramap`:**
> En el mapa extendido (`doubles`), v5 alcanzó un demoledor **96.0% de Win Rate**. La versión v4 sufría de sobreajuste topológico a `singles` al carecer de atención cruzada tokenizada; en cambio, el bloque *Cross-Attention* de 17 tokens de v5 permitió detectar centros de expansión secundarios y asediar las bases remotas de v4 con el escuadrón `MAIN` y artillería antes de que pudiera consolidar refinerías periféricas.

### 3.2 Desglose por Facción y Posición (Spawn Slot)

* **Por Facción de v5:**
  * **Alemania (`germany`):** 27 victorias / 29 partidas (**93.1% Win Rate**, 0 truncadas)
  * **Francia (`france`):** 34 victorias / 37 partidas (**91.9% Win Rate**, 0 truncadas)
  * **Inglaterra (`england`):** 31 victorias / 34 partidas (**91.2% Win Rate**, 2 truncadas)
* **Por Posición de Inicio (Spawn Slot Symmetry):**
  * **Slot `Multi0`:** 46 victorias / 50 partidas (**92.0% Win Rate**)
  * **Slot `Multi1`:** 46 victorias / 50 partidas (**92.0% Win Rate**)
  * *Sesgo posicional medido:* **0.0%** (paridad estadística perfecta).

---

## 4. Benchmark Oficial contra Bots Exploiters en Pool Dual (40 Partidas c/u)

| Oponente Bot | Mapa | Partidas | Victorias | Derrotas | Truncadas | Win Rate | IC 95% (Wilson) | Decisive WR (W/(W+L)) |
|---|---|---|---|---|---|---|---|---|
| **`hard`** | Pool Dual (`singles` + `doubles`) | 40 | **33** | 6 | 1 | **82.5%** | [68.0%, 91.3%] | **84.6%** |
| **`turtle`** | Pool Dual (`singles` + `doubles`) | 40 | **28** | 2 | 10 | **70.0%** | [54.6%, 81.9%] | **93.3%** |
| **`brutal`** | Pool Dual (`singles` + `doubles`) | 40 | **17** | 22 | 1 | **42.5%** | [28.5%, 57.8%] | **43.6%** |
| **`bc_v5` (Anchor)** | Pool Dual (`singles` + `doubles`) | 40 | **40** | 0 | 0 | **100.0%** | [91.2%, 100.0%] | **100.0%** |

### 4.1 Análisis de Comportamiento contra Bots

1. **Dominio vs `hard` (82.5%):** El bot normal de OpenRA no puede sostener intercambios favorables contra la micro-gestión de `SPECOPS` y el uso medido de habilidades tácticas. En `doubles`, v5 expande refinerías más rápido y ahoga la economía del bot.
2. **Quiebre de Defensas vs `turtle` (70.0% WR / 93.3% Decisivo):**
   * En `singles`, el ratio fue del **80.0%** (16W / 1L).
   * En `doubles`, la IA turtle fortifica bases dispersas; la artillería aliada demolió las defensas sin bajas críticas (solo 2 derrotas en 40 partidas). Las 10 partidas truncadas ocurrieron cuando el bot construyó torretas aisladas en bordes extremos del mapa grande, mitigadas progresivamente con `OPENRA_RL_EARLY_WIN=1`.
3. **Resistencia vs `brutal` Rush (42.5%):**
   * El bot `brutal` cuenta con ingresos inflados por trampa (+150% cash) y spam instantáneo de infantería/tanques desde el tick 0.
   * v5 supera holgadamente el objetivo del plan ($\ge 30\%$) obteniendo un sólido **42.5%** mediante el repliegue defensivo temprano (`army_keep`), aguantando el rush hasta nivelar la tecnología.

---

## 5. Evolución Histórica de la Liga de Auto-Juego (Fase 5.3)

La liga ejecutó 80 iteraciones con muestreo heterogéneo:
* **30% Self-Play on-policy** ($\pi_\theta \text{ vs } \pi_\theta$).
* **20% Historical Snapshots (PFSP)** del Hall of Fame.
* **35% Bots Exploiters C#** (`hard`, `brutal`, `turtle`).
* **15% Discovery Exploiters & Anchor** (`bc_v5`).

| Iteración | Elo Liga | Composite Score | vs `bc_v5` WR | vs `hard` WR | vs `brutal` WR | vs `turtle` WR | Varianza Explicada ($R^2$) |
|---|---|---|---|---|---|---|---|
| **Iter 10** | 1216.1 | 0.177 | 15.0% | 17.5% | 17.5% | 22.5% | 0.9694 |
| **Iter 20** | 1278.5 | 0.317 | 50.0% | 30.0% | 35.0% | 22.5% | 0.9670 |
| **Iter 30** | 1342.0 | 0.552 | 87.5% | 50.0% | 62.5% | 40.0% | 0.9722 |
| **Iter 40** | 1528.9 | 0.542 | 95.0% | 40.0% | 75.0% | 40.0% | 0.9683 |
| **Iter 50** | 1576.9 | 0.551 | 97.5% | 52.5% | 80.0% | 32.5% | 0.9708 |
| **Iter 60** | **1657.6** | 0.658 | 95.0% | 70.0% | 72.5% | 45.0% | 0.9642 |
| **Iter 70** | 1589.3 | **0.773** | 92.5% | 75.0% | 75.0% | **70.0%** | 0.9616 |
| **Iter 80** | **1612.4** | **0.729** | **100.0%** | **80.0%** | **75.0%** | 52.5% | **0.9701** |

---

## 6. Auditoría Técnica de Componentes y Entrega

| Componente | Archivos Clave | Estado |
|---|---|:---:|
| **Contrato gRPC v5 & Protobuf** | [`proto/rl_bridge.proto`](file:///c:/Users/lordc/Desktop/OpenRA-RL/proto/rl_bridge.proto) (`DECIDE_ABILITY`, `DECIDE_MACRO_OPS`) | ✅ Verificado |
| **Destacamento `SPECOPS` en Motor C#** | [`RLCommander.cs`](file:///c:/Users/lordc/Desktop/OpenRA-RL/OpenRA/OpenRA.Mods.Common/Traits/Player/RLCommander.cs), [`SquadManagerBotModule.cs`](file:///c:/Users/lordc/Desktop/OpenRA-RL/OpenRA/OpenRA.Mods.Common/Traits/BotModules/SquadManagerBotModule.cs) | ✅ Verificado |
| **Cross-Attention Espacial (17 Tokens)** | [`ra_rl/features.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/ra_rl/features.py) (`FEATURE_VERSION = 4`), [`ra_rl/model.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/ra_rl/model.py) | ✅ Verificado |
| **Clipped Reverse KL & PPO v5** | [`ra_rl/ppo.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/ra_rl/ppo.py), [`ra_rl/selfplay.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/ra_rl/selfplay.py) | ✅ Verificado |
| **Evaluador Multi-Mapa & Aliasing de Peers** | [`ra_rl/eval.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/ra_rl/eval.py) (`--peer-policy`, multi-map cycling) | ✅ Verificado |
| **Pipeline Resumible & Stop/Resume Seguro** | [`scripts/run_v5_pipeline.py`](file:///c:/Users/lordc/Desktop/OpenRA-RL/scripts/run_v5_pipeline.py) | ✅ Verificado |
| **Mitigación de Incompletas en Contenedores** | [`docker-compose.yaml`](file:///c:/Users/lordc/Desktop/OpenRA-RL/docker-compose.yaml) (`OPENRA_RL_EARLY_WIN=1`) | ✅ Verificado |
| **Guía de Contingencia y Diagnóstico** | [`docs/v5-contingencia-y-diagnostico.md`](file:///c:/Users/lordc/Desktop/OpenRA-RL/docs/v5-contingencia-y-diagnostico.md) | ✅ Verificado |

---

## 7. Conclusión y Veredicto

La iteración **v5 (Autonomous General)** ha establecido un nuevo techo de rendimiento en OpenRA-RL. Al alcanzar **1612.4 Elo**, **92.0% de Win Rate contra el Campeón v4** y una tasa de victoria del **100% contra el modelo base supervisado**, se demuestra empíricamente la efectividad de:
1. Reemplazar listas negras estáticas por atención cruzada espacial y guardias dinámicas.
2. Incorporar el destacamento `SPECOPS` para preservar y accionar unidades de alto impacto tecnológico sin perderlas en el combate masivo.
3. El uso de *Clipped Reverse KL* para guiar el despegue inicial sin asfixiar la innovación táctica del agente en etapas maduras del auto-juego.
