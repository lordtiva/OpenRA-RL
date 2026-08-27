# Filosofía de traducción bot → RL: 4 pilares

> **Origen:** análisis del motor `OpenRA/mods/ra/rules/ai.yaml` (verificado contra fuente) + crítica teórica de RL sobre sobre-ingeniería de reward. Consolida el hilo del 2026-08-27 donde se cruzó la ingeniería del bot con la teoría de PPO.

---

## 1. Qué tiene el bot de máxima dificultad que no tiene tu `beginner`

Tu rival actual (`--bot-type beginner`) no es "fácil", es **deliberadamente capado**. El mapeo que usamos es `hard → normal` y `brutal → rush` (`openra_env/server/openra_process.py:BOT_TYPE_MAP`). El `beginner` tiene delays y límites artificiales que no existen en `normal`/`rush`.

| Sistema | `beginner` (tu sparring actual) | `hard=normal` / `brutal=rush` | Consecuencia para RL |
|---|---|---|---|
| **Economía** | `HarvesterBotModule InitialHarvesters 1`, `AdditionalRefinery 0`, `ProductionMinCash 1000`, `NewProductionCashThreshold 15000` | `normal: Harvesters 4`, `rush: 2`, `AdditionalRefinery 2`, `ProductionMinCash 500`, `Threshold 7000-8000` | El hard abre 2ª refinería sin esperar $15k y sostiene 4 cosechadoras. Tu agente nunca vio presión de escalar porque el rival tampoco escala. |
| **Tempo de obra** | `StructureProductionActiveDelay 200 + Inactive 500 + Random 100`, `pbox 5000`, `gun 8000`, `dome 20000` | `normal: pbox 1500, gun 2000, ftur 1500, dome 6000`; `rush: pbox 2000` | El hard construye ~3× más rápido (APM efectivo). Con 1 decisión/80 ticks, tu agente se ahoga si no aprende el loop proc→harv→cashflow. |
| **Tech** | Solo `dome` | `dome, atek, stek, fix, afld, hpad, mslo` | El `beginner` jamás saca `atek/stek` → nunca ves `3tnk/4tnk/arty/v2rl/mig`. El hard sí te outtechea. |
| **Composición** | `UnitsToBuild e1:70 e3:20 dog:5 harv:5` límites `e1 4, e3 2, harv 2` — solo infantería | `normal: e1 65, 1tnk 40, 2tnk 50, 3tnk 50, 4tnk 25 + arty/v2rl/heli/mig/yak/ss/dd/ca`, `harv 8` sin límite práctico en tanques | Sparring débil enseña que 8 rifles en `attack_move` alcanzan. Contra masa `normal` eso es aniquilado por **Lanchester N²**. |
| **Defensa** | `pbox 2, gun 1` | `normal: pbox 9, gun 9, ftur 10, tsla 5, agun 5, sam 5 + gap` | El `beginner` no castiga dejar la base vacía. El hard sí. |
| **Ataque** | `SquadManager SquadSize 3+2, RushInterval 5000, MinimumAttackForceDelay 150000` (~100 min) | `normal: SquadSize 40`, `rush: 20`, `medium: 20+15, Delay 5000` | El `beginner` gotea de a 3. El hard junta 40 y cae en oleadas. Mandar 3 vs base con defensas → `w_kills - w_deaths` negativo natural. |
| **Reparación / Energía** | **Sin** `BuildingRepairBotModule` ni `PowerDownBotModule` | `BuildingRepairBotModule` + `PowerDownBotModule` (apaga `dome/tsla/mslo` sin luz) | El `beginner` deja morir edificios al 10% HP sin reparar. El hard no — tu `defense_first -3.0` jamás dispara vs `beginner`. |
| **Expansión** | Sin `McvExpansionManager` | `MinimumConstructionYardCount 2`, `ExpansionTolerate 1,2` | El hard abre 2ª base cuando se vacía la mina. El `beginner` nunca. |

**Conclusión de ingeniería:** el `beginner` es un gran sparring para aprender el loop *refinería temprana → cashflow* sin morir a los 3 minutos, pero **enseña malos hábitos** si te quedás ahí (goteo, no escalar, no defender base).

---

## 2. Crítica RL: dónde la ingeniería del bot no se traduce directo a reward

> Esta sección corrige las 3 propuestas de Fase 3 que el análisis de motor sugería, desde PPO puro.

### 2.1 Dónde el análisis del motor tiene 100% de razón

1. **Vicio del sparring débil** — confirmado por Lanchester. Entrenar solo vs `beginner` fija el óptimo local en "8 rifles + attack_move".
2. **Cuello de APM** — el bot resuelve en paralelo cada tick (cola inf, cola edificios, harvesters, guardias, micro). Tu agente: **1 decisión / 80 ticks** en espacio unificado. Gastar una decisión en `repair` de un silo es desperdicio de ancho de banda.

### 2.2 Riesgos de sobre-ingeniería de reward (no hacer)

#### ⚠️ `w_mass` — penalizar `attack_move` con <10 unidades

*Peligro:* mata scouting y harass temprano (1 jeep a explorar, 2 perros a matar harvester).
*Alternativa RL:* no prohibir con `if`. La asimetría `w_kills 0.15 / w_deaths 0.05` + un rival `medium/hard` ya lo castiga natural: 3 vs base con 4 defensas = `-$300` y `+0` → retorno negativo. Dejar que el gradiente lo descubra.

#### ⚠️ `w_tech` plano por construir `atek`/`stek`

*Peligro:* hito estático `+0.8` por edificio → la red lo construye aunque la estén demoliendo, farmeando reward antes de morir (visto en `w_building` vs muros).
*Alternativa RL:* **PBRS por Tier** — potencial `Φ(s)` por *acceso* a unidades avanzadas, no por evento de poner el edificio. Si el edificio cae, el potencial cae y el reward se anula (no farmeable). Diferir hasta que `sonda-horizonte` mida tech.

#### ⚠️ Micro-rewards densos mal calibrados

Cualquier `w_*` <0.01/bloque puede ser ahogado por ruido de `combat_scale`. Calibrar contra `combat 0.15/1000` y `mining_rate 0.04/1000`, no en vacío.

---

## 3. Traducción elegante bot → RL (4 pilares)

```
                  ┌──────────────────────────────────────────────┐
                  │        FILOSOFÍA DE TRADUCCIÓN A RL          │
                  └──────────────────────┬───────────────────────┘
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
1. CURRÍCULUM       2. AUTONOMÍA DE                 3. CONCIENCIA       4. PRESIÓN DE
DE RIVALES             SOPORTE                      LANCHESTER          EXPANSIÓN
(Beginner→Hard)     (Capa de confort en C#)         (Escalares de masa) (Agotamiento mineral)
```

### Pilar A — Currículum de sparring (la verdadera solución)

PPO solo optimiza para lo necesario. No fuerces juego pro vs `beginner`.

| Fase | `--bot-type` | Objetivo norte | Señal que desbloquea |
|------|--------------|----------------|----------------------|
| **2 (actual)** | `beginner` | Dominar loop macro: `proc` temprana → cashflow → `winrate >60%` en Escenario A | `has_refinery`, `can_afford_proc`, `early_refinery` + `first_ore` |
| 3A | `easy` | Medir vs rival con vehículos y grupos 10-15 | Composición mixta emerge sin `w_mass` |
| 3B | `medium` / `hard` (=`normal`) | Supervivencia vs defensas serias + `atek/stek` | Presión ambiental fuerza `3tnk/4tnk/arty` y `garrison` |

Regla: cambiar de rival solo por `winrate`, no por reward.

### Pilar B — Autonomía de soporte (cerrar brecha de APM)

AlphaStar / OpenAI Five no gastan decisión de alto nivel en mantenimiento trivial.

* **Reparación y energía:** en lugar de obligar a la red a gastar 1/80 ticks en `repair`, delegar a capa de entorno/C# si `cash>500` y `HP<35%` → `BuildingRepairBotModule` del hard hecho por el adapter. Reserva 100% de la red para *dónde mover el ejército y qué tech abrir*.
* **Estado actual:** `SCALAR_DIM 19` ya expone `garrison_ratio` y `has_refinery` para que la cabeza de acción sepa si puede vaciar la base. La reparación automática es el siguiente paso si `defense_loss` sigue sin disparar.

### Pilar C — Conciencia de Lanchester (en lugar de `w_mass`)

En vez de penalizar ataques chicos, **mostrar** la ventaja:

```python
# obs_encoding.py — escalar propuesto (no implementado aún, a evaluar en Fase 3)
military_ratio = own_army_value / max(visible_enemy_army_value, 300.0)
# <1.0 → V(s) aprende que avanzar sangra; >1.5 → política dispara army_attack_move
```

El crítico aprende el gradiente solo; la política mantiene capacidad de harass/scout.

### Pilar D — Presión de expansión por agotamiento

El `hard` expande (`McvExpansionManager`) porque la mina inicial se vacía en `>10000` ticks.

* Tu reward `w_mining_rate * Δearned/1000` ya crea esa presión: cuando la mina local cae a 0, el reward económico se apaga.
* Con el encoder **U-Net + CoordConv** (RF ~40 celdas, `network.py`) el canal `resource density` (Ch2) ya ve campos lejanos. La red aprende a mover harvs o a plantar `proc` avanzado para re-encender el cashflow — sin reward extra.

---

## 4. Recomendación operativa (2026-08-27)

1. **No toques el run en curso** — dejar 100-150 iters con `eradicate_v3` (19 escalares + `w_refinery_early/w_first_ore/w_garrison`) para verificar primeras victorias vs `beginner`.
2. **Criterio de promoción:** `winrate >60-70%` sostenido en Escenario A → `--bot-type easy`, no sobre-optimizar `beginner`. Verás composición pesada emerger por presión, no por reward.
3. **Diferir `w_mass`/`w_tech`:** evaluar al ver meseta, y preferir escalares/PBRS antes que hitos planos.

*Fuente de verdad de números del bot:* `OpenRA/mods/ra/rules/ai.yaml` (HarvesterBotModule, BaseBuilderBotModule, SquadManagerBotModule, UnitBuilderBotModule). Verificado 2026-08-27.*
