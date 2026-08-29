# Capa 0 — estado empírico post Run 8 / Run 9

> **Fecha:** 2026-08-29 · **No reemplaza** `12-plan-4-capas-siguiente-nivel.md`. Ese plan se deja tal cual: es realista para 2070+5600X+32GB y el orden (entorno → BC/SIL → red → self-play) sigue siendo el correcto.
> **Qué es esto:** desglose de qué de la Capa 0 ya está, qué no, y qué hacer si el run actual (resume iter 219, `eradicate_v4`, a_short vs beginner) sigue plano a iters 400–450.
>
> **Corte 442 (hecho):** meseta confirmada, incomplete 24%→60%, `army_attack_move` → 0%. Asalto sostenido implementado en `auto_support.py` (proc+harv + ≥4 combate → `army_attack_move` cada bloque). Resume `latest.pt` (la política que ya construye); no volver a 229.
>
> **Corte 560:** el asalto **sí movió el wr** (6% → ~18% en ventana 444–560; wr20 0.06 → 0.15–0.23, pico 0.50 en 466). Wins más rápidos (36k → 24k). Incomplete sigue ~52–61% (enB ~9–15: no es el perro en niebla). H sana. **Seguir**; no `win_early` todavía. `best.pt` sigue 229 (iwr 1.0 congela el puntero).
>
> **Corte 600:** el 0.20 wr20 fue un 3/4 (iter 600) y en 601 ya volvió a 0.15. Ventana 561–601 **empeoró** vs 501–560 (win 17%→10%, wr20 0.17→0.09). H 0.9 son tandas 4/4 lose con ownB=0, al iter siguiente H≈2: no es el colapso Run8. El asalto ya cobró; más PPO solo no está subiendo el piso. Siguiente: Capa 1 (BC/SIL), no otro corte a 650.
>
> **Capa 1 (arranca ~iter 604):** `L = L_PPO + λ_bc L_BC + λ_sil L_SIL`. Maestro `ScriptedTeacher` (proc antes de barracks, `army_attack_move`). 1 episodio teacher/iter. SIL buffer 4k steps de win/raze. λ_bc 1→0 en 80 iters. Resume `latest.pt` (iter 603). Run 9 archivado en `rl/ckpts/Run 9 (a_short assault 220-603)/`.

---

## Veredicto sobre el Documento 12

El plan es alcanzable **como roadmap de 6–12 meses**, no como “la semana que viene wr 70%”.

| Capa | ¿Cabe en el hardware? | ¿El orden es el correcto? |
|---|---|---|
| 0 entorno (win posible + asalto sostenido) | sí, software | sí: primero. Sin `win` estable, BC y transformer sobreajustan al incomplete |
| 1 BC + SIL | sí, 8GB | sí: tabula rasa a 10² partidas/h es masoquismo |
| 2 transformer entidades ~+3–4M | sí, 2070 | sí, **después** de que un push se sostenga |
| 3 easy/hard + 3 ckpts PFSP | sí, 2 daemons | sí, **cuando wr>30% vs beginner** |

Lo que **no** hay que “corregir” del 12: no Dreamer, no red 50M, no liga de 12, no apagar shaping ahora, no tirar PPO.

La única calibración (no cambio de plan): el “0% → 60–70% wr sin tocar la red” de `09-fullstack-run3.md` §7 asumía el coloso Run3 (`$80k vs $7k`, wr=0 por un perro en niebla). **Este run no es ese agente.** Capa 0 acá es convertir un empate largo en un cierre, no cobrar un paliza ya hecha.

---

## Cómo se ve este run (no es colapso)

Resume desde `best.pt` iter 219 tras archivar Run 8 (`no_op` 99%, H≈0.002). Mitigaciones activas: mask combate hasta proc, auto-deploy MCV, `w_no_econ_lose=4`, skip PPO si batch >80% `no_op`, watchdog de política muerta + restore `best.pt` con Adam fresco.

Muestra (iters 220–347, ~512 partidas):

| Señal | Valor | Lectura |
|---|---|---|
| wr era | ~0.11 (54/492 a iter 342) | meseta, no deriva |
| wr20 | oscila 0.00–0.35, típico ~0.10 | ruido de 4 eps/iter |
| H | ~1.5–1.72 | sana (Run 8 moría a 0.002) |
| `no_op` | ~5% | no es el modo deploy-sit |
| `first_ore` | siempre 1.5 | bootstrap económico funciona |
| `no_econ_lose` | 0 | el parche no está disparando (bien) |
| `army_attack_move` | ~0.1% del hist | el remate de ejército no se usa |
| `best.pt` | iter 229, 4/4, +17 rew | spike, no el nivel típico |

Outcomes (misma ventana): **11% win, 60% lose, 29% incomplete**. En el último tercio el incomplete sube a **36%** y los wins se alargan (25k → 37k ticks). El agente aprende a **durar**, no a rematar.

---

## Incomplete ≠ “casi gana”

Todos los incomplete pegan el mismo techo: **51792 ticks** (624 decisiones × 80 + setup). No es “faltan 2 minutos”.

| Resultado | n | ticks (mediana) | Qué es |
|---|---|---|---|
| win | ~11% | 33k (p90 46k) | si puede cerrar, cierra **antes** del cap |
| lose | ~60% | 24k (97/309 <12k) | wipe temprano; el beginner ya aniquila |
| incomplete | ~29% | **siempre 51792** | empate |

Tandas 4/4 incomplete (únicas donde ownB/enB no están mezclados): ownB **5.2**, enB **13.9**, raze 2.5, mining flojo, reward **+2.4**. El bot se expandió. Comparar: lose puro ≈ −5, win puro ≈ +17. Quedarse vivo hasta timeout es un óptimo local cómodo (`w_timeout=−1` vs `w_lose=−2.5`).

La sonda (`07-sonda-horizonte.md`) ya midió: en este escenario el beginner **no ataca** en 51k ticks de pasividad. Alargar `max-steps` **no** hace que el rival te remate; alarga el SimCity. Con γ=0.995, un win al paso 624 vale ~4% en t0 (`0.995^624≈0.044`); más horizonte vuelve el `+8` invisible.

**No alargar partidas.** El cuello no es reloj.

---

## Las 3 piezas de Capa 0, una por una

Criterio de promoción del 12: `raze>0` y `winrate>0` vs beginner. **Ya se cumple** (raze ~1.5, wr ~11%). El trabajo que queda es el espíritu de la capa (que un push se sostenga y que el MDP declare cuando de verdad ganaste), no la métrica mínima.

| # | Qué pide el 12 | Estado 2026-08-29 | ¿Desbloquea *este* incomplete? |
|---|---|---|---|
| 1 | Asalto sostenido: `army_attack_move` como `auto_support` (la red elige *cuándo* hay ejército; el entorno re-emite cada bloque) | **A medias.** El tipo existe y salta harvesters. Keep-alive en `auto_support` **solo sigue** un push si la política ya eligió `attack_move`/`army_attack_move`. Hist ~0.1% → casi nunca arranca. | **Sí.** Es el hueco de este plateau. |
| 2 | Declaración: `win` si no queda producción enemiga N ticks | **Hecho y castrado.** `win_early` en `ExternalBotBridge` (500 ticks). La condición `eneProd==0` está **comentada** (falso positivo: `ProductionQueue` daba 0 con ConYard vivo). Solo queda patrimonio enemigo &lt;10% del propio. | **No ahora.** Con enB=14 el 10% no dispara. El engine hace bien en no declarar. Reabrir cuando el asalto deje enB≈0. |
| 3 | Horizonte vs γ: 624×80, γ 0.995–0.997, cortar cuando el engine declare | **Hecho** (γ=0.995). | Alargar sería contraproducente. |

`attack_move` normal = **un** `actor_id`. Si la red elige una recolectora, esa recolectora marcha. El grupo es `army_attack_move` (sin actor; el C# filtra `Harvester`).

Comandos que el engine ya tiene y la política v0.1 no muestrea (`ENABLED_TYPES`): `sell`, `repair` (sí auto_support), `set_rally_point`, `guard`, `enter_transport`/`unload`, `power_down` (sí auto_support), `set_primary`, `surrender`. `patrol` está en el proto y **no** en `ActionHandler`. Para cerrar partidas no falta un comando nuevo: falta que el push de ejército se sostenga.

---

## Qué hacer si a ~400–450 sigue plano

Un cambio de régimen por vez. **No tocar la red.**

**Corte ejecutado a iter 442.** Datos 401–443: win 8%, lose 32%, **incomplete 60%**, ownB 4.2, enB 8.4, army hist **0.01%**, H 1.71 (sana), wr era 0.089 (80/896). Reward **subió** porque timeout paga ~+3 vs lose −5: está farmeando el empate, no colapsando.

1. **Asalto sostenido de verdad (capa 0.1 — primero). HECHO.**  
   `auto_support`: si hay proc+harv **y** ≥4 unidades de combate, emite `army_attack_move` (el C# salta harvesters) al enemigo visible, si no al beacon, **aunque** la decisión de esa tanda sea `train`/`build`. Keep-alive de `last_push` se mantiene para ejércitos chicos.  
   Relanzar `auto_train` sobre **`latest.pt`** (la política que ya sobrevive y construye), no sobre el spike 229.  
   Expectativa honesta: 11% → **20–30%**, no 60–70%. Medir 50–80 iters: ¿baja el incomplete? ¿sube wr20? Si el asalto convierte incomplete en lose (empuja y se muere) → parar el push automático, no sumar `w_timeout` en el mismo corte.

2. **Solo si el asalto sube raze y el incomplete queda con enB≈0.**  
   Ahí sí es el bug de MDP del 12 / Run3. Reabrir `win_early` contando **tipos** (`fact`/`weap`/`barr`/`proc`/`tent`), no el trait `ProductionQueue`. 500 ticks a 0 producción enemiga → `win_early`.

3. **Si el asalto convierte incomplete en lose** (empuja y se muere): parar el push automático. Recién entonces incomplete más caro (`w_timeout` cerca de 2.5) o SIL de las partidas que sí ganan (Capa 1). No las dos cosas juntas.

### Qué no hacer en ese corte

- Más `max-steps` / más ticks.
- Subir `w_raze` a 5.
- Bajar garrison/mining (el 12 lo pide **si raze=0**; no es el caso).
- Capa 2 (transformer) ni Capa 3 (self-play pide wr>30%).
- BC del `scripted_bot` todavía: es Capa 1, y tiene sentido *después* de que un push se sostenga.

---

## Relación con Run 7 / Run 8

| Run | Síntoma | Lección |
|---|---|---|
| 7 (201–310) | 100% `attack_move` sin base | combate legal antes de proc = reward-hack. Mask hasta proc. |
| 8 (220–408) | 99% `no_op`+deploy, H≈0, −2.57 &gt; pelear (−6) | el mask sin watchdog de *este* modo convierte spam-attack en spam-noop. Morir desnudo era más barato que jugar. |
| 9 (220–…, este) | H sana, eco viva, wr~11% plano, incomplete 36% | mitigaciones de colapso OK. El techo ahora es APM de asalto, no PPO ni horizonte. |

Ckpts del Run 8: `rl/ckpts/Run 8 (a_short collapse no_op 220-408)/`. Resume vivo: `best.pt` / `latest.pt` = iter 219 (luego 229 si el 4/4 sigue siendo best).

---

*Guardado: 2026-08-29 — rama `exp/rl-2026-08-28-grok`. Companion de `12-plan-4-capas-siguiente-nivel.md`; no modifica el plan.*
