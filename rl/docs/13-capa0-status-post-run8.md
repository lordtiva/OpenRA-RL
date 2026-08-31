# Capa 0 — estado empírico post Run 8 / Run 9

> **Fecha:** 2026-08-30 (visor + parche de destino) · **No reemplaza** `12-plan-4-capas-siguiente-nivel.md`. Ese plan se deja tal cual: es realista para 2070+5600X+32GB y el orden (entorno → BC/SIL → red → self-play) sigue siendo el correcto.
> **Qué es esto:** desglose de qué de la Capa 0 ya está, qué no, y qué hacer si el run actual (resume iter 219, `eradicate_v4`, a_short vs beginner) sigue plano a iters 400–450.
>
> **Corte 442 (hecho):** meseta confirmada, incomplete 24%→60%, `army_attack_move` → 0%. Asalto sostenido implementado en `auto_support.py` (proc+harv + ≥4 combate → `army_attack_move` cada bloque). Resume `latest.pt` (la política que ya construye); no volver a 229.
>
> **Corte 560:** el asalto **sí movió el wr** (6% → ~18% en ventana 444–560; wr20 0.06 → 0.15–0.23, pico 0.50 en 466). Wins más rápidos (36k → 24k). Incomplete sigue ~52–61% (enB ~9–15: no es el perro en niebla). H sana. **Seguir**; no `win_early` todavía. `best.pt` sigue 229 (iwr 1.0 congela el puntero).
>
> **Corte 600:** el 0.20 wr20 fue un 3/4 (iter 600) y en 601 ya volvió a 0.15. Ventana 561–601 **empeoró** vs 501–560 (win 17%→10%, wr20 0.17→0.09). H 0.9 son tandas 4/4 lose con ownB=0, al iter siguiente H≈2: no es el colapso Run8. El asalto ya cobró; más PPO solo no está subiendo el piso. Siguiente: Capa 1 (BC/SIL), no otro corte a 650.
>
> **Capa 1 (arranca ~iter 604):** `L = L_PPO + λ_bc L_BC + λ_sil L_SIL`. Maestro `ScriptedTeacher` (proc antes de barracks, `army_attack_move`). 1 episodio teacher/iter. SIL buffer 4k steps de win/raze. λ_bc 1→0 en 80 iters. Resume `latest.pt` (iter 603). Run 9 archivado en `rl/ckpts/Run 9 (a_short assault 220-603)/`.
>
> **Corte visor 817 (2026-08-30):** `best.pt` 780 era 4/4 vs beginner y wr20 0.60, pero el live muestra **SimCity + hormiguero en casa**, no un push. Caps en `rl/ckpts/Screenshot 2026-08-30 091502.png` y `091710.png`. El asalto de `auto_support` re-emitía `army_attack_move` cada bloque a `last_push` de la red (celda en el blob propio). Parche: destino = enemigo visible / beacon; no re-ordenar unidades que ya caminan. Relanzar desde `latest` ~817.
>
> **Visor 835–836 (4/4 win):** con Title `Singles` resuelto, `support_dests[0]=[95,11]` y el blob llega (`dist_beacon` 10–12). El 4/4 lo carga **auto_support**, no la cabeza de celda (`last_push` sigue en casa). Falencias de la red → sección *Deuda de la cabeza de celda* (Capa 2, no ahora).
>
> **Hunt ~854:** wr era ~30% al alza, wr20 ~0.40. El ejército **llega** al beacon y se queda idle; incomplete con enB 5–10 (edificios resagados en niebla) → timeout 51k. Parche: si ≥4 combate ya están en el beacon y no hay objetivo visible, dest rota por waypoints al sur/oeste del NE. El acercamiento sigue siendo beacon. Relanzar `auto_train` + visor para cargarlo.
>
> **Corte ~900 → Capa 3 easy:** wr20 ≥0.50 **43 iters** (856–898), lose 4.7%, era 36.1%, `best.pt` 898 4/4 wr20 0.70. Incomplete vs beginner ~25%. Cumple el 12 (`wr>30%`). **Avance = `--bot-type easy`**, misma red / asalto / SIL. **No Capa 2.** Antes de easy: dest defiende si hay enemigo a ≤18 de un edificio propio (el stray-ignore del beginner ignoraba raids en casa). `resolve_beacon` también en `_batch_of`. Restore: `best.pt` 898. Archivo: `rl/ckpts/Run 10 (a_short capa1 beginner 604-900)/`. Expectativa: wr20 se cae (easy pega); si lose>25% por 15 iters → restore 898.
>
> **Corte 935 (Run 11 paliza easy):** 35 iters, **0/140** (lose 95%, incomplete 5%, 0 wins). Lose media 14k ticks, `defense_loss`≈−4 plano, cosecha easy ~3×, H sana (~1.2: no es colapso Run 8). El dest “defend” existía pero **no cancelaba el path**: el blob caminaba al beacon y el easy razeaba la casa. El crédito de dest tampoco: PPO/SIL veían click en casa. **No Capa 2.** Archivo: `rl/ckpts/Run 11 (a_short capa1 easy 901-935)/`. Resume `best.pt` **899** vs **beginner**. Parches de este corte: (1) `apply_dest_credit` — `cell_flat` de `army_attack_move`/`attack_move` = dest de soporte; (2) defend recall — raid en casa re-emite `army_attack_move` aunque el blob ya camine, salvo que ya estén peleando ahí. Bar: wr20 vs beginner ≥0.40 en ~20+ iters y visor `last_push` ≈ `support_dests`. Capa 2 (transformer) **después** de eso.
>
> **Corte 977 (Run 12 dest-credit):** 78 iters, wr 0.19, **incomplete ~70%** plano, wr20 0.20, 0 tandas 4/4, `best.pt` sigue 899. El visor 915–921: `last_push=[95,11]` (crédito OK) pero el blob oscila x=40–50 hasta timeout. Causa: after recall el dest vuelve a beacon y **no se re-emite** `army_attack_move` (unidades caminando a casa). Archivo: `rl/ckpts/Run 12 (a_short dest-credit beginner 900-977)/`. Resume **899**. Parche: re-asalto si dest es beacon/hunt, ≥4 combate en casa y <4 en dest. No Capa 2. Bar: incomplete <40% y wr20 ≥0.40.
>
> **Corte 923 (Run 13 SIL bomba):** re-asalto no se pudo juzgar. `sil_nll` 2–4 → **7.8e6** (`1e9/128` = un dest en celda con logit −1e9). Hunt `y=36` pisa agua. `pi_loss` 1e24–1e28, `grad_norm` inf, incomplete 75%. Archivo: `rl/ckpts/Run 13 (a_short reassault sil-bomb 900-923)/`. Resume **899**. Este corte: (1) dest/hunt → `remap_move_cell` + skip si sigue tapada; SIL `lp.clamp(-20)`; `HUNT_Y_MAX` 36→32. (2) **Capa 0 de la tabla órdenes:** rally al dest en tent/weap + stance AttackAnything al nacer + sell hp&lt;12% (no fact/proc). No Capa 2.
>
> **Corte 939 (Run 14 harv al dest):** dest pasable + rally/stance. SIL sano (`sil_nll` ~4–11, no 7e6). wr ~0.10, wr20 0, incomplete alto, `attack_move` 90–150/iter. Visor: la recolectora camina al NE. Causa: (1) dest credit reescribe `attack_move` per-unit al dest y el slot puede ser harv (C# `army_attack_move` salta `Harvester`, el per-unit no); (2) rally de `weap` al dest — HARV sale de Vehicle queue. Live #3: weap=1, **11 harvs**, dest `[95,11]`. Archivo: `rl/ckpts/Run 14 (a_short dest-passable harv-leak 900-939)/`. Resume **899**. Parche: rally combate solo `tent`/`barr`/`kenn`; credit no toca `attack_move` de harv/mcv; adapter `move`/`attack_move`/`attack` sobre harv → `harvest`. No Capa 2.
>
> **Corte 987 (Run 15 train-sbag):** harv OK. Latest 985 vs best 899: 3/4 lose desnudo vs 1/4. Cinta: `train:sbag`/`train:brik` (muros no estaban en `BUILDING_ITEM_TYPES` → cola de edificios, 0 rifles, beginner a tick 7k). Incomplete de ambos = mismo mill (dest beacon, `nd≥4`≈0, centroide x≈45). Archivo: `rl/ckpts/Run 15 (a_short train-sbag 900-987)/`. Resume **899**. Parche: `sbag`/`brik`/`fenc` son BUILD; auto-tent después de proc (como auto-proc). No Capa 2.
>
> **Skirmish GUI (humano vs PPO):** el bot `rl-agent` ya estaba en `rl-bot.yaml`. Ahora el lobby lo muestra como **PPO Agent**. GUI: `FastAdvance(ticks=0)` inyecta órdenes sin acelerar; no `Game.Exit` ni `win_early` contra humano. Sidecar `rl/play_skirmish.py` (autostart desde `Activate`, o `.\play_skirmish.ps1`). Compilar: `OpenRA\make.cmd all` — **no** es el Docker de train. Mapa: Singles / a-short. El 899 todavía mill-ea vs beginner; contra un humano hoy gana el humano. No Capa 2.
>
> **Corte 947 (Run 16 drip-4 + NaN):** wr 0.16, incomplete **81%**, visor 6am: `MIN_ARMY=4` + rally al beacon = oleadas de 4 que mueren en x≈45 (`nd≥4`≈0). PPO crash `Categorical logits nan` (shape 1×22) al update; latest.pt **sin** nan en pesos. Archivo: `rl/ckpts/Run 16 (a_short drip-4 nan 900-947)/`. Resume **899**. Parche: pack **12 en casa** antes de `army_attack_move`; rally **staging** (~10 celdas del fact) hasta el pack; no `attack_move` de 1–3 ociosos; `_categorical` nan→no_op. No Capa 2. Docker `markers=1` / gRPC fail = daemon; recrear compose si vuelve.
>
> **Corte 977 (Run 17 pack12 + bomba SIL/PPO):** pack 12 **sí** (visor 922: 4/4 win, `nh` 11–14 antes del push, `n_support_am=0`, harv en casa, `policy` incluye `[95,11]`). `best.pt` **922** wr20 0.65 iwr 1.0. No Capa 2: wr20≥0.40 racha máx **7** iters, incomplete ~36% en el pico. Iter 923: `sil_nll` 19.6, `pi_loss=inf` (ratio PPO `exp(lp_new-lp_old)` con dest-credit y adv<0; el skip de logits nan no cubría inf). Latest 977 wr20 0, tandas `llll` ownB=0. Archivo: `rl/ckpts/Run 17 (a_short pack12 sil-bomb 900-977)/`. Resume **922**. Parche: clamp log-ratio ±8, skip loss/grad no finito, SIL skip nll≥18. No Capa 2.
>
> **Corte 1010 (Run 18 plateau pack12 → Capa 2):** 88 iters post-922, win **35%**, lose 3%, incomplete **62%** plano, wr20 oscila 0.20–0.55 (last 0.45). SIL/H sanos, 0 bombas. `best.pt` sigue **922** (ningún 4/4 nuevo). El pack no baja el mill: la cabeza de celda sigue ciega al sujeto (Ch6). Archivo: `rl/ckpts/Run 18 (a_short pack12 plateau 923-1010)/`. Resume **922** + **Capa 2** (un régimen): transformer 2×4h d=64 residual-gate 0, scatter 8 ch zero-init, `dist_cell|unidad`, Net2Net `cell_head` 224→296. Adam fresco. Criterio 20 iters: H no colapsa, `last_push` sigue al sujeto. Indexer QSA / GDN / easy = **no** este corte.
>
> **Corte 950 (Run 19 Capa 2 + script de asalto → rueditas off):** 923–950, incomplete ~63%, mismo mill. Visor 922: con soporte patrón `(95,11)` / pack 12; sin soporte 2/4 win y clicks distintos, 300 unidades que la red no elige (MAX_UNITS=48). El dest hardcoded es spawn-asimétrico (GUI al revés = tropas a la derecha). Archivo: `rl/ckpts/Run 19 (a_short capa2 assault-script 923-950)/`. Resume **922**. `SUPPORT_ASSAULT=False`: no pack/hunt/recall/rally-guerra/crédito de dest. Eco on (deploy/proc/tent/harvest/repair/sell/power/stance). Sin parche Ch7/Ch8 en (95,11). `army_attack_move` sigue legal en la política. No easy.
>
> **Corte 952 (Run 20 Capa 2 cerrada vs beginner → Capa 3 easy):** 30 iters, **102/116 win (87.9%)**, incomplete 9.5%, wr20 **1.0**, 17 tandas 4/4, `best.pt` **951**. Capa 2 del 12 (transformer + scatter + `celda|unidad`) **hecha**; 2b (QSA/GDN/GRU 512) no: Ch6 ya no es el techo vs beginner. Archivo: `rl/ckpts/Run 20 (a_short capa2 beginner 923-952)/`. Resume **951**, `--bot-type easy`, asalto support sigue off. Expectativa: wr20 se cae (easy pega en casa). Si lose>25% ~15 iters → restore 951. No self-play, no APC, no MAX_UNITS este corte.

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
| 1 | Asalto sostenido: `army_attack_move` como `auto_support` (la red elige *cuándo* hay ejército; el entorno re-emite cada bloque) | **Hecho y corregido 2026-08-30.** El arranque (iter 442) sí subió wr. El visor a 817 mostró el vicio: re-order cada 80 ticks hacia `last_push` de la cabeza de celda (Ch6 propia) = 280 e1 en el mineral de casa, 0 enemigos visibles, beginner intacto en niebla. Destino ahora es enemigo visible / beacon; `army_attack_move` de grupo solo si hay ≥4 ociosos. | **Sí.** El wr 40–60% era “a veces el blob tropieza el beacon”. |
| 2 | Declaración: `win` si no queda producción enemiga N ticks | **Hecho y castrado.** `win_early` en `ExternalBotBridge` (500 ticks). La condición `eneProd==0` está **comentada** (falso positivo: `ProductionQueue` daba 0 con ConYard vivo). Solo queda patrimonio enemigo &lt;10% del propio. | **No ahora.** Con enB=14 el 10% no dispara. El engine hace bien en no declarar. Reabrir cuando el asalto deje enB≈0. |
| 3 | Horizonte vs γ: 624×80, γ 0.995–0.997, cortar cuando el engine declare | **Hecho** (γ=0.995). | Alargar sería contraproducente. |

`attack_move` normal = **un** `actor_id`. Si la red elige una recolectora, esa recolectora marcha. El grupo es `army_attack_move` (sin actor; el C# filtra `Harvester`).

Comandos que el engine ya tiene y la política v0.1 no muestrea (`ENABLED_TYPES`): `sell`, `repair` (sí auto_support), `set_rally_point`, `guard`, `enter_transport`/`unload`, `power_down` (sí auto_support), `set_primary`, `surrender`. `patrol` está en el proto y **no** en `ActionHandler`. Para cerrar partidas no falta un comando nuevo: falta que el push de ejército se sostenga.

---

## Qué hacer si a ~400–450 sigue plano

Un cambio de régimen por vez. **No tocar la red.**

**Corte ejecutado a iter 442.** Datos 401–443: win 8%, lose 32%, **incomplete 60%**, ownB 4.2, enB 8.4, army hist **0.01%**, H 1.71 (sana), wr era 0.089 (80/896). Reward **subió** porque timeout paga ~+3 vs lose −5: está farmeando el empate, no colapsando.

1. **Asalto sostenido de verdad (capa 0.1). HECHO, corregido 2026-08-30.**  
   `auto_support`: proc+harv **y** ≥4 combate → destino **enemigo visible / beacon** (nunca `last_push` de la red en `a_short`). Grupo `army_attack_move` solo si hay ≥4 ociosos; si no, `ATTACK_MOVE` a ociosos. Relanzar sobre **`latest.pt` ~817**.  
   Expectativa: que el blob **salga de casa**. Si el asalto convierte incomplete en lose (empuja y se muere) → parar el push automático, no sumar `w_timeout` en el mismo corte.

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

## Visor 2026-08-30 — el wr 60% no era un push

`best.pt` iter 780: 4/4, enB=0, wr20 0.70, reward +21. El live contra el mismo ckpt (caps `rl/ckpts/Screenshot 2026-08-30 091502.png` y `091710.png`) a ~43–47k ticks:

| HUD | Valor | Lectura |
|---|---|---|
| Enemigos vis. | 0/0 | niebla; no hay pixel rojo/naranja |
| `E:` del roster | tent×18 proc×8… = 33 | **edificios propios**, no el enemigo |
| Enemy value | 0 | patrimonio *visible* (niebla), no raze |
| Unidades | ~280–344 (`e1`×215–280) | blob en el mineral de casa, entre harvs |
| Recuadro naranja NE | beacon `(95, 11)` | ahí debería estar la base beginner |
| Reward | ~6.6 | incomplete camino al timeout, no win |
| Última acción | `build barracks` / `place scout` | sigue SimCity con 24–33 edificios |

No es un sensor del beginner de “ya perdí, no ataco”. La sonda de horizonte ya midió: en `a_short` el beginner **no empuja en 51k ticks** si nadie le toca la puerta. El agente nunca llega.

Causa en código (`auto_support.support_commands` pre-parche):

1. Destino = `last_push` de la política, después enemigo, después beacon.
2. La cabeza de celda no está condicionada a la unidad (`11-revision-quisquillosa.md`). Con 280 rifles, Ch6 (densidad propia) gana: `army_attack_move` cae **en el blob**.
3. `CreateArmyAttackMoveOrder` re-ordena **todas** las no-harvester cada macro tick. 80 ticks después el path se cancela. Hormiguero.

Por eso el 4/4 de 780 y estas partidas coexisten: si el click de celda cae cerca del beacon, razan; si cae en casa, SimCity hasta el tope. PPO no va a dejar el SimCity mientras el entorno le pida marchar al mineral propio.

Parche (relanzar ~817): `_push_cell` ignora `last_push` cuando hay beacon o enemigo visible; `army_attack_move` de grupo solo con ≥4 ociosos; si el blob ya camina, solo `ATTACK_MOVE` a ociosos (cap 16).

**Bug extra (live_games 829–831):** el parche no disparó. `ObservationSerializer` escribe `world.Map.Title` = `"Singles"`, y `BEACON_BY_MAP` claveaba `fase2_a_short.oramap`. Lookup fallía → fallback a `last_push` → `support_dests` = celdas de la red. Centroide final ~`(16,11)`–`(33,22)`, `dist_to_beacon` 62–79. Arreglo: `resolve_beacon` acepta Title `Singles` y el filename.

---

## Visor 835–836 — el parche sí dispara (4/4 win)

Cuatro partidas live post-relaunch (`rl/ckpts/live_games.jsonl`, ckpt `latest` 835–836, `map_name=Singles`, vs beginner):

| # | Iter | Resultado | Ticks | `support_dests[0]` | Dist. final al beacon | Centroide final | Unidades fin |
|---|------|-----------|-------|--------------------|------------------------|-----------------|--------------|
| 4 | 835 | **win** | 48972 | `[95,11]` | 67.8 (ya razó; rifles nuevos en casa tiran el promedio) | `(27.5, 17)` tras haber tocado `(79.5)` con 8 enemigos vis. | 94 |
| 5 | 835 | **win** | 21748 | `[95,11]` | **11.3** | `(84.0, 8.5)` | 72 |
| 6 | 835 | **win** | 45071 | `[95,11]` | **10.4** | `(84.7, 12.2)` (push tardío) | 70 |
| 7 | 836 | **win** | 23823 | `[95,11]` | **12.0** | `(89.5, 21.7)` | 75 |

Las 3 de antes (829–831, Title miss) fueron incomplete 51k, dist 62–79, `n_ene_vis=0`, 300 rifles en el mineral.

`n_support_army` bajó 285–400 → 25–55: el grupo ya no se re-emite cada 80 ticks; el path llega. Train en paralelo (mismo código): 837 fue 4/4, enB=0, raze 5.67 → `best.pt` nuevo; wr20 ~0.50–0.60. λ_bc=0, SIL 0.5.

### Qué ya estaba en el train (beacon) vs qué pide relanzar (hunt)

Visor y `rl.train` comparten el módulo. El 4/4 al beacon ya corre. El hunt ~854 **no** está en el proceso vivo hasta Ctrl+C + `auto_train` / visor.

| Pieza | Dónde | Estado |
|---|---|---|
| Destino de asalto | `auto_support._push_cell` → `resolve_beacon` | vivo: enemigo / beacon, nunca `last_push` en mapa A |
| Hunt post-pile | mismo `_push_cell` | **en disco**, pide relanzar |
| Keep-alive | `support_commands` | vivo: grupo solo con ≥4 ociosos |
| Clave Title | `BEACON_BY_MAP["Singles"]=(95,11)` | vivo |
| Receta | `auto_train.TRAIN_ARGS` | a_short / beginner / eradicate_v4 / auto-support / PPO+SIL / 1200 |

`policy_push_cells` / `last_push` en casa **no es un fallo del parche**. Es la red. El asalto lo ejecuta el entorno.

---

## Hunt ~854 — pile-up en el beacon, timeout con enB>0

A iter 854: wr era 0.301 (302/1004) al alza, wr20 ~0.40. Wins reales (837 4/4, 845 3/4, 855 3/4). El incomplete que queda **no** es “no salieron de casa”:

- Iters con incomplete: enB iter-promedio 5–10 (hay edificios enemigos vivos).
- Live 851 win: dist_beacon **2.8**, 230 e1 apilados en `(91.9–92.3, 11–14)`, `n_ene_vis=0`.
- Live 851 incomplete: dist 53 (otra partida ni llegó). El caso que vimos en el visor: blob en el naranja, beginner con un `powr`/`tent` en niebla 15–25 celdas al sur, reloj a 51k.

Causa: `_push_cell` era `visible[0] → beacon`. Attack-move a la celda donde ya están = idle. OpenRA no explora niebla sola. `enemies[0]` además prefería un scout en el medio (`[48,29]`) sobre un edificio en la base.

Parche (mismo `auto_support`, no es Capa 2):

1. Edificio enemigo visible, el más cercano al beacon.
2. Unidad visible, la más cercana al beacon. Si el ejército **ya está** en el beacon y esa unidad está a >20 celdas (stray), se ignora.
3. Si ≥4 combate a ≤8 celdas del beacon y no hay objetivo: waypoint `HUNT_OFFSETS[tick // 1600]` al sur/oeste del NE (`x≥40`, `y≤36`, nada de mineral de casa).
4. Si todavía van de camino: beacon, igual que antes.
5. `last_push` solo sin mapa A.

En el visor, después del relanzar: `support_dests` ya no se queda en `[[95,11],[95,11],…]` cuando el centroide está encima. Debería aparecer `(95,25)`, `(79,17)`, … y `dist_beacon` puede **subir** un poco (barren el sur) mientras enB baja. Success = menos incomplete a 51792 con enB>0, no que el centroide se clave en 11.

No es `win_early`: el engine tiene razón, quedan edificios. No tocar la red.

---

## Órdenes vs scripted — cuándo, no “después”

El C# (`ActionHandler`) ya traduce casi todo lo que usa `examples/scripted_bot.py`. El agujero no es el puente: es **quién aprieta el botón**. El scripted emite 12 prioridades por tick; la red emite **1** cada ~80 ticks. APM tonto va a `auto_support` (Capa 0). Lo que necesita “esta unidad, ese actor” espera al pointer (Capa 2). **No** meter tipos nuevos en `ENABLED_TYPES` para que PPO los aprenda. **No** hay Capa 4: el plan es 0→1→2→3. Nada de esta lista puede llegar a Capa 3 pendiente.

Cortes (un régimen por vez):

1. **Hecho (corte 923):** dest pasable (SIL) + rally al dest + stance AttackAnything al spawn + sell ruinas. Resume 899.
2. **Hecho (corte 939):** harv no marcha al dest (rally `weap` off, credit skip harv/mcv, adapter harvest). Resume 899.
3. **Hecho (corte 987):** muros no van a TRAIN; auto-tent tras proc. Resume 899.
4. **Hecho (corte 947):** pack 12 + rally staging. Resume 899.
5. **Hecho (corte 977):** clamp ratio PPO + SIL skip nll saturado. Resume **922**.
6. **Hecho (corte 1010):** Capa 2 transformer + scatter + `celda|unidad` (Net2Net 922).
7. **Hecho (corte 950):** asalto de soporte OFF (pack/hunt/rally/dest-credit/recall). Eco/micro on. Resume **922**.
8. **Hecho (corte 952):** Capa 2 vs beginner cerrada (wr 88%). Resume **951**, Capa 3 **easy**.
9. **Siguiente:** juzgar easy (lose media, `defense_loss`, harvest_edge). No QSA/APC/self-play el mismo corte.
10. **Capa 2b / 3b:** hard cuando wr20 vs easy se sostenga; indexer QSA / `MAX_UNITS` si el incomplete es 300 e1 ciegos; APC/guard después.

| Orden | Dónde | Cuándo | Notas |
|---|---|---|---|
| `set_rally_point` | **support**, no la red | **Hecho corte 923; weap off 939** | e1 (tent/barr/kenn) spawnea y camina al dest. **No** weap/hpad/syrd: HARV sale de Vehicle. |
| `set_stance` AttackAnything al nacer | **support** | **Hecho corte 923** | `target_x=3` si `stance≠3`. Un unit/bloque. |
| `sell` HP muy baja | **support** (como repair) | **Hecho corte 923** | hp&lt;0.12, no fact/proc. Un sell/bloque. |
| `guard` | **no es tipo nuevo ahora** | Capa 0 ya tiene recall a casa. “Dejá 2 en la fact” = regla de dest, no `GUARD` | A la política, **después de Capa 2** (unidad + edificio). |
| `set_primary` | **no** | Solo si TRAIN pega a un solo tent | El C# recorre colas y manda al primer edificio que pueda. Con 16 tents no es el cuello. |
| `enter_transport` / `unload` | política, **después del pointer** | **Capa 2 cerrada** (smoke 20 iters, `last_push` sigue al sujeto) **y** wr20 vs beginner ≥0.40 | Sin `celda\|unidad` el APC es un click a Ch6. No es Capa 0. Mask v0 los tiene apagados a propósito. |
| `surrender` | **nunca** en la política | — | Mask apagado. El win lo declara el engine. |

Ya cubierto en support (no reabrir): repair HP&lt;35%, harvest idle, power_down brownout, deploy MCV, auto-proc/harv, auto-tent, `army_attack_move` de grupo, dest credit (no harv/mcv per-unit), defend recall, re-asalto, rally tent-only. Muros = BUILD.

---

## Deuda de la cabeza de celda (Capa 2 — corte 1010)

Fuente: live 835–836 + `network.py` `dist_cell` + `11-revision-quisquillosa.md` + Capa 2 del doc 12. El chasis `AlphaLiteNet` es familia correcta para 1×2070. El click en casa es sesgo inductivo incompleto **más** crédito roto, no “tirar la red”.

### 1. El autorregresivo es de mentira en el paso que importa

```
P(tipo) → P(unidad | tipo) → P(celda | tipo) → P(ítem | tipo)
```

`dist_unit` sí mira el rifle (HP, idle, `cell_x/y`). **`dist_cell` no.** Recibe fmap U-Net + embedding de tipo (64-d, idéntico en las ~5k celdas) + GRU broadcast. Un MCV y un rifle ven el mismo heatmap. AlphaStar condiciona el pointer al sujeto. El 12 ya lo pone: *Concat del embedding del slot elegido*.

### 2. Ch6 es una montaña, el beacon es un topecito

Canales: Ch5 edificios propios, Ch6 densidad propia (a veces 200+ e1 en el ore), Ch7/Ch8 enemigo + parche pedagógico 7×7 a 0.85 en el beacon. `cell_head` es conv 1×1: puntúa celdas “raras”. Lo más raro durante 50k ticks es el blob de casa. El NE no gana esa pelea espacial aunque el GRU “sepa” que hay que atacar.

### 3. La misma cabeza pone edificios y ataca

`TYPES_USE_CELL` incluye `place_building` (click **en casa**) y `army_attack_move` (click **al beacon**). El único switch es el embedding de tipo, 64 números vs 96 canales de mapa. El gradiente de “poné el barracks acá” es fuerte y constante; el de “mandá el ejército allá” es raro. Aunque el tipo *podría* gatear Ch5 vs Ch7, el aprendizaje es cuesta arriba.

### 4. 48 slots, media, densidad anónima

`MAX_UNITS=48`, promedio enmascarado, sin atención. Con 90–300 actores la red no ve el ejército: ve un vector. Ch6/Ch8 son conteo, no identidad. El 12 pide transformer 2 capas / 4 heads / d=64 sobre 48 slots **y** scatter (pintar cada unidad en el fmap). No un ViT ni 50M params. Agujero = set de unidades, no el terreno.

### 5. El win no acredita la celda (PPO y SIL)

El visor guarda `policy_push_cells` en casa y `support_dests` en `(95,11)`. PPO/SIL ven la acción **de la red** (casa). El engine gana por el **soporte** (beacon). El gradiente lee: “clickeaste el mineral y ganaste”. SIL clona exactamente esas celdas. Refuerza el vicio.

**Hecho en el corte 935** (Capa 0/1, no transformer): `apply_dest_credit` reescribe el `cell_flat` de `army_attack_move`/`attack_move` al dest de soporte **y** muta el comando de política, con recálculo F1 de log π. PPO/SIL y el visor ven la misma celda. **No** mezclar con Capa 2.

### 6. La pedagogía Ch7/Ch8 estuvo apagada casi todo el run

Hasta la clave `"Singles"`, `rollout._batch_of` hacía `BEACON_BY_MAP.get(obs.map_info.map_name)` y `map_name` era Title `"Singles"` → `beacon=None`. El visor pintaba el naranja con el filename del `.oramap`; el tensor de la política no. Recién post-relaunch el mapa de train tiene el parche. Unas ~20 iters no reescriben una cabeza entrenada a clickear Ch5/Ch6.

`_batch_of` ya usa `resolve_beacon(obs)` (corte 900). Seguir usándolo en Capa 2; no volver al `.get` por filename.

### Qué sí está bien (no reescribir de raíz)

U-Net + CoordConv (RF ~40 celdas), máscaras, `log_prob` solo de cabezas activas (F3), GRU 416, vocab por roles, `army_attack_move` + `auto_support` fuera del presupuesto de PPO. Tirar la red ahora pierde eco/`train` que ya funcionan; el asalto lo sigue cargando el entorno.

### Cuándo sí tocar la red (Criterio del 12, no del 4/4 del visor)

Un PR, ~3–4M params extra, **después** de wr20 vs beginner sostenido (el 12 pide wr>30% para Capa 3; para Capa 2 basta que el push del *entorno* no sea un spike):

1. Concat slot elegido (o su `(x,y)`) en `cell_head`.
2. Scatter de unidades en el fmap.
3. Transformer de entidades 2 capas / 48 slots.

Crédito de dest (punto 5) **ya está** (corte 935). Capa 2 (corte 1010) mete transformer + scatter + pointer **en un PR** como el 12, con residual/zero-init para no borrar el GRU 922. Indexer QSA = corte aparte.

### Qué robar de Qwen3.8-Flash-Next (cuando Capa 2, no ahora)

Fuente: [blog](https://qwen.ai/blog?id=qwen3.8-flash-next) + [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) (2026-08-26). Es un MoE 125B (6B activos) para 1M tokens. **No copiar el modelo.** Robar el patrón: *GDN recuerda (estado fijo), QSA recupera (pocos bloques importantes)*. Eso es el agujero de `AlphaLiteNet`, mal implementado (GRU comprime a un vector; `cell_head` hace softmax sobre ~5k celdas de Ch6).

| Pieza Qwen | Qué hace | ¿Capa 2 en 2070? |
|---|---|---|
| **GDN** (3/4 capas) | Atención lineal, delta rule (borra/reescribe una key, no acumula) | **2b**, no el primer PR. Upgrade del GRU 416 (asociar “vi un powr en (88,22)”). A 624 steps no hace falta linealidad. Si se hace: short-conv en q/k antes de comprimir, L2-norm q/k, gate de salida **sigmoide** no SiLU |
| **QSA** (1/4) | Indexer a micro-bloques → top-k → atención densa solo ahí | **Sí, sobre el mapa**, no sobre 48 unidades. `48²` es gratis (el 12 ya pide atención densa al set). El pointer: bloques 8×8, indexer condicionado a **tipo + slot**, top-8, conv 1×1 ahí. `place` mira Ch5; `army_attack_move` mira Ch7/Ch8. **No compartir el índice** U-Net ↔ transformer (el paper: IndexShare falla en híbridos) |
| **Gated Residual** (4 ramas) | Residual ancho + gate sigmoide al leer | Solo **GatedNorm** (`RMSNorm ⊙ sigmoid(MLP)`) en el residual del transformer nuevo. 4 ramas Hyper-Connection en 6–8M es teatro |
| **N-gram / Engram 51B** | Tabla de frases, off-GPU | No. Scatter + 10-d de unidad cubren el patrón local |
| **MoE 512 / Muon** | Escala LLM | No. AdamW + AMP. Muon solo si el transformer ya está y el crítico se pone raro |
| **ViT sobre el mapa** | — | No. U-Net RF~40 está bien; el agujero es el set |

Receta QSA que sí copiar (no el mismo resume que el pointer): ellos destilan atención densa → indexer y después entrenan sparse. En Capa 2: primero `celda\|unidad` densa (PR del 12); si Ch6 sigue ganando, ahí el indexer de bloques.

Orden del PR cuando toque (el 12 + media pieza de Qwen):

1. Transformer entidades 48 slots (denso).
2. Scatter al fmap.
3. `dist_cell \| unidad` (concat). Sin esto el indexer de (4) elige Ch6 igual.
4. *Opcional si (3) queda corto:* indexer de bloques espaciales condicionado a tipo+unidad.
5. GatedNorm en el residual nuevo.
6. **No** GDN, MoE, Muon ni n-gram en ese PR.

`_batch_of` ya usa `resolve_beacon` (no reintroducir `.get` por filename).

---

## Relación con Run 7 / Run 8

| Run | Síntoma | Lección |
|---|---|---|
| 7 (201–310) | 100% `attack_move` sin base | combate legal antes de proc = reward-hack. Mask hasta proc. |
| 8 (220–408) | 99% `no_op`+deploy, H≈0, −2.57 &gt; pelear (−6) | el mask sin watchdog de *este* modo convierte spam-attack en spam-noop. Morir desnudo era más barato que jugar. |
| 9 (220–…, este) | H sana, eco viva, wr~11% plano, incomplete 36% | mitigaciones de colapso OK. El techo ahora es APM de asalto, no PPO ni horizonte. |

Ckpts del Run 8: `rl/ckpts/Run 8 (a_short collapse no_op 220-408)/`. Resume vivo: `best.pt` / `latest.pt` = iter 219 (luego 229 si el 4/4 sigue siendo best).

---

*Guardado: 2026-08-31 — rama `exp/rl-2026-08-28-grok`. Companion de `12-plan-4-capas-siguiente-nivel.md`; no modifica el plan. Corte 1010: Capa 2 Net2Net desde 922 (Run 18 plateau archivado).*
