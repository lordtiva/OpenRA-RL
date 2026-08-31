# Facciones, bandos y mods — cómo lo resuelve el agente

> **Fecha:** 2026-08-31 · El train de `a_short` / Singles es **solo `ra`**, ckpt **Aliados**, spawn SW. Este doc es la regla para no romper el resume cuando (si) se toca otro bando o mod.
> **No** es un plan de entrenar los 3 mods. Es el contrato: la red nunca ve `england` / `russia` / `gdi`.

---

## ¿Hay que reentrenar cada mod / bando / facción?

**No.** No son 10 trains. Hoy es **un** proceso (`ra` + Aliados). El resto se comparte o se pospone.

La red no aprende “soy england”. Aprende **roles** (`infantry_basic`, `defense_gun`, `barracks`). En cada partida el engine le pasa solo lo construible (`available_production`) y el adapter elige el concreto barato (`pbox` vs `ftur`). Eso ya está.

| Qué | ¿Otro train de cero? | Qué hacer |
|---|---|---|
| **País** (england / france / germany, o russia / ukraine) | **No** | Mismo ckpt. El árbol es el del bando; la unidad especial es un ítem más del rol. |
| **Bando RA** (Allies ↔ Soviet) | **No** de cero | Mismo ckpt y mismos roles (`tent`↔`barr`, `pbox`↔`ftur`). Más adelante **mezclar partidas** soviet para que no se acostumbre solo a Aliados. El beacon es el **spawn**, no el bando. |
| **Random / RandomAllies / …** | **Nunca** | El lobby elige un país y recién ahí arranca. No son un ejército. |
| **Otro mod** (`cnc`, `d2k`) | **Sí, otro ckpt** | Otro catálogo de actores. Se reusa la red (mismos heads) y se amplía `ROLE_OF_ITEM`. No se pega el 999 de RA y a entrenar Nod. |
| **`ts`** | No | Fuera del bundle, incompleto. |

Orden:

1. **Ahora:** un train, `ra` Aliados, a_short. Cubre los 3 países aliados sin enterarte.
2. **Cuando gane al easy/hard:** mismos pesos, a veces soviet en el mismo mapa (beacon según slot). No un segundo proyecto.
3. **Si algún día GDI/Nod o Dune:** un ckpt nuevo **por mod**, no 2× o 3× facciones dentro del mod.

El trabajo pesado (PPO, Capa 2, SIL, el 999) **no se replica por facción**. Se replica, como mucho, **por mod**, y solo cuando quieras ese juego.

---

## Tres capas, no una

| Capa | Ejemplo | Quién la resuelve | ¿La red la ve? |
|---|---|---|---|
| **Mod** | `ra` / `cnc` / `d2k` (`ts` fuera del bundle) | El daemon / `--mod`. Un ckpt por mod. | No. Otro árbol de actores = otro `ROLE_OF_ITEM`. |
| **Bando (`Side`)** | RA: Allies / Soviet. TD/D2k: la facción *es* el bando | Lobby + `available_production` del engine | No. Ve roles (`barracks`, `defense_gun`). |
| **Facción (`InternalName`)** | `england` / `france` / `germany` / `russia` / `ukraine` | Lobby. `Selectable: False` en los padres `allies`/`soviet` | No. La unidad especial cae a un rol (`specialist`, `tank_heavy`, …). |
| **Random\*** | `Random`, `RandomAllies`, `RandomSoviet` | El lobby **antes** de `reset`. Nunca son un ejército | Nunca. No van al vocab. |

Los Random no se entrenan ni se listan en `ROLE_OF_ITEM`. Si el server recibe `RandomAllies`, que resuelva a `england|france|germany` y recién ahí arranque la partida.

---

## Lo que ya hace el código (no rehacer)

1. **`rl.roles.ROLE_OF_ITEM`** — `e1` y el rifle soviético son `infantry_basic`; `pbox`/`gun`/`hbox`/`agun` son `defense_gun`; `ftur`/`tsla`/`sam` son `defense_turret`. La cabeza de ítems indexa **roles**.
2. **`available_production`** — el engine ya filtra por facción. El adapter solo ve lo construible *ahora*.
3. **Concreto más barato** (corte 1002): `cheapest_of` usa el costo de `openra_env/game_data.py`. Allies con tent → `pbox` $600, no `agun`. Soviet con barr → `ftur`, no `tsla`.
4. **PLACE Building + Defense** (mismo corte): las torretas viven en la cola `Defense`. PLACE las planta. Antes se encolaba `gun` y se plantaba `tent`.

Spawn / beacon **no** son facción: son geometría del mapa. En A-short el SW tiene mineral en casa y el NE es `(95,11)`. Invertir el slot = el dest mira tu fact. Eso se arregla con beacon-por-spawn, no con un embedding de `ukraine`.

---

## Cómo crecer sin romper el 999

| Querés | Qué hacer | Qué no |
|---|---|---|
| Jugar **soviet** en el mismo mapa | Mismo ckpt (roles). Beacon según **slot**, no según bando. Visor: PPO en SW. | Un `faction_id` en los escalares. |
| País (england vs germany) | Ignorar. El árbol aliado es el mismo; la unidad especial es un ítem más del rol. | Un head de 5 países. |
| **cnc** o **d2k** | Nuevo `ROLE_OF_ITEM` (gdi/nod, casas). Mismo `AlphaLiteNet` si los roles caben. Ckpt **aparte** (Net2Net solo de módulos iguales). | Mezclar `ra`+`cnc` en un resume. |
| `ts` | No. Fuera del bundle, incompleto. | — |
| Liga multi-mod | Capa 3 self-play **dentro de ra** primero. | 3 mods × 7 facciones tabula rasa. |

Un régimen = un mod + un mapa + un bot. El traductor de roles es el que hace que soviet/allies no sean dos redes.

---

## Conteos (bundle oficial)

| Mod | Bandos reales | Facciones jugables | Random (wrappers) | No seleccionables |
|---|---|---|---|---|
| ra | 2 (Allies, Soviet) | 5 países | 3 | 2 padres |
| cnc | 2 (GDI, Nod) | 2 | 1 | 0 |
| d2k | 3 Casas | 3 | 1 | 4 |
| **bundle** | **7** | **10** | **5** | **6** |
| ts (fuente) | 2 | 2 | 1 | 0 |

Hoy: **1** de esas 10 (`ra` Allies, da igual england/france/germany).

*Guardado: 2026-08-31 — corte PLACE Defense + `cheapest_of`. Rama `exp/rl-2026-08-28-grok`.*
