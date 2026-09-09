# RA Aliados: contrato cerrado

> **Fecha:** 2026-09-09
> **Para quién:** antes de un run que pretenda ser “el agente juega OpenRA mod RA de Aliados”.
> **Relacionado:** [`facciones-mods-roles.md`](facciones-mods-roles.md); [`ra-completo.md`](ra-completo.md); onboard [`../start/onboard.md`](../start/onboard.md).
> **Estado:** features de **código** del agente Aliados **cerradas**. No queda tintero de chasis. Curriculum / tapes / otro ckpt no son huecos de este corte.

---

## Veredicto

El chasis land (P0–P4 de `23`) **está mergeado** en `alphalite-v2`. Encima de eso, el agente Aliados ya **no** es un contrato de papel:

1. El lobby **traba** el bando: slot `rl-agent` / `rl` → `RandomAllies` (england / france / germany). `Random` explícito también se reescribe.
2. Uniques aliados **no** caen a `misc` ni a TRAIN-as-building (`e7`, `medi`, `mech`, `pdox`, fakes, …).
3. APC `enter_transport` / `unload` están en `ENABLED_TYPES` y emiten orden.
4. Capture (ingeniero) e Infiltrate (spy) remapean el `ATTACK` con actor.
5. Superpoderes listos (`Chronoshift`, nuke, Iron, spy plane, …) son `SUPPORT_POWER`. GPS se dispara solo al cargar.
6. `patrol` = AttackMove ida + AttackMove queued vuelta.
7. Chrono Tank (`ctnk`) `deploy` + celda → `PortableChronoTeleport`.
8. `IDENTITY_ITEMS`: Tanya / medic / mech / spy / Chronosphere / Allied tech / Chrono Tank / Phase / MH60 **no** se esconden detrás del concreto más barato del rol.

Para **lanzar** el train falta rebuild C# / Docker (lobby + órdenes nuevas). Python solo no alcanza.

---

## Contrato (no se discute en el run)

| Cosa | Regla |
|---|---|
| Mod | `ra` solamente. `cnc`/`d2k` = otro ckpt. |
| Bando del **agente** | Aliados. Lobby: `RandomAllies` → england / france / germany. |
| País | Irrelevante para la red (roles). Las uniques son un ítem más; las de `IDENTITY_ITEMS` se eligen por internName. |
| Rival | Sigue `Random` (bot scripted). No es “solo vs soviet”. |
| Red | Nunca ve `england` / `russia`. Ve roles + `available_production` (+ internName en IDENTITY_ITEMS). |
| Random wrappers | Nunca van al vocab. Se resuelven **antes** del primer tick. |

Default de código: `player_faction=""` en proto → C# usa **`RandomAllies`** en el slot `rl-agent`. Pasar `Random` explícito **también** se reescribe a `RandomAllies` (el contrato gana). Python: `rl.allies.resolve_player_faction` espeja C#. Para mezclar soviet más adelante: otro corte, `player_faction=RandomSoviet` o `russia`.

---

## Qué está implementado

### A. Lock de facción

- Proto: `CreateSessionRequest.player_faction` (4) y `enemy_faction` (5). Vacío = defaults.
- C# `RLSessionManager.ResolvePlayerFaction`: vacío / `Random` → `RandomAllies`. `england` / `france` / `germany` pasan.
- `SetupLobbyInfo`: slot `rl-agent` / `rl` → player faction; el resto → enemy faction (default `Random`).
- `LockFaction` del mapa, si existe, sigue ganando (`SyncClientToPlayerReference`).
- Python: `openra_env.config.GameConfig.player_faction = "RandomAllies"`; `bridge_client.create_session` lo manda.
- Rebuild **C# / Docker** obligatorio: un hot-patch de Python no cambia el lobby.

### B. Catálogo (anti-crash + roles)

| InternName | Qué es | Rol de entidad (`ROLE_OF_ITEM`) | Cabeza de producción |
|---|---|---|---|
| `e7` | Tanya | `specialist` | internName (`IDENTITY_ITEMS`) |
| `medi` | Médico | `infantry_basic` | internName |
| `mech` | Mecánico | `specialist` | internName |
| `spy` | Espía | `specialist` | internName |
| `pdox` | Chronosphere | `tech` | internName |
| `atek` | Allied tech center | `tech` | internName |
| `gap` | Gap generator | `tech` | internName |
| `ctnk` | Chrono Tank (Germany) | `tank_medium` | internName |
| `stnk` | Phase Transport | `transporter` | internName |
| `mh60` | Longbow | `heli` | internName |
| `iron` / `mslo` / `stek` | Iron / silo / Soviet tech | `tech` | internName (por si el rival/tree los ofrece) |
| `fpwr` `tenf` `syrf` `spef` `weaf` `domf` `fixf` `fapw` `atef` `pdof` `mslf` `facf` | Fakes Francia | `civic` | rol `civic` (`cheapest_of`) |

`pdox` / `iron` / `mslo` y fakes entran a `BUILDING_ITEM_TYPES`. Sin eso, BUILD Chronosphere iba por TRAIN (mismo clasazo que `sbag` visor 985).

**`IDENTITY_ITEMS` vs `cheapest_of`:** el traductor de `15` sigue vigente para el resto del árbol (pbox vs gun vs agun). Las uniques de la tabla **no** se pliegan al rol barato en el item-head. El xf **sigue** viendo el rol de entidad; ids de `ROLE_VOCAB` no se tocan. Vocab de ítems: roles primero, IDENTITY_ITEMS **al final** (append-only). Combat TRAIN de esas uniques sigue gated hasta proc+harv (`COMBAT_TRAIN_ROLES`).

### C. APC load/unload

`enter_transport` y `unload` **ya** estaban en `ACTION_TYPES` (antes de `army_attack_move`). Están en `ENABLED_TYPES` + máscaras + `index_to_command`.

- `enter_transport`: infantería propia + transporte (`apc`, `lst`, `tran`, `stnk`). Passenger = unit head; target = transporte más cercano.
- `unload`: transporte con `passenger_count > 0`.

El teacher **sigue filtrando APC** en a_short (rush intacto). PPO puede TRAIN `apc` si hay weap; ahora además puede cargarlo.

### D. Capture / Infiltrate (sin ActionType nuevo)

`CreateAttackOrder` con `target_actor_id`:

- sujeto con trait `Captures` → orden `CaptureActor` (ingeniero);
- sujeto `spy` → orden `Infiltrate`;
- si no, `Attack` como siempre.

La política ya emite `attack` con actor. No hay cabeza nueva.

### E. Superpoderes

- Proto: `GameObservation.ready_support_powers` (23); `ActionType.SUPPORT_POWER` (24).
- C# `ObservationSerializer.SerializeSupportPowers`: keys de `SupportPowerManager` con `Ready`.
- Adapter: type-head `support_power` enmascarado si no hay ready (GPS filtrado: el engine lo dispara solo en `GpsPower.Charged`).
- Pick: Chronoshift → AdvancedChronoshift → Nuke → Iron Curtain → spy plane / paratroopers / parabombs → primer ready no-GPS.
- Chronoshift: `actor_id` = unidad fuente, celda = dest; C# pone `ExtraLocation` = footprint de origen. Dest fuera de mapa → fact propia.
- GPS **no** es un click del agente.

### F. Patrol

- Ya existía `ActionType.PATROL` (proto 22). Ahora está en `ACTION_TYPES` (append) + `ENABLED_TYPES`.
- C# `QueuePatrol`: AttackMove a la celda + AttackMove queued a la posición actual.

### G. Chrono Tank

`deploy` + celda sobre `ctnk` → `PortableChronoTeleport`. MCV / resto sigue `DeployTransform`. `deploy` usa celda (`TYPES_USE_CELL`).

---

## Fuera de este corte (no es feature incompleta del agente Aliados)

| Cosa | Por qué no es tintero |
|---|---|
| Teacher rush-first (`powr`→`proc`→`tent`→`e1`) y filtro APC | Curriculum. El chasis ya puede TRAIN/cargar APC y uniques. |
| Pointer de ataque 2c-C | Revertido a propósito (smoke wr20→0). |
| Soviet mezclado / `cnc` / `d2k` | Otro régimen / otro ckpt ([`facciones-mods-roles.md`](facciones-mods-roles.md)). |
| SFT navy/air | El teacher ya emite syrd/hpad/ship/heli en mapas `naval_viable`. Las tapes salen de `--onboard` con `--map-pool water`, no de más código. |
| SFT corto sell/repair/rally/guard | El path existe (P1/P2). Las tapes se graban en el run. |

---

## Cómo correr el train (post-rebuild)

Imagen / binario OpenRA **nuevo** (C#). Python solo no alcanza.

```powershell
# daemons con el bridge recompilado (lobby + SUPPORT_POWER + patrol + Chrono Tank)
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d --build

# mismo camino que start/onboard.md (Allies lock es default, no hay flag nuevo obligatorio)
.venv\Scripts\python.exe rl\auto_train.py --scratch --onboard
```

Criterio de que el lock funciona: en el visor / `GetState.player_faction` el agente es `england|france|germany`, nunca `russia|ukraine`. Rival puede ser cualquiera.

Resume de un ckpt land: **partial load**. `patrol` + `support_power` se **appendean** al type-head (`adapt_v2_state_dict` padea). `IDENTITY_ITEMS` entran al vocab de ítems al final (no reordenan roles). `enter_transport`/`unload` ya existían (no crecen type-head).

Un régimen por vez: este corte crece type-head. Scratch o resume con `strict=False`. **No** mezclar reward/arch el mismo resume.

---

## Mapa de código

| Capa | Archivo |
|---|---|
| Contrato Python | `rl/allies.py` |
| Roles + IDENTITY_ITEMS | `rl/roles.py` |
| Adapter (ENABLED, pick, emit) | `rl/action_adapter.py` |
| Type-head | `rl/network.py` `ACTION_TYPES` |
| Proto | `proto/rl_bridge.proto` y copia OpenRA `…/Protos/rl_bridge.proto` |
| Lobby | `RLSessionManager.ResolvePlayerFaction` |
| Órdenes | `ActionHandler` (Capture/Infiltrate, QueuePatrol, CreateSupportPowerOrder, PortableChronoTeleport) |
| Obs | `ObservationSerializer.SerializeSupportPowers` |
| Config | `openra_env/config.py`, `openra_env/server/openra_process.py`, `config.yaml` |
| Tests | `rl/tests/test_ra_allies.py`, catálogo en `rl/tests/test_roles.py` |

---

## Tests

- `rl/tests/test_roles.py` — catálogo Aliados sin `misc`.
- `rl/tests/test_ra_allies.py` — lock, catálogo, APC, IDENTITY_ITEMS, patrol, Chronosphere, Chrono Tank.

*Guardado: 2026-09-09 — features de código cerradas (lock + catálogo + APC + capture + superpoderes + patrol + Chrono Tank + IDENTITY_ITEMS). Rama `alphalite-v2`.*
