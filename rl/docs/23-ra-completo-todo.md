# 23 — RA completo: gaps vs el train land actual

> **Para quién:** cuando quieras ampliar el action set / mapas más allá del land-only
> de Phase A/B (naval, air, buildings UI, micro).
>
> **Fecha:** 2026-09-08 (P0–P3 cerrados 2026-09-09 en `exp/ra-completo-p0`; P3 teacher tech/defense + map-pool).
>
> **Relacionado:** onboarding land `22-onboard.md`; operación diaria `07-operacion.md`;
> índice `README.md`.

---

## Base reutilizable (no tirar el train)

El Phase A/B land que corre hoy es una **base reutilizable**, no queda obsoleto al
cerrar gaps de RA completo:

- Los pesos / tapes de land **siguen útiles** (eco, push, macros de army/infantry/vehicle).
- Tipos de acción **nuevos** pueden pedir crecer la type-head + carga parcial
  (`strict=False` / Net2Net-style) y un SFT corto sobre los tipos nuevos.
- Las tapes land no se tiran: navy/air necesitan **tapes nuevas** (mapas agua/aire,
  teacher o demos con esos órdenes).
- **No hace falta parar** el train land en curso para documentar o priorizar esto.

---

## Harvesters — aclaración de semántica

| Cosa | Qué hace de verdad |
|------|--------------------|
| `harvesters_move` | **Todas** las harvs MOVE a la **misma** celda (macro de grupo). |
| Una sola harv + celda | Via **unit head**: `move` / `attack_move` / `attack` sobre un harv emite **MOVE** a `(cx,cy)` para **ese** `actor_id` (ya no se reescribe a `harvest` sin celda). |
| `harvest` | Usa el harvester **seleccionado** si es válido; pasa `target_x/y` (CommandModel/HARVEST lo acepta). Si la cabeza no apunta a un harv, fallback `_any_harvester`. |

P0 (rama `exp/ra-completo-p0`): path singular harv→celda **existe** (unit head + `harvest`+cell). El macro de grupo sigue siendo all→same cell.

## TODO priorizado (P0 → P4)

### P0 — Naval / air macros + bucket vehículo; harvest por harv (opcional) — DONE

- [x] Macros de grupo **naval** y **air** (análogos a `army_*` / `infantry_*` / `vehicle_*`).
- [x] Separar el bucket **vehicle** (hoy mezcla tierra; no alcanza para navy/air).
- [x] (Opcional) `harvest` / move con **celda por harvester** (no solo `harvesters_move` all→same cell).

**Estado:** cerrado en `exp/ra-completo-p0` (rebase sobre `alphalite-v2` / AMP `dfb2b02`). Tip: `fe02a7c`. Tests: `test_ra_completo_p0`, `test_alphalite_v2`, `test_onboard`.

### P1 ? Building slot head (sell / repair / rally / power_down / set_primary) ? DONE

- [x] Path de **building slot**: `sell`, `repair`, `set_rally_point`, `power_down`, `set_primary` en `ENABLED_TYPES`; `ActionIndex.building_ids` + m?scara de tipo si no hay edificios; `index_to_command` emite `CommandModel` correcto (`actor_id`; rally + `target_x/y`).
- [x] Sampling/act/eval: **`building_head` dedicado** (`building_mlp` + `building_scorer`) sobre `building_feats` / `building_valid` cuando el tipo ? `TYPES_USE_BUILDING` (ya no reusa unit feats/scorer). Slot sigue en `unit_slot` ? adapter `building_ids`.
- [x] `command_to_indices` mapea `actor_id` ? slot de `building_ids` (SIL/teacher).
- [x] Cabeza neural **dedicada** a edificios + soft kind-masks (sellable / can_produce / power) sin vaciar filas.
- [x] Partial load: `building_*` ausente en land ckpts ? missing keys bajo `strict=False` (soft-add / init fresco). `adapt_v2_state_dict` no toca type-head size.
- [x] No mezclar con el mismo corte que crece type-head + maps ? P1 reusa filas ya existentes en `ACTION_TYPES` (sin crecer type-head).
- [ ] Caveat: kind-masks son heur?sticas suaves (no engine-perfect); SFT corto sobre sell/repair/rally recomendable antes de confiar en prod.
- [ ] Caveat: `set_rally_point` condiciona celda con pool de unidades (no embedding de edificio) ? suficiente para P1.

**Estado:** DONE en `exp/ra-completo-p0`. Tests: `test_ra_completo_p1` (+ load-compat smoke). Land macros intactos. Guard → P2 (DONE).


### P2 — Micro: group stance / stop, focus fire (+ guard) — DONE

- [x] Stance / stop a nivel **grupo**: macros append-only `army_stop` / `army_set_stance` (N× STOP / SET_STANCE sobre combat via `group_actor_ids`). `set_stance` single ahora emite `target_x` (stance; default AttackAnything).
- [x] Focus fire: `attack` enmascara sin enemigos visibles; resolución `_focus_enemy_at_cell` elige `target_actor_id` (prefer wounded / nearest to cell) en vez de solo attack-move a celda.
- [x] Habilitar RL **`guard`** en `ENABLED_TYPES` + `index_to_command` (`actor_id` escort + `target_actor_id` harv/MCV/building). Máscara: hace falta combat + target válido. Forma grupo opcional: `army_guard` (N combat guard same target).

**Estado:** DONE en `exp/ra-completo-p0`. Tests: `test_ra_completo_p2` (+ p0/p1/alphalite smoke). Type-head crece +3 (`army_stop`/`army_set_stance`/`army_guard`) → partial load. Caveat: engine Guard rango limitado; SFT corto recomendable sobre guard/focus antes de prod.

### P3 — Producción RA completa + mapas agua/aire + spawn variety — DONE

- [x] Inventario + wire de mapas water/mixed seleccionables sin romper default a_short.
  - rl/map_catalog.py: display_name (titulo original), has_water vs naval_viable.
  - a_short: has_water=True (lagos) pero naval_viable=False (navy gated).
  - Preferir paths oficiales OpenRA/mods/ra/maps/<name>.oramap (MAIN o submodule);
    thin copies en rl/scenarios/ solo como fallback (submodule vacio en este worktree).
  - Basenames oficiales: doughnut.oramap, bombardment-islands.oramap,
    tournament-island.oramap, x-lake.oramap, archipelago.oramap (no stock_*).
  - Junk archivado en rl/scenarios/_archive/ (probe/amin160/a_minus + old stock_*).
  - Pools: land | mixed | official_2p_small | water (water = naval-viable mixed,
    NO "pure water").
  - train --map-pool official_2p_small (o mixed/water/comma keys).
  - auto_train sigue con --scenario a_short (sin cambio de default).
- [x] Mascaras de produccion: forbid naval BUILD/ship TRAIN si no naval_viable;
  unmask en mapas navy-viable. Airbase/heli legales en land; defense ya sale del gate eco.
- [x] Teacher: _optional_naval_air — syrd/spen + TRAIN ship/heli ligero si mapa
  navy-viable + cash. Land/a_short: no-op.
- [x] Teacher: _optional_tech_defense — arbol land-first **después** de eco/barracks
  (roles ya existentes): cheap defense (pbox/hbox/ftur) → powr cushion → dome →
  weap → fix → atek/stek → adv defense. **No** mete weap/dome en BUILD_PRIORITY
  (sigue powr → proc → barracks); gate = phase attack o 
_combat >= RUSH_ATTACK_MOVE
  para no romper el rush Allies en a_short.
- [x] Spawn/layout variety hook: --map-pool (>=2 mapas).
- [ ] **P3+ optional / leftover:** multi-spawn rotation del engine (no hookeado).
- [x] Curriculum onboard / auto_train map-pool flag.
  - --onboard-map-pool official_2p_small (o mixed / water / comma keys).
  - Alias --map-pool (con o sin --onboard): inyecta en TRAIN_ARGS.
  - Default unset = --scenario a_short (C resume / MAIN land intacto).
  - Persistido en curriculum.json como map_pool; resume sin key = land.
- [x] ase2_a.oramap archivado en 
l/scenarios/_archive/; activo solo
  ase2_a_short. Catalog key  soft-alias → _short.

**Out of scope / P3+ skip (no bloquean cerrar P3):**
- Mapas aire **dedicados** / prebuilts fase2 water con base preconstruida
  (scenario authoring). Air sigue disponible vía **hpad** en mapas mixed /
  navy-viable cuando el teacher/agent construye dome→hpad.
- Tapes BC navy/air dedicadas (grabar aparte cuando haga falta SFT).
- Multi-spawn rotation del engine (opcional; pool de mapas ya da variety).

**Estado (2026-09-09):** P3 DONE en exp/ra-completo-p0.
Tests: 	est_ra_completo_p3 (catalog/masks/onboard + teacher naval + tech/defense).
Pool official_2p_small: a_short, Doughnut, Bombardment Islands,
Tournament Island, X-Lake.
Listo para **P4** (obs/dash naval+air).

Como correr:
`
# train directo
python -m rl.train --map-pool official_2p_small ...
# auto_train / onboard (default sigue a_short si no pasas pool)
.venv/Scripts/python.exe rl/auto_train.py --map-pool official_2p_small
.venv/Scripts/python.exe rl/auto_train.py --scratch --onboard --onboard-map-pool official_2p_small
`


### P4 — Obs / dash para naval y air

- [ ] Canales / escalares / roles en obs para naval y air.
- [ ] Dashboard: mix de acciones y contadores que no escondan navy/air bajo “vehicle”.

---

## Notas de implementación (recordatorio)

1. Crecer `ENABLED_TYPES` / type-head → partial load + SFT corto de los tipos nuevos.
2. Tapes land OK para el tronco; **grabar tapes navy/air** aparte.
3. Seguir la convención del repo: **un cambio de régimen por vez** (vocab vs reward vs arch).
4. El train land actual **puede seguir**; este doc no es señal de stop ni de scratch.

