# 23 — RA completo: gaps vs el train land actual

> **Para quién:** cuando quieras ampliar el action set / mapas más allá del land-only
> de Phase A/B (naval, air, buildings UI, micro).
>
> **Fecha:** 2026-09-08 (P0 cerrado 2026-09-09 en `exp/ra-completo-p0` @ fe02a7c).
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

### P1 — Building slot head (sell / repair / rally / power_down / set_primary) — USABLE (reusa unit head)

- [x] Path de **building slot**: `sell`, `repair`, `set_rally_point`, `power_down`, `set_primary` en `ENABLED_TYPES`; `ActionIndex.building_ids` + máscara de tipo si no hay edificios; `index_to_command` emite `CommandModel` correcto (`actor_id`; rally + `target_x/y`).
- [x] Sampling/act/eval: la unit head usa **`building_valid`** cuando el tipo ∈ `TYPES_USE_BUILDING` (batch en `_batch_of`, `_sample_ar` / `evaluate_actions` / BPTT). Así el slot no colapsa al own-mask de unidades.
- [x] `command_to_indices` mapea `actor_id` → slot de `building_ids` (SIL/teacher).
- [ ] Cabeza neural **dedicada** a edificios (hoy reusa unit feats/scorer sobre el mismo slot index).
- [x] No mezclar con el mismo corte que crece type-head + maps — P1 reusa filas ya existentes en `ACTION_TYPES` (sin crecer type-head).

**Estado:** usable en `exp/ra-completo-p0` (sin cabeza dedicada). Tests: `test_ra_completo_p1`. Land macros intactos.

### P2 — Micro: group stance / stop, focus fire (+ guard)

- [ ] Stance / stop a nivel **grupo** (no solo unidad suelta).
- [ ] Focus fire (elegir target actor con intención de foco, no solo attack-move a celda).
- [ ] Habilitar RL **`guard`** (OpenRA Guard-on-actor = escoltar harvesters/MCV). Ya está en ActionType / proto / MCP / `guard_target`; falta en `ENABLED_TYPES` + `index_to_command`. Escolta clásica C&C; limitaciones del engine (rango). (P0.5/P2 micro.)

### P3 — Producción RA completa + mapas agua/aire + spawn variety

- [ ] Usar el **árbol de producción RA** de punta a punta (no solo el subset land del teacher).
- [ ] Mapas con **agua / aire** en el curriculum.
- [ ] Más variedad de **spawns** / layouts (evitar overfitting a un solo a_short).

### P4 — Obs / dash para naval y air

- [ ] Canales / escalares / roles en obs para naval y air.
- [ ] Dashboard: mix de acciones y contadores que no escondan navy/air bajo “vehicle”.

---

## Notas de implementación (recordatorio)

1. Crecer `ENABLED_TYPES` / type-head → partial load + SFT corto de los tipos nuevos.
2. Tapes land OK para el tronco; **grabar tapes navy/air** aparte.
3. Seguir la convención del repo: **un cambio de régimen por vez** (vocab vs reward vs arch).
4. El train land actual **puede seguir**; este doc no es señal de stop ni de scratch.

