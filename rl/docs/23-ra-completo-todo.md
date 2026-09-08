# 23 — RA completo: gaps vs el train land actual

> **Para quién:** cuando quieras ampliar el action set / mapas más allá del land-only
> de Phase A/B (naval, air, buildings UI, micro).
>
> **Fecha:** 2026-09-08.
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
| \harvesters_move\ | **Todas** las harvs MOVE a la **misma** celda (macro de grupo). |
| Una sola harv + celda | Via **unit head**: \move\ / \ttack_move\ / \ttack\ sobre un harv emite **MOVE** a \(cx,cy)\ para **ese** \ctor_id\ (ya no se reescribe a \harvest\ sin celda). |
| \harvest\ | Usa el harvester **seleccionado** si es válido; pasa \	arget_x/y\ (CommandModel/HARVEST lo acepta). Si la cabeza no apunta a un harv, fallback \_any_harvester\. |

P0 (rama \exp/ra-completo-p0\): path singular harv→celda **existe** (unit head + \harvest\+cell). El macro de grupo sigue siendo all→same cell.

## TODO priorizado (P0 → P4)

### P0 — Naval / air macros + bucket vehículo; harvest por harv (opcional)

- [ ] Macros de grupo **naval** y **air** (análogos a `army_*` / `infantry_*` / `vehicle_*`).
- [ ] Separar el bucket **vehicle** (hoy mezcla tierra; no alcanza para navy/air).
- [ ] (Opcional) `harvest` / move con **celda por harvester** (no solo `harvesters_move` all→same cell).

### P1 — Building slot head (sell / repair / rally / power_down / set_primary)

- [ ] Cabeza o path de **building slot**: `sell`, `repair`, `rally`, `power_down`, `set_primary`.
- [ ] No mezclar con el mismo corte que crece type-head + maps (un régimen por vez).

### P2 — Micro: group stance / stop, focus fire

- [ ] Stance / stop a nivel **grupo** (no solo unidad suelta).
- [ ] Focus fire (elegir target actor con intención de foco, no solo attack-move a celda).

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

