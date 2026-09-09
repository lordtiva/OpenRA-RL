# Archived scenarios

Moved out of the active `rl/scenarios/` pool (P3 cleanup):

- `fase2_probe*.oramap`, `fase2_amin160*`, `fase2_a_minus_short` — unused
  curriculum probes; kept for archaeology only.
- `stock_doughnut.oramap`, `stock_bombardment_islands.oramap` — old vendored
  renames. Active catalog prefers official OpenRA basenames
  (`doughnut.oramap`, `bombardment-islands.oramap`) from
  `OpenRA/mods/ra/maps/` (MAIN checkout or thin copies in `rl/scenarios/`).

Do not wire these back into `map_catalog.MAP_CATALOG` without a reason.
