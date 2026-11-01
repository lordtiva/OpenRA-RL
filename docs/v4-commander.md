# Commander v4 — contrato del entorno y self-play lock-step

En `v4`, tanto el slot primario (`Multi1`) como el slot par (`Multi0`) pueden correr el
cerebro `rl-agent` simultáneamente en **lock-step de una sola llamada RPC**
(`FastAdvance`). Además, el comandante táctico evoluciona de un único blob terrestre
a un sistema **multi-escuadrón + aviación + poderes de soporte + facciones bilaterales**.

---

## 1. Lock-step dual sin condiciones de carrera (`PostWorldTick`)

En `MultiSessionMode` (`RLSessionManager.TickSession`), `world.Tick()` ejecuta primero
`ITick.Tick()`, `commander.Tick()` e `IssueOrder()` para **todos** los actores jugador
(`Multi1` y `Multi0`). Recién al finalizar el tick completo del mundo se invoca
`ExternalBotBridge.PostWorldTick()`, que:

1. Comprueba si se alcanzó el límite de ticks, si terminó la partida (`world.IsGameOver`)
   o si **cualquiera** de los dos comandantes (`Multi1` o `Multi0`) emitió nuevas
   solicitudes pendientes (`HasNewCommanderRequests`).
2. Serializa `obs` (`Multi1`) y adjunta `obs.peer_observation` (`Multi0`), cada una
   respetando estrictamente la niebla de guerra (`Shroud`) de su propio jugador y con
   `global_summary` relativo a su bando (`own` vs `enemy`, suma cero exacta).
3. Devuelve ambas observaciones en un único `FastAdvanceResponse`, mientras que las
   respuestas del cliente llegan en un único `FastAdvanceRequest` (`decisions` para
   `Multi1` y `peer_decisions` para `Multi0`).

En Python, `ra_env.SelfPlaySession` expone este ciclo como `step(p1_decisions, p0_decisions, ticks) -> DualStepResult`.

---

## 2. Facciones bilaterales (`Allies` + `Soviet`)

`RLSessionManager.ResolvePlayerFaction` respeta `player_faction="Random"` (o cualquier
facción explícita de las 5 de multijugador: `england`, `france`, `germany`, `russia`,
`ukraine`), conservando `RandomAllies` únicamente cuando `player_faction` llega vacío
por compatibilidad con `v3`. Cada `GameObservation` informa `player_slot` y `player_faction`.

---

## 3. Solicitudes de decisión (`DecisionRequest`)

| `kind` | `queue` | Cuándo se emite | Respuesta (`Decision`) |
|---|---|---|---|
| `DECIDE_BUILD` | `Building`, `Defense` | Cola libre (cadencia `BaseBuilder`) | `choice` en `options`, o `""` = esperar 100 ticks |
| `DECIDE_TRAIN` | `Infantry`, `Vehicle`, `Aircraft`, `Ship` | Cola libre (~30 ticks) | `choice` en `options`, o `""` = esperar 100 ticks |
| `DECIDE_ARMY` | `MAIN` | Cada 250 ticks | `target_index = k` con `choice = "attack"` (micro normal) o `choice = "all_in"` (`Squad.Fearless = true`, sin retirada difusa); `target_index = -1` (`hold`); sin respuesta = mantener orden |
| `DECIDE_ARMY` | `HARASS` | Cada 250 ticks cuando hay raiders rápidos (`jeep`, `ftrk`, `1tnk`, `apc`, hasta 4 unidades reportadas en `obs.harass_unit_ids`) | `target_index = k` (`attack`/scout); `target_index = -1` (`hold`: reintegrar raiders al ejército principal); sin respuesta = mantener orden |
| `DECIDE_ARMY` | `AIR` | Cada 250 ticks cuando hay aeronaves de combate (`heli`, `mh60`, `hind`, `yak`, `mig`) | `target_index = k` (ataque aéreo / reconocimiento); `target_index = -1` (`hold`); sin respuesta = mantener orden |
| `DECIDE_SUPPORT` | `OrderName` (`SovietSpyPlane`, `SovietParatroopers`, `UkraineParabombs`, `Chronoshift`, `AdvancedChronoshift`, `NukePowerInfoOrder`, `GrantExternalConditionPowerInfoOrder`) | Cada 25 ticks cuando un poder cargado está listo (`sp.Ready`, omitiendo `GpsPower` automático) | `target_index = k` (`attack`: disparar sobre `targets[k]`); `target_index = -1` (`hold`: posponer 150 ticks) |

---

## 4. Candidatos tácticos honestos a la niebla (`MaxTargets = 16`)

Todos los candidatos en `req.targets` (hasta 16) provienen exclusivamente del conocimiento
legítimo del jugador:

1. `enemy_spawn`: ubicaciones `mpspawn` leídas de `world.Map.ActorDefinitions` (agnóstico
   al número de spawns, 2–8 jugadores), excluyendo el propio spawn y descartando spawns ya
   explorados sin estructuras enemigas cercanas.
2. `enemy_building`: estructuras enemigas vistas y aún no confirmadas destruidas en la
   niebla (hasta 8, ordenadas por cercanía a la base propia).
3. `enemy_harvester`: cosechadoras enemigas vistas en los últimos 300 ticks (deduplicadas
   en radio de 3 celdas, hasta 2).
4. `enemy_mobile`: punta de lanza móvil enemiga vista en los últimos 300 ticks (hasta 2).
5. `unexplored` / `stale`: sectores terrestres alcanzables de 16×16 celdas aún sin explorar
   (`< 50%`) o sin visión reciente (`> 2000` ticks), permitiendo cazar expansiones ocultas.

---

## 5. Reglas de sanidad de producción y expansión

- **Regla estricta de rearme aéreo**: en Red Alert cada avión (`yak`, `mig`) requiere 1
  `afld` dedicado y cada helicóptero (`heli`, `mh60`, `hind`) requiere 1 `hpad` dedicado.
  `Features.item_mask` y `ScriptedPolicy` bloquean entrenar aeronaves por encima del número
  de plataformas propias construidas (`<= 4`).
- **Saturación de base y anillo extendido**: `BaseBuilderQueueManager` intenta primero el
  radio estándar (`MaxBaseRadius`) y cae a un anillo extendido (`MaxBaseRadius * 2`,
  respetando adyacencia válida `IsCloseEnoughToBase`) cuando la base principal se llena,
  reportando `obs.placement_failures` para habilitar expansiones con un segundo `mcv`.
- **Cero asignaciones LOH innecesarias**: `SerializeSpatialMap` queda desactivado por
  defecto en modo comandante (`OPENRA_RL_SPATIAL_MAP=0`), ahorrando ~1.77 MB por paso y
  superando los ~6.200 ticks/s con 6 daemons.
