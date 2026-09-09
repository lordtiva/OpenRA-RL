# Fase 2 — Curriculum militar (base pre-construida)

**Estado:** diseño verificado a nivel código, pendiente de implementación.
**Disparador que la activó:** el enjambre de infantería converge a paridad
militar (cbt ≈ −0.02) pero 0 wins en 9780 episodios; con horizonte doble
(16952 ticks) SANGRA (cbt −5): el AI beginner la supera en guerras largas.
Falta el gradiente hacia progresión tech y uso agresivo del ejército.

## Mecanismo del server (VERIFICADO en openra_environment.py:3057-3091)

`reset()` acepta kwargs que el cliente ya pasa (`EnvClient.reset(**kwargs)`):

| Kwarg | Efecto |
|---|---|
| `map_data` | bytes `.oramap` en base64 → se escribe en `mods/<mod>/maps/` y se juega ese mapa |
| `map_name` | nombre del escenario a usar |
| `bot_type` | dificultad del AI **por sesión** (vacío = dummy pasivo, sin AI) |

En modo multi-sesión, si no hay bot_type el enemigo entra como "dummy"
(slot ocupado pero sin AI) — útil para escenarios de tiro al blanco.

## Diseño del escenario

### Plantilla

Partir de `singles.oramap` (el mapa actual) como plantilla: los `.oramap`
son ZIPs con `map.yaml` (actores, waypoints de spawn) + binarios de terreno.
Script propuesto: `rl/make_scenario.py`

1. Descomprimir el `.oramap` original
2. Inyectar actores en el `map.yaml`: edificios con `Owner: Multi1` (nuestro
   slot) posicionados relativos al waypoint de inicio de Multi1
3. Reempaquetar → base64 → `reset(map_data=..., bot_type="beginner")`

### Escalonado propuesto

| Etapa | Base propia inicial | Objetivo de aprendizaje | Criterio para avanzar |
|---|---|---|---|
| A | Conyard + 2 powr + proc + barr + weap + 8 e1 | Usar el ejército: atacar, no solo defender | winrate ≥50% vs beginner |
| B | Sin weap (debe construirla) | El paso tech como camino al poder | wins con weap producida |
| C | Solo conyard + powr | Economía mínima + tech + guerra | wins sostenidos |
| D | Juego completo actual (hoy) | Todo integrado, horizonte 17k ticks | winrate ≥50% |

- La etapa C/D re-introducen la economía que ya saben (no se desperdicia:
  las etapas previas son cortas gracias al throughput).
- `bot_type` escalable por reset: beginner → easy cuando el winrate lo diga.

## Cambios de código necesarios

1. `rl/make_scenario.py` — generador de escenarios (~150 líneas, zipfile+yaml)
2. `rollout.py` — `collect_one_episode(..., scenario=None)`; si hay escenario,
   `env.reset(map_data=scenario.b64, bot_type=scenario.bot)` en vez de
   `env.reset()`
3. `train.py` — flags `--scenario`, `--bot-type`; métrica por etapa
4. OJO multi-sesión: cada conexión del pool pide SU reset con el mismo
   escenario (el server escribe el mapa una vez; idempotente)

## Riesgos

- **Ownership de actores**: verificar en un smoke test que Multi1 recibe los
  edificios pre-colocados (la obs debe listarlos en `buildings`)
- **Waypoints**: los edificios deben caer en celdas válidas del área de spawn;
  usar los waypoints `spawn_multi1` como referencia y validar contra
  `passability` (canal 3 del tensor espacial)
- **El shaper cuenta edificios pre-existentes**: `reset()` ya excluye los tipos
  iniciales del bonus `new_types` (correcto); `buildings` no pagará por los
  pre-construidos (correcto: el gradiente debe venir del USO, no de existir)

## Orden de ejecución

1. Generar escenario A + smoke test visual de obs (¿vemos la base?)
2. Run corto 50 iters etapa A → ¿aparece agresión? (attacks >0, kills >0)
3. Escalar según criterios de la tabla
