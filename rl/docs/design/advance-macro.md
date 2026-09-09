# Diseño: avance macro con interrupciones (`advance()`)

**Estado:** diseñado, pendiente de implementación al terminar el run v3.1 (iter 250).
**Objetivo:** multiplicar muestras/hora y convertir el frame-skip fijo en decisiones
macro adaptativas, sin tocar el motor ni el reward.

## El problema que resuelve

Hoy cada decisión del agente avanza **2 ticks** (pasos de 2 ticks × frame-skip k=8).
Un episodio de 8000 ticks = ~4000 viajes ida-y-vuelta
(serialización protobuf → WebSocket → Python → forward de red → comando).
El motor ya corre sin pacing en headless (fast-forward por ráfagas de hasta
5000 ticks, `OpenRA/OpenRA.Game/Game.cs` líneas 838-849), así que el cuello
real es la latencia del diálogo agente↔motor, no la simulación. Medido:
~148 s de pared por tanda de 12 episodios donde la simulación pura sería ~13 s
(8000 ticks × 4 conexiones/server ÷ ~2200 ticks/s agregados).

## Qué ya existe en el fork (no hay que escribir C#)

| Pieza | Dónde | Detalle |
|---|---|---|
| Tool `advance(ticks)` | `openra_env/server/openra_environment.py:1427` | Clamp **[1, 50] ticks por llamada** para mantener cada llamada gRPC <2 s |
| Interrupciones server-side | ídem | Chequeo cada **25 ticks**; 9 señales por defecto: `game_over`, `enemy_spotted`, `unit_destroyed`, `under_attack`, `building_discovered`, `enemy_building_destroyed`, `own_building_destroyed`, `unit_arrived`, `production_complete` |
| RPC rápido | `env._bridge.fast_advance_unary(...)` | Unario, saltea el canal DropOldest del streaming |
| Campos de corte | `bridge_client.py:332-334` | `interrupted`, `interrupt_reason`, `actual_ticks_advanced` ya parseados en el obs dict |

## Cambios en nuestro código

### 0. Transporte — RESUELTO (verificado contra el código)

- El protocolo `/ws` de OpenEnv acepta `{"type": "mcp", "data": <json-rpc>}`
  y enruta al entorno DE ESA SESIÓN (`http_server.py:1433-1450`). No hace
  falta reconstruir Docker ni tocar el server.
- Formato JSON-RPC exacto (idéntico al de `openra_env/mcp_ws_client.py`,
  que ya existe como referencia): `{"jsonrpc":"2.0","method":"tools/call",
  "params":{"name":"advance","arguments":{"ticks":N}},"id":k}`.
- **Restricción crítica**: cada conexión `/ws` es SU PROPIA sesión de juego
  (`_create_session()` por conexión). Las llamadas a `advance` deben ir por
  LA MISMA conexión que reset/step del episodio.
- Respuesta de `advance`: tick, done, result, economy, conteos de unidades/
  edificios/enemigos, units_summary (idle/hp/celdas), explored_percent,
  reward_vector (del server), y los flags `interrupted`, `interrupt_reason`,
  `actual_ticks_advanced`. **NO trae** military (kills_cost/deaths_cost/
  assets_value) ni spatial_map.

### 1. Cliente: exponer `advance`

Agregar a `OpenRAEnv` (openra_env/client.py) un método fino que hable
JSON-RPC por su propio socket (mismo patrón `_send_recv` de mcp_ws_client):

```python
async def advance(self, ticks: int) -> dict:
    """Avanza hasta N ticks (clamp server: 50/llamada); devuelve resumen
    con interrupted/interrupt_reason/actual_ticks_advanced."""
```

### 2. Rollout macro (`rollout.py`) — arquitectura del bloque

El shaper necesita contadores militares que `advance` no devuelve ⇒ el
moldeo se hace en los LÍMITES del bloque vía `step()` (los contadores son
ACUMULATIVOS: el delta entre inicio y fin captura todo lo pasado dentro,
incluido combate; única pérdida: edificio construido Y destruido dentro del
mismo bloque, despreciable):

```
decidir (obs completa actual) → comandos
result = step(comandos)            # +2 ticks, obs completa → shaper.step
restante = PRESUPUESTO - 2
mientras restante > 0 y no done:
    adv = advance(min(50, restante))
    restante -= adv.actual_ticks_advanced
    si adv.done o adv.interrupted: cortar
result = step(NO_OP)               # +2 ticks, obs COMPLETA p/ próxima decisión
                                   # (y cierre de deltas del bloque)
si adv.interrupted: decidir YA (micro-emergente); presupuesto completo si no
```

- Presupuesto inicial: 160 ticks/decisión (~6.4 s de juego). Episodio:
  ~4000 → ~52 decisiones (~25× menos viajes incluso contando llamadas internas).
- Interrupts: las 9 por defecto ya activadas server-side (chequeo c/25 ticks).
- Reward: ShapedReward INTACTO; finalize() sigue pagando margen al truncar.
  Acreditación: TODO el bloque pertenece a la muestra de la decisión.
- Telemetría nueva por decisión: interrupt_reason, ticks del bloque.
- Tolerancia engine: try/except alrededor de advance() igual que step()
  (degradar a NO_OP; abortar tras 5 errores consecutivos).
- Flag `--macro-ticks` en train.py/diagnose (0 = comportamiento viejo).

### 3. Hiperparámetros a revisar tras el cambio

- `--max-steps`: pasa a contar *decisiones macro* (50 ≈ episodio completo).
  Renombrar internamente a presupuesto de ticks para no confundirse.
- GAE: horizonte efectivo se achica 80×; `gamma=0.99` probablemente siga OK
  porque el reward por decisión crece proporcionalmente, pero vigilar
  `v_loss` las primeras 10 iters.
- Entropía: menos decisiones = menos exploración por episodio; si H cae,
  subir temperatura de muestreo antes que tocar el bonus.

## Riesgos

1. **Cambio de régimen**: comparar curvas pre/post requiere marcar el corte en
   metrics.jsonl (el auto-rotate de train.py ya lo maneja si arrancamos fresco;
   si reanudamos, insertar marcador manual `{"note": ...}`).
2. **Calidad de decisión por tick menor**: con obs cada 160 ticks, reacciones
   finas (esquivar tanques) empeoran. Aceptado: contra beginner/normal importa
   la economía+masa; revisar winrate antes de culpar al granularidad.
3. **Clamp 50/llamada**: 160 ticks = ≥4 llamadas gRPC por decisión; siguen
   siendo 25× menos viajes que hoy (4000→~150 por episodio contando llamadas).

## Plan de validación (aceptación)

1. Smoke test: 1 episodio macro contra localhost:8000 — reward total del
   episodio comparable (±20%) al régimen viejo con misma política (temp 0.35,
   diagnose). Los componentes deben sumar igual al total.
2. Throughput: ticks/s agregados esperando ≥2× (objetivo 3-5×).
3. Run corto de 30 iters: reward medio no debe derrapar >0.5 respecto al
   nivel pre-cambio; luego run completo.
