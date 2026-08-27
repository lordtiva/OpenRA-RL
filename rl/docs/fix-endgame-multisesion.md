# Fix end-game multi-sesión: por qué NINGUNA partida declaraba win/lose

**Fecha**: 2026-08-25 · **Descubrimiento**: test de IA-vs-IA propuesto por el
operador (dificultad actual vs siguiente nivel, 10 partidas scripteadas).

## Síntoma

En toda la historia del proyecto (748+ iters, >4700 episodios) el engine jamás
declaró `win`/`lose`: todas las partidas morían `incomplete@<cap>`. Ni
siquiera SURRENDER declaraba (aceptado por el server, juego seguía con
`done=False`). El bonus win+8/lose−4 del régimen 2-B era código muerto en la
práctica. Cero apariciones de "Game over detected" en rl-bridge.log histórico.

## Cadena causal (3 eslabones)

1. **Condición de derrota impráctica sin ShortGame**: `HasNoRequiredUnits`
   exige destruir TODOS los `MustBeDestroyed`. En el mod ra, vehículos e
   infantería también lo llevan → había que matar hasta la última cosechadora
   de un bot que las re-produce infinitamente. Medido: easy destruyó 6/7
   edificios propios en 78k ticks y aun así no alcanzó.
2. **Guard IsCurrentWorld**: `MissionObjectives.CheckIfGameIsOver` llama a
   `World.EndGame()` vía `Game.RunAfterDelay(GameOverDelay, ...)` con guard
   `Game.IsCurrentWorld(player.World)`. En multi-sesión hay N mundos vivos y
   `CreateSession` pisa `Game.OrderManager` continuamente → el guard descartaba
   el EndGame de casi todas las sesiones casi siempre.
3. **Reloj congelado**: `Game.RunAfterDelay` depende del tick del loop
   principal del juego. En modo multi-sesión no hay loop principal (las
   sesiones se tickean manualmente vía `RLSessionManager.TickSession`) →
   los callbacks quedaban encolados para siempre. Por eso ni un parche sobre
   el guard alcanzaba: había que eliminar el delay.

## Fix (MissionObjectives.cs)

```csharp
if (ExternalBotBridge.MultiSessionMode)
{
    player.World.EndGame();   // directo, sin delay ni guards
    return;
}
Game.RunAfterDelay(Info.GameOverDelay, () => { ... });  // single-session intacto
```

Multi-sesión es headless: el delay de notificación no tiene sentido ahí.

## Verificación post-fix

| Rival | Mapa | Resultado | Ticks | Decisiones |
|---|---|---|---|---|
| easy | fase2_probe_short (1 edificio propio + ShortGame) | **`'lose'`** | 30.226 | ~189 |
| medium | idem | **`'lose'`** | 12.705 | ~80 |

rl-bridge.log registra "Game over detected" ×2. La cadena completa funciona:
WinState → EndGame → puente serializa done/result → cliente lo recibe.

## Escenario activo: fase2_a_short.oramap

Escenario A completo (misma geometría, mismas unidades) + regla ShortGame
por mapa (`Rules: World: MapOptions: ShortGameCheckboxEnabled: True`,
`Locked: True`). Con la marca `RequiredForShortGame: true` que ya traen
`^Building` y MCV en el mod, perder todos los edificios+MCV declara —
condición jugable y alcanzable (easy demuestra que puede aplastar una base
en ~80–190 decisiones). Trainer relanzado con `--scenario a_short`, resume
desde iter 748, watchdog actualizado al mismo flag antes de reactivar.

## Lecciones

- El daemon cachea mapas **por nombre** en memoria: contenido nuevo bajo un
  nombre existente se ignora silenciosamente (y pisa el archivo compartido).
  Todo mapa nuevo necesita nombre propio; sondas derivan `map_name` del file.
- Los builds Docker usan `Dockerfile.local`, que compila el SUBMÓDULO local
  (no confundir con el Dockerfile original que clona GitHub bleed).
- Las sondas deterministas baratas (25–45s c/u) encontraron lo que cientos de
  horas de entrenamiento no podían revelar: medir antes de entrenar funciona.
