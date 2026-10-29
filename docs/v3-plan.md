# v3 — plan y criterios

v2 (AlphaLiteNet, currícula macro-first) quedó congelado en el tag `v2-final`.
Este documento es la fuente de verdad de v3.

## Por qué v2 no escalaba

1. **La política no era la que jugaba.** Una decisión cada 50–80 ticks sobre
   31 tipos × unidad × celda × ítem no alcanza para un RTS. El adapter reescribía
   la acción con heurísticas (remap, stage, guard, peel, pack-12, war_objective,
   auto_support) que cambiaban de run en run: crédito ruidoso y un MDP móvil.
2. **Muestras.** ~10² partidas/h y 4 episodios por iteración: `wr20` tenía ±0.2
   de ruido y guiaba promociones, restores y "sequías".
3. **Parches encadenados.** ~47 runs reanudados (Net2Net, cargas parciales,
   reward v3→v6) sin una línea base desde cero comparable.
4. **Maestro débil.** El BC imitaba a un script Python, no al bot C# del motor.
5. **La red no era el cuello.** Mejor resultado: 88% vs beginner, ~30% vs easy,
   0 vs medium.

## Principios

- La red decide **estrategia** (qué producir, a dónde va cada escuadrón, postura).
  Un ejecutor fijo, dentro del motor, hace los clics y el micro.
- El ejecutor y el contrato del MDP son **parte del entorno**: se versionan y no
  cambian durante un experimento.
- Todo cambio de régimen arranca **desde cero** y se compara con el mismo
  protocolo de evaluación (100 partidas, semillas fijas, intervalo de confianza).
- Juego en **lock-step headless**: el motor solo avanza dentro de `step()`.

## Alcance inicial

1v1, agente Aliados (`RandomAllies`) vs bot `Random`, un mapa.
"Experto" = ≥70% vs `hard` (bot OpenRA `normal`). Mapas extra y Soviet, después.

## Fases

| Fase | Qué | Pasa si |
|---|---|---|
| 0 | Limpieza, cliente gRPC lock-step, benchmark, determinismo | partidas/h medido; misma semilla → misma partida |
| 1 | Acciones macro + ejecutor C# + política a mano | el script gana a `hard` ≥60% |
| 2 | Grabar decisiones de los bots C# → BC | la red clonada ≥50% vs `easy` |
| 3 | PPO desde BC | ≥50% vs `hard` en eval de 100 partidas |
| 4 | Self-play / PFSP | ≥70% vs `hard`/`brutal`, sin colapsar a una estrategia |
| → | PR a `main` | Fase 3 estable |

## Estado

### Fase 0

- Repo reducido a: `OpenRA/` (submódulo), `proto/`, `ra_env/`, `tests/`, Docker.
- `ra_env`: cliente gRPC directo al daemon multi-sesión (sin FastAPI/OpenEnv/MCP).
- Imagen solo C# (`Dockerfile`), 2 daemons en `docker-compose.yaml`.
- Motor: `OPENRA_RL_DETERMINISTIC=1` siembra `World.LocalRandom` con la semilla
  de la partida (los módulos del bot lo usan; antes divergía a los ~300 ticks).
- Herramientas: `python -m ra_env.bench`, `python -m ra_env.determinism`.

Resultados medidos: ver [`v3-phase0-results.md`](v3-phase0-results.md).

### Fase 1 ✅

- Commander en el motor: el `rl-agent` corre los módulos del bot `normal`; el
  agente decide BUILD / TRAIN / ARMY. Contrato: [`v3-commander.md`](v3-commander.md).
- Paridad del ejecutor (`builtin` vs `hard`): 51%. Política a mano vs `hard`: 72%.
- `python -m ra_rl.eval` es el protocolo de evaluación oficial.

Resultados y bugs del motor corregidos: [`v3-phase1-results.md`](v3-phase1-results.md).

### Fase 2 ✅

- `builtin` responde por el mismo camino que un agente: cada decisión del maestro
  viene con el estado exacto en que se tomó.
- Features fijas (`ra_rl/features.py`), red MLP + puntero (`ra_rl/model.py`),
  grabación (`ra_rl.record`), BC (`ra_rl.bc`), DAgger con el `scripted` como
  etiquetador.
- BC(`builtin`) 100% vs `easy`; BC(`scripted`) + 2 rondas de DAgger (`bc_s2`)
  69% vs `hard` (66,5% en semillas nuevas, a la par del maestro).
- Regla nueva: se ajusta y se elige en semillas que no son las oficiales.

Resultados: [`v3-phase2-results.md`](v3-phase2-results.md).
