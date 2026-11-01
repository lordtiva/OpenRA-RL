# OpenRA-RL v4

Agente de aprendizaje por refuerzo y **self-play híbrido** para Red Alert (OpenRA,
mod `ra`, facciones bilaterales **Aliados + Soviéticos**). Fork de
[yxc20089/OpenRA-RL](https://github.com/yxc20089/OpenRA-RL); el motor modificado
vive en el submódulo `OpenRA/` ([lordtiva/OpenRA](https://github.com/lordtiva/OpenRA)).

- **`v2-final`**: congelado (AlphaLiteNet micro).
- **`v3-final`**: congelado en `main` (`80.0%` vs `hard`, `74.0%` vs `brutal`, `85.0%` vs `turtle`, armónica `0.794`; ver [`docs/v3-plan.md`](docs/v3-plan.md) y [`docs/v3-commander.md`](docs/v3-commander.md)).
- **`v4` (rama activa)**: **Commander v4** multi-escuadrón (`MAIN` con `all_in`, `HARASS`, `AIR`, `DECIDE_SUPPORT`) + **Self-Play Lock-Step Real** (`SelfPlaySession`). Contrato y arquitectura: [`docs/v4-commander.md`](docs/v4-commander.md) y [`docs/v4-plan.md`](docs/v4-plan.md).

## Estructura

| Ruta | Qué |
|---|---|
| `OpenRA/` | Motor C# con bridge gRPC multi-sesión y lock-step dual `PostWorldTick` (`RLSessionManager`, `ExternalBotBridge`, `RLCommander`) |
| `proto/rl_bridge.proto` | Contrato agente ↔ motor (idéntico al del submódulo) |
| `ra_env/` | Cliente Python lock-step: `Bridge`, `GameSession` (vs bot C#) y `SelfPlaySession` (`RL-vs-RL` en un solo RPC) |
| `ra_rl/` | Políticas (`builtin`, `scripted` bilateral v4, `random`, `net`), features (`FEATURE_VERSION = 3`), `CommanderNet` CTDE, imitación (`record`, `bc`), refuerzo SMDP (`ppo`, `league`), **liga híbrida self-play** (`selfplay`) y evaluación (`eval`) |
| `scripts/` | Orquestador v4 (`run_v4_pipeline.py`), pipeline v3 (`run_recovery_pipeline.py`), telemetría (`analyze_ppo.py`) y protobuf (`gen_proto.py`) |
| `tests/` | Suite unitaria (`pytest`) e integración contra daemon real (`pytest -m daemon`, incluyendo `test_selfplay_daemon.py`) |
| `Dockerfile`, `docker-compose.yaml` | Imagen C# headless; 6 daemons en `:9999`–`:10004` |

## Arranque

```powershell
git submodule update --init OpenRA
docker compose up -d --build
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pytest              # unit tests (44)
.\.venv\Scripts\python.exe -m pytest -m daemon    # integración contra daemon (8)
```

## Entrenamiento y Evaluación v4 (Commander v4 + Self-Play Híbrido)

```powershell
# Pipeline completo v4 (Fase 4.1 BC+DAgger bilateral -> Fase 4.2 Warm-Start -> Fase 4.3 Self-Play -> Eval 10000-10099):
.\.venv\Scripts\python.exe scripts/run_v4_pipeline.py --stage all

# Evaluación estándar bilateral contra bots C# (Aliados + Soviéticos):
.\.venv\Scripts\python.exe -m ra_rl.eval --policy scripted --bot hard --player-faction Random --games 100

# Evaluación head-to-head RL-vs-RL (alternando slots Multi1/Multi0 automáticamente):
.\.venv\Scripts\python.exe -m ra_rl.eval --policy net:runs/v4/selfplay/checkpoint_best.pt --opponent-policy net:runs/v4/bc_v4/model.pt --player-faction Random --games 100
```

## Herramientas

```powershell
# Benchmark de throughput de simulación
.\.venv\Scripts\python.exe -m ra_env.bench --daemons localhost:9999,localhost:10000 --sessions 2 --decision-ticks 250 --games 12

# Verificación de determinismo exacto por semilla
.\.venv\Scripts\python.exe -m ra_env.determinism --seed 123 --ticks 12000
```

Si modificas `proto/rl_bridge.proto`: sincronízalo con
`OpenRA/OpenRA.Mods.Common/Traits/Player/Protos/rl_bridge.proto`, ejecuta
`.\.venv\Scripts\python.exe scripts/gen_proto.py` y reconstruye con `docker compose up -d --build`.

## Licencia

GPL-3.0 (ver `LICENSE`).
