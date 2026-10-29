# OpenRA-RL v3

Agente de aprendizaje por refuerzo para Red Alert (OpenRA, mod `ra`, Aliados).
Fork de [yxc20089/OpenRA-RL](https://github.com/yxc20089/OpenRA-RL); el motor
modificado vive en el submódulo `OpenRA/` ([lordtiva/OpenRA](https://github.com/lordtiva/OpenRA)).

v2 (AlphaLiteNet) está congelado en el tag `v2-final`. Plan, diagnóstico y
criterios de v3: [`docs/v3-plan.md`](docs/v3-plan.md).

## Estructura

| Ruta | Qué |
|---|---|
| `OpenRA/` | Motor C# con el bridge gRPC multi-sesión (`RLSessionManager`, `ExternalBotBridge`) |
| `proto/rl_bridge.proto` | Contrato agente ↔ motor (copia idéntica a la del submódulo) |
| `ra_env/` | Cliente Python lock-step: `Bridge`, `GameSession`, benchmark, determinismo |
| `ra_rl/` | Políticas del commander (`builtin`, `scripted`, `random`, `net`), features, red, imitación (`record`, `bc`) y evaluación estándar (`eval`) |
| `tests/` | Unit tests; `-m daemon` corre contra un daemon real |
| `Dockerfile`, `docker-compose.yaml` | Imagen solo C#; 6 daemons en `:9999`–`:10004` |

## Arranque

```powershell
git submodule update --init OpenRA
docker compose up -d --build
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# red y entrenamiento (CUDA): torch desde el índice de PyTorch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pytest              # unit
.\.venv\Scripts\python.exe -m pytest -m daemon    # contra el daemon
```

## El agente

El slot del agente es un **commander**: corre los módulos del bot `normal` de
OpenRA y solo toma decisiones estratégicas (qué construir, qué entrenar, a dónde
va el ejército). Contrato: [`docs/v3-commander.md`](docs/v3-commander.md).

```powershell
# evaluación estándar: 100 partidas, semillas fijas, una partida por daemon
.\.venv\Scripts\python.exe -m ra_rl.eval --policy scripted --bot hard --games 100
```

Imitación (Fase 2): grabar decisiones de un maestro, entrenar, evaluar la red.

```powershell
.\.venv\Scripts\python.exe -m ra_rl.record --teacher builtin --games 1800 --out data/builtin
.\.venv\Scripts\python.exe -m ra_rl.bc --data data/builtin --out runs/bc_builtin
.\.venv\Scripts\python.exe -m ra_rl.eval --policy net:runs/bc_builtin/model.pt --bot easy
```

Refuerzo (Fase 3): PPO desde el checkpoint de imitación.

```powershell
.\.venv\Scripts\python.exe -m ra_rl.ppo --init runs/phase2/bc_s2/model.pt --out runs/phase3/ppo_v1 --games-per-iter 48 --iterations 100
```

## Herramientas

```powershell
# partidas/h para varios intervalos de decisión
.\.venv\Scripts\python.exe -m ra_env.bench --daemons localhost:9999,localhost:10000 --sessions 4 --decision-ticks 25,100,250 --games 16

# misma semilla -> misma partida
.\.venv\Scripts\python.exe -m ra_env.determinism --seed 123 --ticks 12000
```

Si cambiás `proto/rl_bridge.proto`: copiarlo también a
`OpenRA/OpenRA.Mods.Common/Traits/Player/Protos/`, correr
`python scripts/gen_proto.py` y reconstruir la imagen.

## Licencia

GPL-3.0 (ver `LICENSE`).
