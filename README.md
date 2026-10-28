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
| `tests/` | Unit tests; `-m daemon` corre contra un daemon real |
| `Dockerfile`, `docker-compose.yaml` | Imagen solo C#; 2 daemons en `:9999` y `:10000` |

## Arranque

```powershell
git submodule update --init OpenRA
docker compose up -d --build
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest              # unit
.\.venv\Scripts\python.exe -m pytest -m daemon    # contra el daemon
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
