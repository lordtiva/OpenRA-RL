# RL — Documentación de la competencia

Esta carpeta es la **fuente de verdad** de la competencia RL (PPO AlphaStar-lite → ONNX → Flutter). Todo lo que estaba disperso en `docs/*.md` se consolidó acá. `docs/` en la raíz queda solo con `banner.png` y `architecture.png` del proyecto original.

## Índice

| # | Documento | Qué cubre | Estado |
|---|-----------|-----------|--------|
| 00 | `roadmap-agente.md` | Roadmap vivo por fases (v3.1 → v4-macro → Fase 5 ONNX) | Actualizado 2026-08-27 |
| 01 | `diseno-advance-macro.md` | Diseño `advance()` con interrupciones (52 decisiones / ~10s por episodio) | Implementado y activo |
| 02 | `auditoria-pipeline-2026-08-24.md` | Auditoría F1-F10 del pipeline (bugs A1-B6 verficados contra fuente) | Cerrada, tests verdes |
| 03 | `revision-externa-rl.md` | Revisión externa 1 (previa a la auditoría) | Cerrada |
| 04 | `era-economica.md` | Era económica: medición `earned` vs pendiente OLS, cierre en iter 1969 | Cerrada (FALLO documentado) |
| 05 | `fix-endgame-multisesion.md` | Fix `MissionObjectives → EndGame` directo (winrate 0 por `RunAfterDelay`) | Aplicado |
| 06 | `fase2-curriculum.md` | Curriculum militar Escenario A (base pre-construida→juego completo) + escalera A→D | Diseño verificado |
| 07 | `sonda-horizonte.md` | Sonda: ¿cuánto tarda un `win/lose` declarado en Escenario A? + `horizonte 624` | Medido |
| 08 | `parche-grande-2026-08.md` | Parche grande 2026-08: fin de partida + `RlGlobalSummary.earned` + CoordConv/U-Net + BPTT + `eradicate_v3` | Aplicado 2026-08-26 |
| **09** | `06-filosofia-rl.md` | **Filosofía de traducción bot→RL: 4 pilares + crítica a sobre-ingeniería de reward** | **Nuevo — consolida análisis `ai.yaml` + teoría RL** |
| **10** | `07-operacion.md` | **Comandos PowerShell (contenedor / train / dashboard), reglas de run limpio, criterios de promoción** | **Nuevo** |

## Cómo leer esto

1. **Si entrás por primera vez:** `roadmap-agente.md` → `parche-grande-2026-08.md` §1-5 → `06-filosofia-rl.md`.
2. **Si vas a tocar reward:** `parche-grande-2026-08.md` §5 + `06-filosofia-rl.md` §2-3 (riesgos de `w_mass`/`w_tech`) + `reward_shaping.py` como fuente.
3. **Si vas a tocar arquitectura:** `diseno-advance-macro.md` + `parche-grande-2026-08.md` §3 (CoordConv/U-Net/broadcast GRU).
4. **Si vas a lanzar un run:** `07-operacion.md` (3 comandos).

## Mapa de código ↔ doc

| Doc | Código fuente | Proto / Config |
|-----|---------------|----------------|
| `advance()` macro | `openra_env/server/bridge_client.py`, `openra_env/server/openra_environment.py`, `openra_env/generated/rl_bridge*` | `proto/rl_bridge.proto` `GlobalSummary` |
| Auditoría F1-F10 | `rl/rollout.py`, `rl/network.py`, `rl/trainer.py`, `rl/economy_race.py` | `rl/tools/verify_offline.py` (30 checks) |
| Reward `eradicate_v3` | `rl/reward_shaping.py` (`w_refinery_early`, `w_first_ore`, `w_garrison`) | `rl/obs_encoding.py` `SCALAR_DIM=19` |
| Red | `rl/network.py` `AlphaLiteNet` (CoordConv 11ch, U-Net, GRU 416, 4 cabezas) | `rl/obs_encoding.py` `SPATIAL_CHANNELS=9` |
| Bot rival | `openra_env/server/openra_process.py` `BOT_TYPE_MAP` | `OpenRA/mods/ra/rules/ai.yaml` |

## Convenciones

- **Un cambio de régimen por vez** (incluye estado latente: no cambiar reward+red+vocab en el mismo resume).
- **SCALAR_DIM 19** (desde 2026-08-27): `has_refinery`, `can_afford_proc`, `garrison_ratio`. Ckpts con 16 son incompatibles — run limpio obligatorio.
- **Métrica norte:** `winrate` vs `beginner` en Escenario A. Reward medio y `P(win)` son diagnóstico.
- **Escala:** antes de tocar red, verificar que el cuello no sea señal/entorno (lección de la era económica).
