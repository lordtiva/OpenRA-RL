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
| **09** | `08-avance-run2.md` | **Avance Run2: salto Run1→Run2 (U-Net/BPTT/máscaras), tabla comparativa, diagnóstico coloso pacifista** | **Congela Run2 (2026-08-27)** |
| **10** | `09-fullstack-run3.md` | **Full-stack Run3: SCALAR 21 + auto_support + eradicate_v4 (w_raze 2.0, w_timeout 1.0) — coloso con remate** | **Run3 listo para lanzar (2026-08-27)** |
| **11** | `10-benchmark-alphastar-openai-five.md` | **Benchmark: dónde estás vs AlphaStar/OpenAI Five — qué ya tenés y las 3 brechas (self-play, imitation, annealing)** | **Posicionamiento — mapa de familia** |
| **12** | `11-revision-quisquillosa.md` | **Contra-benchmark quisquilloso: dónde infla la tabla, 6 brechas no contadas y veredicto recalibrado (chasis vs motor)** | **Termómetro de nivel** |
| **13** | `12-plan-4-capas-siguiente-nivel.md` | **Plan 4 capas para tu hardware (2070+5600X): Capa 0 win posible + Capa 1 BC/SIL + Capa 2 transformer/scatter + Capa 3 self-play + throughput** | **Roadmap 6-12 meses (orden por ROI) — no se modifica** |
| **13b** | `13-capa0-status-post-run8.md` | **2c-B + war nudge v2 (peel local). Resume 1046. Run 29 yank.** | **Companion del 12 — 2026-09-01** |
| **13c** | `14-capa2c-identidad-matchup.md` | **Capa 2c: A+B shipped. C smoke falló, revertido. Nudge sin beacon. No 128/256.** | **Spec — 2026-09-01** |
| **13d** | `15-facciones-mods-roles.md` | **No reentrenar cada país. Un train RA Aliados; soviet=mismo ckpt; otro mod=otro ckpt** | **Contrato — 2026-08-31** |
| **14** | `06-filosofia-rl.md` | **Filosofía de traducción bot→RL: 4 pilares + crítica a sobre-ingeniería de reward** | **Consolida análisis `ai.yaml` + teoría RL** |
| **15** | `07-operacion.md` | **Comandos PowerShell (contenedor / train / dashboard / skirmish vs PPO), reglas de run limpio, criterios de promoción** | **Nuevo** |

## Cómo leer esto

1. **Si entrás por primera vez:** `roadmap-agente.md` → `parche-grande-2026-08.md` §1-5 → `08-avance-run2.md` → `09-fullstack-run3.md` → `10-benchmark-alphastar-openai-five.md` + `11-revision-quisquillosa.md` + `12-plan-4-capas-siguiente-nivel.md` (mapa + termómetro + plan) + `13-capa0-status-post-run8.md` (qué de Capa 0 ya está) + `14-capa2c-identidad-matchup.md` (deuda de identidad / matchup).
2. **Si vas a tocar reward:** `08-avance-run2.md` §3 + `09-fullstack-run3.md` §1-2 + `11-revision-quisquillosa.md` §4-5 + `12-plan-4-capas-siguiente-nivel.md` Capa 0 (riesgo garrison/annealing) + `06-filosofia-rl.md` §2-3 + `reward_shaping.py` como fuente.
3. **Si vas a tocar arquitectura:** `diseno-advance-macro.md` + `parche-grande-2026-08.md` §3 (CoordConv/U-Net/broadcast GRU) + `13-capa0-status-post-run8.md` *Deuda de la cabeza de celda* y *Qué robar de Qwen3.8-Flash-Next* (Capa 2: pointer + QSA al mapa; no GDN/MoE el mismo PR). **Capa 2c (slots / rol / enemigos / attack-actor):** `14-capa2c-identidad-matchup.md` — tres PRs Net2Net, no 128/256, no 2b el mismo corte.
4. **Si vas a lanzar un run:** `07-operacion.md` (3 comandos).
6. **Si pensás “¿un train por facción?”:** `15-facciones-mods-roles.md` (no: un ckpt `ra` Aliados cubre los 3 países; soviet=mismo ckpt; `cnc`/`d2k`=ckpt nuevo).
5. **Si vas a tocar el action set** (`sell`/`rally`/`guard`/`APC`/`stance`): `13-capa0-status-post-run8.md` *Órdenes vs scripted* — Capa 0 = support, Capa 2 = pointer, nunca `surrender`. No meter tipos nuevos en `ENABLED_TYPES` el mismo corte que la red.

## Mapa de código ↔ doc

| Doc | Código fuente | Proto / Config |
|-----|---------------|----------------|
| `advance()` macro | `openra_env/server/bridge_client.py`, `openra_env/server/openra_environment.py`, `openra_env/generated/rl_bridge*` | `proto/rl_bridge.proto` `GlobalSummary` |
| Auditoría F1-F10 | `rl/rollout.py`, `rl/network.py`, `rl/trainer.py`, `rl/economy_race.py` | `rl/tools/verify_offline.py` (30 checks) |
| Reward `eradicate_v4` | `rl/reward_shaping.py` (`w_raze 2.0`, `w_timeout 1.0`) + `rl/auto_support.py` | `rl/obs_encoding.py` `SCALAR_DIM=21` |
| Red | `rl/network.py` `AlphaLiteNet` (CoordConv 11ch, U-Net, GRU 416, 4 cabezas, Capa 2 xf/scatter) | `rl/obs_encoding.py` `SPATIAL_CHANNELS=9` `MAX_UNITS=96` combat-first |
| Bot rival | `openra_env/server/openra_process.py` `BOT_TYPE_MAP` | `OpenRA/mods/ra/rules/ai.yaml` |

## Convenciones

- **Un cambio de régimen por vez** (incluye estado latente: no cambiar reward+red+vocab en el mismo resume).
- **SCALAR_DIM 21** (desde Run3): `has_refinery`, `can_afford_proc`, `garrison_ratio`, `military_ratio`, `tech_tier`. Ckpts con 19/16 son incompatibles — run limpio obligatorio.
- **Métrica norte:** `winrate` vs `beginner` en Escenario A. Reward medio y `P(win)` son diagnóstico.
- **Escala:** antes de tocar red, verificar que el cuello no sea señal/entorno (lección de la era económica).
