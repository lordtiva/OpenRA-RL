# Archivo — no es fuente de verdad

Documentos **cerrados, de una era, o de un run**. El código ya no se guía por acá. Índice vivo: [`../README.md`](../README.md).

Se mueven acá, no se borran: sirven para no repetir un régimen que ya falló (era económica, coloso pacifista, 2c-C revertido).

---

## `informes/` — reviews de un snapshot

| Archivo | Qué era | Por qué archivo |
|---------|---------|-----------------|
| `Informe-1.md` | Review general del stack (red, adapter, SIL) | Snapshot; `IDENTITY_ITEMS` / Aliados lock no existían |
| `Informe-2.md` | Diagnóstico coloso + timeouts 53k | Era del mill / pack-12 |
| `Informe-3.md` | Reward-hacking garrison vs timeout | Motivó `eradicate_v4`; el shaper es el código |

## `auditorias/` — F1–F10 y termómetros Ago 2026

| Archivo | Qué era |
|---------|---------|
| `auditoria-pipeline-2026-08-24.md` | Bugs A1–B6 del pipeline. Cerrada, tests verdes |
| `revision-externa-rl.md` | Review previa a esa auditoría |
| `10-benchmark-alphastar-openai-five.md` | Posicionamiento vs AlphaStar / OpenAI Five (Run2/Run3) |
| `11-revision-quisquillosa.md` | Contra-benchmark; chasis vs motor |

## `eras/` — v3.1, era económica, Run2/Run3

| Archivo | Qué era |
|---------|---------|
| `roadmap-agente.md` | Roadmap congelado en iter 1969 / v3.1 (CNN32, wr 0%) |
| `era-economica.md` | Incentivo minero → cosecha 0. **Fallo documentado** |
| `fase2-curriculum.md` | Escalera A→D de diseño. Superseded por `start/onboard.md` |
| `fix-endgame-multisesion.md` | `MissionObjectives → EndGame`. Aplicado |
| `sonda-horizonte.md` | Horizonte 624 decisiones. Medido |
| `parche-grande-2026-08.md` | Fin de partida + CoordConv/U-Net + BPTT + `eradicate_v3`. Aplicado |
| `08-avance-run2.md` | Congela Run2 (coloso pacifista) |
| `09-fullstack-run3.md` | Run3 listo para lanzar (Ago-27). SCALAR 21 de entonces |

## `runs/` — diarios de un corte

| Archivo | Qué era |
|---------|---------|
| `13-capa0-status-post-run8.md` | Diario Runs 8–36 (~49 kB). Companion empírico del plan 4 capas |
| `17-run43-no-war-nudge.md` … `21-run47-map-qsa.md` | Knobs de un resume (43–47). El puente RL-vs-RL vivo está en `design/rl-vs-rl.md` |

---

## Nombres viejos → sitio nuevo

Los números `00`–`24` del índice plano **ya no se usan**. Si un comentario de código o un commit apunta a un filename:

| Antes (plano `rl/docs/`) | Ahora |
|--------------------------|-------|
| `07-operacion.md` | `start/operacion.md` |
| `22-onboard.md` | `start/onboard.md` |
| `24-ra-aliados.md` | `contract/ra-aliados.md` |
| `15-facciones-mods-roles.md` | `contract/facciones-mods-roles.md` |
| `23-ra-completo-todo.md` | `contract/ra-completo.md` |
| `06-filosofia-rl.md` | `design/filosofia-rl.md` |
| `12-plan-4-capas-siguiente-nivel.md` | `design/plan-4-capas.md` |
| `14-capa2c-identidad-matchup.md` | `design/capa2c-identidad-matchup.md` |
| `diseno-advance-macro.md` | `design/advance-macro.md` |
| `16-rl-vs-rl-run42.md` | `design/rl-vs-rl.md` |
| el resto de la tabla 00–24 | este `_archive/` (misma basename) |

*Archivado: 2026-09-09. Rama `alphalite-v2`.*
