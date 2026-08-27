# OpenRA-RL — Roadmap del agente PPO (AlphaStar-lite)

**Última actualización:** cierre de la **era económica** e inicio del plan de
reparación (2026-08-24, iter 1969). La era económica quedó CERRADA sin cumplir
su criterio de salida; una segunda revisión externa verificó bugs en el
pipeline de gradiente y reordenó prioridades. Estado completo:
[`docs/auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md).
Historial de la era cerrada: [`docs/era-economica.md`](era-economica.md).

**Resumen del estado actual:** trainer DETENIDO en iter 1969 (reward negativo,
cosecha propia ≡ 0 tras ~340 iters de incentivo minero, winrate histórico
0/~2000 episodios). Checkpoint de la mejor paridad militar preservado:
`rl/ckpts/PRESERVADO_iter1700_paridad_militar.pt`. El instrumental de medición
(espectador exacto, `earned`, filtrado zombi, watchdog autónomo) sobrevive y
sigue siendo la base de evaluación. Próximo paso: reparar el pipeline (F1-F10)
y cambiar el problema (curriculum militar con acción grupal), no seguir
parchando pesos del shaper.

---

## Estado congelado (v3.1)

| Ítem | Valor |
|---|---|
| Checkpoint | `rl/ckpts/latest.pt` = iter **237** |
| Reward medio | ~**3.2** (récord 3.53), subiendo ~+0.009/iter desde el fix |
| Componentes típicos | cbt −0.35 · ast +0.19 · bld +2.1 · typ +1.35 · mrg −0.11 |
| Entropía / clip_frac | 1.4-1.5 / 0.05-0.11 (saludables) |
| Winrate | 0% — todos los episodios `incomplete@8000t` (cap de pasos) |
| Red | AlphaLiteNet 0.19M params: CNN32+2ResBlocks → GRU128 → 4 cabezas |
| Entrenamiento | 12 episodios/iter, pool WS de 12 conexiones sobre 3 daemons Docker (:8000/:8010/:8020) |

### Qué demostró v3.1 (régimen del margen truncable)

El fix clave fue pagar el margen militar `tanh((kills−deaths)/3000)` también al
TRUNCAR por cap de pasos (`ShapedReward.finalize()`): antes era código muerto
porque ninguna partida llega a `done`. Efecto medido tras activarlo (iter 114→237):

- El techo estructural de ~2.8 quedó atrás (media 3.2, picos 3.5)
- `combat` mejoró ~40% (−0.69 → −0.35): el impuesto militar se está invirtiendo
- Producción militar real emergió: perreras + perros con reposición (= escaramuzas)
- Riesgo conocido y contenido: spam incipiente de muros baratos (19% de lo
  encolado) — palanca futura si satura: `w_building` proporcional al costo

### Historial de regímenes (lecciones caras, no repetir)

| Régimen | Reward | Qué pasó | Lección |
|---|---|---|---|
| v1 (reward server) | 4.04 plano | bonus supervivencia dominaba; nadie ganaba ni perdía | reward starvation: varianza cero = gradiente cero |
| v2 (shaping propio) | techo 2.4 | exploit "solo plantas de energía" (todo edificio pagaba igual) + argmax en ítems | diversidad hay que PAGARLA distinto por tipo; argmax mata exploración intra-episodio |
| v3 (diversidad) | techo 2.8 | base diversa estable pero margen terminal muerto | los términos terminales que nunca disparan son código muerto |
| **v3.1 (margen truncable)** | **3.2 ↑** | combate mejora, sin exploits nuevos graves | pagar al truncar revivió la señal militar completa |

Archivos de datos: `metrics_v31_*.jsonl` (curva v3.1), `metrics_mixed_backup.jsonl`,
`metrics_v1_regimen_viejo.jsonl`, checkpoints `iter*.pt` cada 10 iters.
Dashboard: `python -m http.server 8501` → localhost:8501/dashboard.html.

---

## Fase 1 — Avance macro con interrupciones (v4-macro) ← IMPLEMENTADA Y ACTIVA

**Diseño completo:** `docs/diseno-advance-macro.md`

### Resultados medidos (2026-08-24)

| Métrica | v3.1 (frame-skip) | v4-macro | Factor |
|---|---|---|---|
| Decisiones por episodio | ~4000 | 52 | 77× |
| Tiempo de pared por episodio | ~150 s | ~10 s | **~15×** |
| Iteración completa (12 ep) | ~165 s | ~13 s | **~12.7×** |

- Transporte: canal MCP JSON-RPC sobre la MISMA conexión /ws del episodio
  (`OpenRAEnv.advance()` nuevo); sin rebuild de Docker
- Interrupciones activas en producción: `unit_destroyed`, `production_complete`,
  `unit_arrived` (~7 cortes/episodio)
- Reward intacto: moldeo en los límites del bloque (step inicial + cierre NO_OP;
  los contadores militares son acumulativos)
- Flags: `--macro-ticks 160 --max-steps 52`; watchdog relanza con estos valores

### Lección registrada: NO hacer resume a través de un cambio de granularidad

El resume v3.1→macro colapsó en ~150 iteraciones: durante la re-calibración del
crítico la política degeneró en "pasear el MCV" (51/51 comandos sobre la unidad
#120, cero producción, loop autoreforzado con `unit_arrived`). Con ARRANQUE
FRESCO bajo semántica macro la construcción diversa volvió en ~10 iteraciones.
**Regla:** un cambio de régimen temporal (frame-skip → macro) exige pesos
nuevos; los checkpoints viejos son artefactos del régimen que los creó.
Preservados: `v4_degenerada_iter395.pt` (el colapso), checkpoint v3.1 = iter 237.

## Fase 1.5 — Reparación del pipeline (BLOQUEANTE, antes de todo train largo)

Detalle completo y evidencia de cada bug:
[`auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md) §2-§3.

| # | Fix | Archivo(s) |
|---|---|---|
| F1 | Coerción completa: índices EFECTIVOS al buffer + recálculo con `h_in` para TODA mutación | `rollout.py` |
| F2 | Vocab persistente: guardado en cada checkpoint y restaurado en `--resume` | `train.py`, `trainer.py` |
| F3 | log_prob solo de las cabezas que la acción usa | `network.py` |
| F4 | Temperatura real (`logits/T` en las 4 cabezas; T=0 argmax total) | `network.py` |
| F5 | Bootstrap de truncado: `last_value=0` si done, `V(s')` si truncate | `rollout.py` (+ valor final en rollout) |
| F6 | Interrupciones contadas SIEMPRE (no condicionadas a telemetry) → metrics | `rollout.py`, `train.py` |
| F7 | Cosecha = Δearned/Δt (no pendiente OLS); corregir etiquetas del dashboard | `economy_race.py`, `dashboard.html` |
| F8 | Sin doble normalización cuando `--adv-mode global` | `trainer.py` |
| F9 | Tests de todo esto (coerción, vocab-resume, GAE-truncado, temperatura) | `verify_offline.py` / tests |
| F10 | Docstring APPO corregido (el update es síncrono) | `train.py` |

Post-victoria (mejora, no bloqueo): `cell_mask`=pasabilidad, condicionar
`dist_cell` a la unidad elegida.

**Criterio de salida de Fase 1.5:** tests nuevos verdes + smoke end-to-end +
diagnóstico con temperatura real coherente. Recién entonces train largo.

## Fase 2 — Curriculum militar (rediseñada post-auditoría)

- **Acción grupal primero**: `army_attack_move` (una decisión mueve todas las
  unidades de combate ociosas). Sin esto ni el curriculum funciona: un rifle
  no gana una guerra y 104 órdenes sueltas por episodio no juegan RA.
- **Escenario A** (`make_scenario.py`): base pre-construida vía `map_data`
  (conyard+powr+proc+barr+weap+8 e1) vs beginner. Economía resuelta: el
  gradiente vive entero en el combate y PUEDE tocar `result=="win"` — hoy el
  término de victoria del engine nunca existe (todas truncadas).
- **Escalera**: A (base completa) → B (base incompleta) → C (solo conyard) →
  D (juego completo). Avance SOLO por winrate.
- **Métrica única norte: winrate contra beginner en escenario A**
  (>50% sostenido ~200 eps = escalón). Reward medio, P(win) heurístico y
  supremacía son diagnóstico, no objetivo.
- Alarma: 0 victorias tras ~2500 eps en A → auditoría con sonda antes de
  tocar parámetros.
- Arranque de pesos: decidir en frío entre fresh o re-calibración vigilada;
  la regla anti-resume-cruzando-semántica pesa a favor de fresh.
- Diversidad barata: rotar mapas desde el día uno (hoy: 1 solo mapa).
- BC del bot scripted como acelerador opcional posterior (requiere
  instrumentar el engine para capturar órdenes — no es gratis).

## Fase 3 — Ensanchar la red 10× (net2net, sin tirar lo aprendido)

- `AlphaLiteNet(ch=64, hidden=256, n_blocks=4)` ≈ 2M params (~8 MB ONNX)
- Transferencia EXACTA por ensanchamiento (Chen et al. 2015):
  - Linears: pesos viejos arriba, filas nuevas en CERO → mismos logits
  - CNN canales 32→64: copia canal a canal ÷2, nuevos en cero
  - ResBlocks extra: inicializados como IDENTIDAD (última conv en cero)
  - GRU 128→256: gates rellenados con ceros; hidden paddeado con ceros
  - Embeddings: columnas nuevas en cero
- Script `widen_checkpoint.py` mapea el state_dict; entra por `--resume`
- Esperado: dip transitorio 5-10 iters (Adam moments nuevos en cero) y recuperación
- Es diagnóstico: si no despega tras ensanchar, la capacidad NO era el cuello
  (señal/diversidad sí); si despega, ganamos velocidad de aprendizaje
- Plan B si net2net falla: destilación (chica enseña a grande mezclando BC+PPO)
- **Orden:** recién después de curriculum — la escala amplifica lo que el
  entorno da; con 1 mapa × 1 rival, amplifica sobreajuste a beginner

## Fase 4 — Liga self-play + PFSP (estilo AlphaStar)

- Reemplazar AI scripted por snapshots propios; liga pequeña 3-5 checkpoints
- Prioritized Fictitious Self-Play: entrenar más contra lo que más te gana
- Requiere: winrate alto vs AI actual primero (no antes)

## Fase 5 — Export ONNX → cliente Flutter

- Mismo pipeline que altotruco_fe e Imperium: export todo-o-nado + vocab sincronizado
- Verificar que las máscaras legales se calculan fuera del grafo (ya es así)

---

## Comandos de operación

```bash
# SIEMPRE con PYTHONPATH= limpio (el desktop inyecta el suyo y rompe venvs)
cd C:/Users/lordc/Desktop/OpenRA-RL

# Servers (3 daemons)
docker compose -f docker-compose.yaml -f docker-compose.scale.yaml up -d

# Trainer (régimen nuevo = sin --resume; continuidad = con --resume)
PYTHONPATH= .venv/Scripts/python.exe -m rl.train \
  --url "http://localhost:8000,http://localhost:8010,http://localhost:8020" \
  --iters 250 --episodes 12 --concurrency 12 --max-steps 4000 --device auto

# Diagnóstico de comportamiento (cada ~50 iters, temp 0.35)
rm -f rl/ckpts/telemetry.jsonl*
PYTHONPATH= .venv/Scripts/python.exe -m rl.diagnose --episodes 2 --temperature 0.35
PYTHONPATH= .venv/Scripts/python.exe -m rl.analyze --telemetry rl/ckpts/telemetry.jsonl

# Tests offline (17 checks, sin server)
PYTHONPATH= .venv/Scripts/python.exe -m rl.verify_offline

# Watchdog: cron job '9acdd65fb42c' (pausar ANTES de matar/relanzar trainer,
# reactivar DESPUÉS de verificar que el nuevo proceso escribe métricas)
```

## Reglas del proyecto

1. Un cambio de régimen por vez (nunca reward+red+junto) — **incluye el estado
   latente**: enganchar reward sobre un resume con vocab potencialmente
   barajado confunde el experimento entero (lección del incentivo minero)
2. Todo reinicio pausa el watchdog y verifica primera iteración antes de reactivarlo
3. Los componentes del reward deben seguir sumando == episode_reward (invariante)
4. Diagnóstico de comportamiento cada ~50 iters: los exploits no se ven en curvas
5. Documentar cada fase acá antes de ejecutarla
6. **Todo fix de gradiente se cierra con un test que falle sin el fix**
   ("[x] aplicado" porque el código existe, sin test, es deuda — lección de la
   auditoría: la coerción "aplicada" tenía 3 bugs)
7. Los .md no afirman comportamiento del código sin verificarlo en fuente
   (docstrings APPO / factorización condicional / "no doble normalización"
   fueron tres afirmaciones falsas convivendo con el código)
