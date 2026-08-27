# Auditoría de pipeline RL — segunda revisión externa (2026-08-24)

Segunda revisión crítica independiente (distinta de la de
[`revision-externa-rl.md`](revision-externa-rl.md)) sobre el estado empírico y
el código. **Cada afirmación fue verificada contra la fuente antes de
aceptarla** — mismo método de siempre. Resultado global: la revisión tiene
razón en lo esencial; varios "fixes aplicados" de la revisión anterior tenían
bugs; la era económica se cierra y el proyecto cambia de secuencia.

---

## 1. Estado empírico al cierre (números reales, no impresiones)

| Señal | iter ~1700 (mejor momento) | iter ~1969 (cierre) |
|---|---|---|
| Reward medio | +3.08 | −1.0 a −1.7 |
| Combate (`cbt`) | −0.03 (paridad) | −1.6 a −2.5 |
| Minería (`min`) | +0.10 | +0.00–0.20 |
| Cosecha propia | ~0 $/kt | ≡ 0 $/kt |
| Cosecha rival | cientos $/kt | −80 a +123 $/kt (ruido, ver B5) |
| Entropía (escala vigente) | — | H ~0.74–0.79 |
| Crítico V(s) | −8.8 | −13 (retorno real ~−1) |
| Winrate | 0 | 0 |
| Resultado de episodios | incomplete@16952t | igual |

**Criterio de salida de la era** (`era-economica.md` §6): cosecha propia
> 100 $/kt en ~300 iters desde el incentivo minero (enganchado ~iter 1630).
**Resultado: FALLO** — ~340 iters después la cosecha propia sigue en cero y el
run estaba *empeorando* (reward +3.08 → negativo, combate paridad → sangrado).

Decisiones tomadas: trainer detenido en iter 1969; checkpoint de la mejor
paridad militar preservado aparte de la rotación:
`rl/ckpts/PRESERVADO_iter1700_paridad_militar.pt`.

## 2. Bugs confirmados contra el código

### Grupo A — núcleo PPO (afectan el gradiente)

| # | Bug | Evidencia en fuente |
|---|---|---|
| A1 | **El ratio de PPO compara acciones distintas**: el buffer guarda los índices MUESTREADOS (`out["type"]`, `unit_slot`, `item_slot` en `pending_sample`) junto al log_prob de la EFECTIVA. Cuando hubo coerción: π_nuevo(a_muestreada) / π_viejo(a_ejecutada). | `rollout.py` `pending_sample` (~L170-185) |
| A2 | **Recálculo de coerción con hidden equivocado**: usa `hidden` reasignado POST-act (L140) en lugar de `h_in` (creado en L120 y sin uso) → log_prob de una distribución temporal distinta a la que muestreó. | `rollout.py` L120 vs L137-147 |
| A3 | **Coerción parcial**: solo entra si `had_item`. Mutaciones sin ítem (attack→attack_move, harvest degradado, deploy) nunca recalculan. Además attack→attack_move cambia el CommandModel pero no `t_name` → el "tipo efectivo" registrado sigue siendo attack. | condición L137; `index_to_command_effective` |
| A4 | **log_prob suma cabezas no usadas**: `evaluate_actions` suma unidad+celda SIEMPRE, incluso en no_op/train/build. La cabeza de celda (~6000 logits) domina la magnitud con ruido puro en acciones que ni la miran. El docstring de la red describe factorización condicional que el código no cumple. | `network.py` `evaluate_actions` (~L225-250) |
| A5 | **GAE sin bootstrap al truncar**: `process_results` llama `add_advantages(traj, gamma, lam)` sin `last_value` → δ_T = r_T. TODAS nuestras partidas son truncadas: el crítico jamás vio "valor restante", solo el pseudo-terminal del margen. | `rollout.py` L364-378 (`next_v = values[-1]`), `train.py` L63 |
| A6 | **Doble normalización con `--adv-mode global`**: `train.py` z-scorea las ventajas en `process_results` y `trainer.update` vuelve a normalizar SIEMPRE (L44-45). El comentario "un solo lugar hace el escalado — no doble normalización" es falso en ese modo. | `trainer.py` L40-45 |

### Grupo B — infraestructura de entrenamiento

| # | Bug | Evidencia en fuente |
|---|---|---|
| B1 | **Vocab no restaurado al resume**: `train.py` crea `vocab = Vocab()` fresco (L101); `load_checkpoint` (trainer.py L144) ignora `ckpt["vocab"]`; `diagnose.py` SÍ lo carga (así funcionaba el diagnóstico). Cada reinicio puede reasignar ids de tipos de la cabeza de ítems → etiquetas barajadas. Explica parte de por qué el incentivo minero "no movió" tras los reinicios del ritual. | grep vocab en train/diagnose/trainer |
| B2 | **Temperature no es temperatura**: `act()` solo hace argmax si T≤0; NUNCA divide logits por T. `diagnose --temperature 0.35` equivalía a T=1.0 en muestreo (y greedy solo en tipo). El "best effort" del diagnóstico no existía. | `network.py` `act()` (~L181-210) |
| B3 | **Interrupts siempre `{}`**: el contador server-side corre, pero `train.py` nunca pasa `telemetry` → la condición `telemetry is not None` es siempre falsa y metrics.jsonl registra `"interrupts": {}` desde hace cientos de iters. Ciegos sobre si `enemy_spotted`/`unit_destroyed` disparan. | grep telemetry en train.py (vacío) |
| B4 | **cell_mask = todo el mapa**: `torch.ones(h*w)`; el canal de pasabilidad existe en la observación y no se usa. Un MCV y un rifle reciben el mismo mapa de destinos (agravado por `dist_cell` ciego a la unidad elegida — ya anotado). | `action_adapter.py` L116 |
| B5 | **Métrica de cosecha mal definida**: `_slope_1k` es regresión OLS sobre una serie MONÓTONA que luego platanea → pendientes imposibles (medido: enemigo −147 $/kt; la recaudación bruta no puede bajar). Correcto: `(earned_fin − earned_inicio) / Δticks × 1000`. Parte del dashboard de la era lee un estimador sesgado. | `economy_race.py` L82-99 |
| B6 | **Docstring falso "solapamiento APPO"**: `train.py` describe recolección k+1 solapada con update k, pero `trainer.update` es síncrono y bloquea el event loop; el `ensure_future` no corre hasta terminar el update. No es grave (PPO on-policy queda más limpio así), pero documenta una arquitectura inexistente. | `train.py` L6-8 y L171 |

### Grupo C — proceso y cobertura

| # | Falla | Consecuencia |
|---|---|---|
| C1 | `revision-externa-rl.md` marcaba "[x] Log_prob de acción efectiva APLICADO" basándose en que el código existía, SIN un test que demostrara la consistencia buffer/ratio. | Los tres sub-bugs A1-A3 pasaron por aplicados. Corregido en ese doc. |
| C2 | El incentivo minero se enganchó sobre RESUME (~iter 1630) combinando cambio de reward + continuidad de pesos + vocab potencialmente barajado (B1). | Experimento confundido: imposible atribuir cuánto del fracaso es diseño vs bug. Mismo patrón que el colapso v4 ("no resume cruzando semántica"), otra piel. |
| C3 | `verify_offline` (17 checks) no ejercita macro, coerción, vocab-en-resume ni GAE-de-truncado; FakeEnv no tiene `advance()`. Hay ~15 scripts de análisis one-shot y casi ningún test del pipeline que realmente corre. | Los bugs A1-B2 eran testeables y nadie los testeaba. |

### Lo que la revisión exageró o matizamos

- **"iter1700 perdido por rotación"**: falso — los checkpoints cada 10 iters
  siguen todos; se preservó copia blindada igualmente porque su punto de fondo
  vale: ese checkpoint vale más que latest.
- **Entropía 1.54 → 0.74**: mezcla dos escalas (1.54 era solo cabeza tipo;
  0.74 es el promedio ponderado nuevo). La lectura conductual (política más
  determinista) es correcta igual.
- **BC del bot scripted**: idea valiosa pero NO es "unas horas": capturar las
  órdenes del bot requiere instrumentar el engine. Queda como acelerador
  opcional DESPUÉS del pipeline, no como paso gratis.
- **P(win) del dashboard "es teatro"**: cierto mientras no haya victorias —
  ya estaba etiquetado así en el propio dashboard; coincide con nuestra
  postura de `calibrar_pwin.py` (rechazo honesto sin corpus de wins).

## 3. Plan aprobado (operador, 2026-08-24) — ✅ FASE 1 APLICADA Y VERIFICADA

### Fase 1 — Parar el sangrado y reparar el pipeline (APLICADA 2026-08-24)

1. Trainer detenido ✅ (iter 1969; watchdog pausado durante la obra)
2. **F1 coerción completa** ✅: índices EFECTIVOS al buffer + recálculo con
   `h_in` para TODA mutación. Incluye la degradación temprana
   `attack→attack_move` en `action_adapter.py` (antes el tipo efectivo
   quedaba "attack"). Test: coherencia buffer/ratio Δ=0.0000.
3. **F2 vocab persistente** ✅: `load_checkpoint(vocab=...)` restaura el
   vocabulario del checkpoint; train.py lo pasa al resume.
4. **F3 log_prob condicional** ✅: conjuntos únicos `TYPES_USE_*` en
   network.py; act() y evaluate_actions suman solo cabezas usadas. Test:
   |lp(no_op)| 2.44 << |lp(move)| 12.41.
5. **F4 temperatura real** ✅: `logits/T` en las 4 cabezas; T=0 → argmax
   total. Test: T=3.0 explora 11 tipos; greedy determinista.
6. **F5 bootstrap de truncado** ✅: rollout computa `_v_next` (V(s')) al
   truncar; add_advantages lo consume y lo limpia. Test: δ_T exacto 2.5.
7. **F6 interrupciones contadas siempre** ✅ (independiente de telemetry) →
   llegan a metrics.jsonl vía outcome["interrupts"].
8. **F7 cosecha = Δearned/Δt** ✅ (`_delta_1k`); OLS queda SOLO para riqueza
   (no monótona). Test: serie monótona → 40.0 $/kt exactos, jamás negativa.
   Pendiente menor: retitular gráfico del dashboard.
9. **F8 sin doble normalización** ✅: trainer ya NO z-scorea; el escalado
   vive en process_results ('episode' = centrado por ep + std del batch,
   escala histórica conservada; 'global' = z-score del batch). Test:
   update NO muta ventajas.
10. **F9 tests** ✅: verify_offline ampliado a ~30 checks en 12 secciones;
    TODOS VERDES. Además test_economy_race E2E contra server real: OK con
    métrica nueva (earned_total propio 0 vs rival 3000 medido en vivo).
11. **F10 docstring APPO corregido** ✅ (describe update síncrono real).

Verificación adicional: diagnóstico de comportamiento corrió por primera vez
con temperatura REAL (T=0.35 divide logits).

Quedan para DESPUÉS de la primera victoria (mejora, no bloqueo):
`cell_mask`=pasabilidad (B4), condicionar `dist_cell` a la unidad elegida,
retitular el gráfico de cosecha del dashboard.

### Fase 2 — Cambiar el problema (no el peso) — SIGUIENTE

12. **Acción grupal ya**: `army_attack_move` (una decisión mueve todas las
    unidades de combate ociosas). Sin esto ni el curriculum funciona:
    un rifle no gana una guerra y 104 órdenes sueltas por episodio no juegan RA.
13. **Escenario A** (`make_scenario.py`): base pre-construida vía `map_data`
    (conyard+powr+proc+barr+weap+8 rifles) vs beginner. Economía resuelta; el
    gradiente vive entero en el combate y PUEDE tocar `result=="win"`.
14. **Train largo recién entonces**, desde pesos limpios post-F1 (decidir en
    frío entre latest y fresh; la regla anti-resume-cruzando-semántica sugiere
    fresh o mínimo re-calibración vigilada).
15. **BC del bot scripted** como acelerador opcional posterior.

### Resultados del control post-reparación (50 iters, 1970-2019, 600 eps)

Corrido tras aplicar F1-F10, mismo régimen, resume desde iter 1969.
Objetivo: medir si el gradiente reparado cambia la tendencia ANTES de
cambiar el problema. Conclusión: **el sangrado se frenó, pero no hay
despegue — confirma que el cuello es el espacio de acción/objetivo, no los
bugs restantes** (justifica pasar a Fase 2 ya).

| Señal | Cierre era vieja | Control F1-F10 |
|---|---|---|
| Reward medio | −1.0 a −1.7 (cayendo) | −1.13 estable (tercios −1.22/−0.96/−1.19) |
| Entropía H | 0.74–0.79 | 0.99–1.04 (exploración restaurada ✅) |
| Clip frac | hasta 0.24 | 0.02–0.08 (muy estable ✅) |
| `min` | 0.00–0.10 | 0.01→0.21 (subiendo ✅) |
| Cosecha propia | ≡ 0 absoluto | +0.6 medio, >0 en 5/50, max +15 $/kt (primeras señales de vida ✅) |
| Interrupts en métricas | `{}` eterno | 50/50 filas pobladas (F6 ✅): building_discovered 4695, enemy_building_destroyed 4275, enemy_spotted 1816, own_building_destroyed 957, unit_destroyed 699, under_attack 108 |
| Combate (`cbt`) | −1.75 a −2.55 | −2.23 → −2.69 (sin cambio — sigue perdiendo todo intercambio) |
| Winrate / supremacía | 0 | 0 · diff material −4831, P(win) est 0.18 |

Lectura: los bugs de gradiente estaban degradando la exploración y la señal
(H recuperada, clip sano), pero el agente sigue sin poder cerrar peleas ni
partidas con 1 comando cada 160 ticks. El plan se mantiene: Fase 2.

**La única métrica norte de las próximas semanas: winrate contra beginner en
escenario A.** Reward medio, P(win) heurístico y supremacía son diagnóstico,
no objetivo.

## 4. Lecciones meta de esta auditoría

1. **Checkbox sin test = deuda**: marcar "[x] APLICADO" porque el código
   existe (no porque un test demuestre el comportamiento) convirtió un fix
   sesgado en "aplicado y verificado". Regla nueva: todo fix de gradiente se
   cierra con test que falle sin el fix.
2. **Docs que afirman cosas del código sin verificarlas** (APPO, factorización
   condicional, "no doble normalización") son bugs de documentación tan reales
   como los de código.
3. **Cambiar reward + resume + vocab en un mismo evento** confunde cualquier
   conclusión posterior. La regla "un cambio de régimen por vez" incluye el
   estado latente del proceso, no solo los pesos.
4. **Instrumentar sirvió para VER el fracaso rápido** (la carrera económica
   detectó cosecha≡0 en horas). La disciplina de medición del proyecto es
   real; lo que faltaba era testear el pipeline que genera el gradiente.
