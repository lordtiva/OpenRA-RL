# Revisión externa de RL (2026-08-24) — evaluación y decisiones

Revisión crítica recibida sobre teoría de RL y dinámica RTS. Este documento
registra el veredicto punto por punto CONTRA EL CÓDIGO REAL (cada afirmación
fue verificada en fuente antes de decidir) y qué se adopta, qué se difiere y
qué se rechaza. Principio: un cambio de régimen por vez, evidencia primero.

## Veredicto por punto

| # | Afirmación externa | Verificación | Decisión |
|---|---|---|---|
| 1 | Embudo: 1 acción/decisión, 104 por episodio | CIERTO (por diseño v4-macro) | Diferido: macro-acciones de escuadrón en roadmap medio plazo |
| 2 | `dist_cell` ciega a la unidad elegida | **CIERTO** (`dist_cell(fmap, chosen_type, mask)` solo condiciona en tipo) | ADOPTADO (medio plazo): condicionar cabeza de celda al embedding de la unidad |
| 3 | GRU sin BPTT = sin memoria aprendible | **CIERTO** (documentado como simplificación deliberada desde v1) | Diferido con razón registrada; limitación conocida |
| 4 | Óptimo local SimCity por reward shaping | PARCIAL: los pesos favorecen construir, pero el agente SÍ combate (cbt activo, escaramuzas medidas) | Parcialmente adoptado: rebalanceo DESPUÉS de leer el experimento minero |
| 5 | Centrado per-episodio destruye señal GAE | CIERTO que existe y corre siempre (`train.py:52`); el impacto depende de qué tan mala sea la línea de base | EXPERIMENTO A/B: flag CLI, medir pendiente de reward con/sin |
| 6 | Coerción post-muestreo viola Policy Gradient | **CIERTO Y SUTIL**: se ejecuta a' pero se atribuye gradiente de a | ADOPTADO (inmediato-2): registrar log_prob de la acción EFECTIVA — ⚠️ la primera aplicación (2026-08-24) tenía 3 bugs; ver [`auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md) §2-A1..A3 |
| 7 | Entropía solo en cabeza de tipo | **CIERTO** (`evaluate_actions`: `entropy = dist_t.entropy()`, comentario "proxy") | ADOPTADO (inmediato): suma de entropías de las 4 cabezas |
| 8 | Interrupción `unit_arrived` roba presupuesto | CIERTO (está en `_DEFAULT_INTERRUPTS`) | ADOPTADO (inmediato): filtrarla de la lista default |
| 9 | Pool espacial 32-dim = cuello de botella | CIERTO pero coherente con nuestra decisión previa | RECHAZADO por ahora: coincide con "no agrandar sin señales de subajuste" |

## Detalle de lo adoptado

### Inmediato (próxima ventana de reinicio)
1. **Entropía total**: `H_type + H_unit + H_cell + H_item` en `evaluate_actions`
   (ponderar 1/N para mantener escala del coeficiente).
2. **Quitar `unit_arrived`** de `_DEFAULT_INTERRUPTS` (server-side, requiere
   rebuild docker — agrupar con próxima ventana C#).
3. **Flag `--no-center-advantage`** para el experimento A/B del centrado:
   hipótesis de la revisión vs nuestra lección alto-truco (la línea de base del
   crítico está DEMOSTRADAMENTE descalibrada — dijo −9.50 cerrando +4.10 — así
   que el centrado grupal puede estar actuando como baseline de emergencia).
   Ganador = mejor pendiente de reward en ~150 iters.
4. **Coerción honesta**: tras las correcciones de seguridad en
   `index_to_command`, recalcular el índice efectivo y guardar ESE log_prob
   (re-evaluar distribuciones con los índices ejecutados). Elimina el sesgo de
   crédito sin tocar el espacio de acción.

### Medio plazo (después de leer el experimento minero)
5. Condicionar `dist_cell` al embedding de la unidad elegida (proposición
   concreta de la revisión, correcta: hoy un MCV y un rifle reciben el mismo
   mapa de destinos). OJO: cambia shapes → cargar checkpoint con
   `strict=False` e inicializar solo la proyección nueva.
6. Macro-acciones de escuadrón (attack_move grupal de ociosos): transforma la
   aplicación militar sin rediseñar todo el espacio.
7. Rebalanceo SimCity (bajar `w_building`/`w_new_type`) SOLO si con minería
   resuelta el combate sigue dominado por el pago seguro de ladrillos.

### Rechazado/diferido con motivo
- **Net2Net 10× ahora**: coincidimos con la revisión (y con nuestra decisión
  previa): el cuello es señal y espacio de acción, no capacidad.
- **BPTT por chunks**: costoso en estabilidad (lecciones propias de
  entrenamientos recurrentes inestables); la memoria del GRU hoy aporta estado
  de inferencia aunque no aprenda dependencias largas. Queda registrado como
  limitación estructural conocida; prioridad baja hasta que combate y economía
  estén resueltos.

## Coincidencias que validan el rumbo

La revisión, independiente, concluye lo mismo que nuestro diagnóstico interno
en tres puntos clave: Fase 2 (curriculum militar) es el camino correcto,
agrandar la red es prematuro, y la economía estaba ausente del problema.
También anticipó parcialmente el hallazgo minero (reward de edificios seguro >
riesgo militar), que ya habíamos medido con la carrera económica.

## Estado de aplicación

> ⚠️ **Corrección 2026-08-24 (auditoría posterior):** los ítems marcados
> "[x] coerción honesta" abajo fueron RE-VERIFICADOS contra el código y
> resultaron tener 3 bugs (buffer con índices muestreados, hidden post-act,
> cobertura parcial de mutaciones). Detalle completo en
> [`auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md) §2.
> La lección quedó registrada: *checkbox sin test que falle sin el fix = no
> está aplicado*. Los demás [x] sí fueron verificados en vivo.

- [x] Entropía total (APLICADO 2026-08-24: suma ponderada de 4 cabezas,
      celda x0.25 e ítem x0.5, normalizada /2.75; escala nueva del métrico H
      NO comparable con la histórica — H_type ≈ 1.1 ahora muestra ~0.53)
- [x] Filtrar `unit_arrived` (APLICADO 2026-08-24, rebuild docker incluido)
- [x] Flag A/B centrado de ventaja (APLICADO como `--adv-mode
      episode|global|none`; verificado en frío: episode anula la señal del
      episodio malo [0,0], global la conserva [−0.91 vs +1.49]. PERO: la
      auditoría encontró doble normalización cuando adv-mode=global —
      trainer vuelve a z-scorear; pendiente F8)
- [ ] Log_prob de acción efectiva (⚠️ DESMARCADO por la auditoría: la
      aplicación tenía bugs A1/A2/A3 — buffer con índices muestreados,
      recálculo con hidden equivocado, mutaciones sin ítem sin recalcular.
      Refacción completa = F1 del plan de reparación)
- [ ] Condicionar celda a unidad (medio plazo)
- [ ] Escuadrones / attack_move grupal (medio plazo → PROMOVIDO a inmediato:
      Fase 2 del plan de reparación)
- [x] No agrandar red todavía (decisión mantenida; la segunda revisión
      coincide explícitamente)
