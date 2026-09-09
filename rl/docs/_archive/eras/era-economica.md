# Era económica — espectador exacto, carrera económica e incentivo minero

> ⚠️ **ERA CERRADA (2026-08-24).** El criterio de éxito de §6 NO se cumplió:
> ~340 iters después del incentivo minero la cosecha propia seguía en ≡ 0 y el
> run empeoraba (reward +3.08 → −1.0/−1.7; combate paridad → sangrado). Una
> auditoría posterior encontró además bugs en el pipeline de gradiente que
> confundieron este experimento (vocab barajado por resumes entre ellos).
> Veredicto, evidencia y plan de reemplazo:
> [`auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md).
> Este documento queda como registro de lo aprendido (que sigue siendo válido)
> y como referencia histórica.

**Estado:** activa desde el 2026-08-24 (~iter 1630 del régimen horizonte doble) hasta iter 1969.
Este documento registra qué aprendimos, qué cambiamos y por qué. Es la fuente
de verdad de la fase; los regímenes anteriores están resumidos en
`roadmap-agente.md`.

---

## 1. El hallazgo que abrió esta era

Implementamos el **modo espectador** (`RlGlobalSummary`, campo proto 21): el
server serializa por bando, en cada tick avanzado, valor de unidades/edificios,
efectivo, kills/deaths en $, conteos y (desde esta era) la **recaudación bruta
acumulada** (`PlayerResources.Earned`). El agente sigue jugando con niebla —
el resumen es solo para EVALUAR.

Con datos exactos de ambos bandos apareció la primera corrección grande al
modelo mental:

> La supremacía foggeada histórica mostraba "$0 enemigo" (nada visible).
> Con la verdad de terreno: **veníamos perdiendo la carrera material**
> (~$3.8k propios vs ~$6.1k del rival) y, peor aún, **no cosechábamos nada**.

## 2. Diagnóstico minero (por qué nunca cosecha)

Método: métrica nueva de **carrera económica** (ver §3) + sonda de política
(`rl/probe_minero.py`) + lectura del código de máscaras. Evidencia:

| Señal | Valor medido | Lectura |
|---|---|---|
| Recaudación propia | **+2 a +12 $/kt** | prácticamente nula |
| Recaudación rival | +420 a +740 $/kt | el script mina desde temprano |
| Edificios colocados | ~4.8/episodio | SÍ construye estructuras |
| Tipos nuevos desbloqueados | ~2.7/episodio | pero ninguna refinería operativa |

**Causa 1 — reward:** encolar una refinería ($300, pago diferido vía ingreso
diluido en `w_assets=0.003`) compite contra rifles con feedback inmediato vía
`w_combat`. La balanza aprendida: todo el subsidio inicial ($5000) va a
infantería; cuando se agota, no hay economía propia para el resto del episodio.
Nadie le dijo jamás que minar importa.

**Causa 2 — bug estructural (descubierto en el proceso):** `_split_production`
clasificaba como "entrenables" todos los ítems producidos por cualquier edificio
propio. Como el cuartel de mando produce TANTO unidades (e1, perro) COMO
edificios (proc, powr), las refinerías caían en el cubo equivocado y la acción
`build` quedaba casi siempre enmascarada. Mismo cubo para rifles y refinerías =
crédito confuso justo en la acción que había que aprender.

## 3. Instrumental nuevo (todo lado agente salvo un campo proto)

| Pieza | Archivo | Qué hace |
|---|---|---|
| `EconomyRace` | `rl/economy_race.py` | Series por episodio de riqueza Y recaudación de ambos bandos; pendientes por regresión ($/kt); decima a 400 muestras |
| Carrera en métricas | `rl/train.py` | `economy_race` (agregados) en metrics.jsonl; series completas en `rl/ckpts/economy_race.jsonl` |
| Dashboard | `dashboard.html` | Gráfico "Carrera económica": cosecha verde/rojo (principal) + riqueza cian/ámbar (secundaria); tarjeta ventaja material + barra Lichess + P(win) |
| `earned` en GlobalSummary | `proto/rl_bridge.proto` ×2 + `ObservationSerializer.cs` + `bridge_client.py` | Campo `Side.earned = PlayerResources.Earned` (recaudación acumulada real del motor). OJO: `stats.Income` NO sirve — es tasa por segundo para el HUD |
| Rotación de eras | `rl/rotar_datos.py` | Archiva metrics+series con timestamp y deja archivos nuevos con marca de era |

### Cómo leer las dos curvas económicas

- **Cosecha (`*_harvest_per_1k`, de `earned`)**: cuánto mineral se extrae cada
  1000 ticks. Siempre ≥ 0, inmune a las muertes. Responde *"quién produce"*.
- **Riqueza (`*_income_per_1k`, pendiente de cash+unidades+edificios)**:
  patrimonio neto. Sube al cosechar/gastar, baja al perder activos. Puede ser
  negativa tarde en el episodio (desgaste). Responde *"quién conserva"*.
  ⚠️ No confundir con recaudación: matarle un rifle al rival NO baja su cosecha.

Honestidad de las series: antes del primer GlobalSummary del episodio el rival
es niebla (y `earned` propio aún no fue leído); las pendientes se computan solo
desde el primer dato conocido — arrastrar ceros inflaba la estimación rival 30×.

### Episodios zombi (filtro anti-contaminación)

El daemon .NET agotado escupe sesiones muertas que producen observaciones
$0-vs-$0. Contaminan medias y corpus. Ya se filtran en: media/corpus de
supremacía (`_sup_valida`), carrera económica (`own_wealth_end > 0`) y
calibrador L3 (filas sin ningún dato de bando).

## 4. Cambios de régimen aplicados (era económica)

| Cambio | Dónde | Justificación |
|---|---|---|
| Split estático train/build | `action_adapter.py` (`BUILDING_ITEM_TYPES`) | bug estructural §2-Causa 2; higiene, no reward |
| Incentivo minero | `reward_shaping.py` (`w_refinery=1.0` primera proc operativa, `w_harvester=0.25` por cosechadora hasta 4) | §2-Causa 1; mismo espíritu no-explotable de `new_types`: pagar hitos acotados, no flujo abierto |
| Watchdog autónomo | `openra_ppo_monitor.py` | el daemon se agota en ~1-2h bajo horizonte doble; 2+ muertes/hora → recrea contenedores él solo (+ limpieza de `openra-rl-agent-1`) antes de relanzar |
| Reset con reintentos | `rl/train.py` worker | 3 intentos ante "bridge failed to start"; aborta limpio si persiste |

**Lo que deliberadamente NO hicimos (y por qué):**

- Pagar ingreso bruto continuo → exploit de "simulador de cosechadoras"
  (mismo patrón que el exploit powr de v2). El ingreso no exige conversión militar.
- Margen terminal de recaudación `tanh((earned_nos−earned_rival)/escala)` →
  EN RESERVA: cero-sumo e inmune a farmeo; siguiente escalón si el incentivo
  minero no mueve la cosecha propia.
- Curriculum militar (Fase 2, `fase2-curriculum.md`) → sigue en cola; ahora con
  mejor justificación (sin economía propia no hay tech tree que sostenga).

## 5. Lecciones caras de esta era (para no repetir)

1. **La niebla mintió por omisión**: toda evaluación basada en lo visible
   subestimaba al rival invisible. El modo espectador corrigió el signo del
   diagnóstico completo.
2. **`stats.Income` ≠ recaudación acumulada**: es tasa/segundo del HUD.
   El acumulado real es `resources.Earned`. Leer la semántica del engine ANTES
   de diseñar métricas.
3. **Regenerar stubs proto pisa parches**: el import absoluto de grpc_tools
   rompe el paquete y los campos nuevos desaparecen del descriptor viejo
   (falla silenciosa → AttributeError recién en runtime). Checklist post-regen:
   import relativo + verificar descriptor + rebuild docker.
4. **El can_produce del server no sirve para clasificar train/build**: lista
   todo lo producible por cada productor. Clasificación estática por nombre.
5. **Healthcheck HTTP verde ≠ daemon sano**: la vida útil interna del daemon
   .NET termina sin señal externa; los síntomas son episodios $0-vs-$0 primero
   y muerte en reset después.
6. **Una métrica bien hecha vale más que diez opiniones**: el debate sobre
   "¿le pagamos la recolección?" se resolvió midiendo: +12 vs +420 $/kt.

## 6. Estado y criterios de salida (CERRADA — ver banner inicial)

- Trainer: horizonte doble (104 decisiones ≈ 16.95k ticks), ~22 s/iter,
  resume continuo; consola imprime `min` (incentivo minero) y
  `eco[cosecha nos X vs rival Y | riqueza ...]`.
- **Criterio de éxito del incentivo minero**: cosecha propia sostenida > 100
  $/kt en el dashboard dentro de ~300 iters. Si no aparece ni el intento
  (cosecha ≡ 0 y `min` ≈ 0), revisar si `build:proc` llega a ejecutarse
  realmente (telemetría de comandos) antes de subir pesos.
  **RESULTADO FINAL: NO CUMPLIDO.** Cosecha propia ≡ 0 tras ~340 iters;
  sonda directa (`rl/probe_cosecha.py`) confirmó refinerías compradas pero
  sin producción de ingreso; el run se detuvo en iter 1969.
  ⚠️ Además, la auditoría posterior halló que la métrica usaba pendiente OLS
  sobre serie monótona (puede dar negativa — imposible para recaudación);
  la métrica correcta Δearned/Δt entra con el plan de reparación.
- **Escalón siguiente en reserva**: margen terminal de recaudación →
  DESCARTADO junto con toda la vía "parchar el shaper" (veredicto de la
  auditoría: el problema es espacio de acción + objetivo, no pesos).
- **Lo que sobrevive a esta era** (instrumental válido): modo espectador +
  `earned`, filtrado de episodios zombi, rotación de datos, watchdog
  autónomo, split estático train/build, lecciones §5.
- **Reemplazo**: plan F1-F10 (reparación de pipeline) + Fase 2 curricular con
  base pre-construida y winrate como métrica única —
  [`auditoria-pipeline-2026-08-24.md`](auditoria-pipeline-2026-08-24.md) §3.
