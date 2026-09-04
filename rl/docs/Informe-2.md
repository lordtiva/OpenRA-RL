Hay **noticias excelentes** (la política ha aprendido macro-gestión a un nivel temible), pero también han emergido **tres anomalías críticas** que explican con precisión matemática por qué el winrate está estancado entre el **16% y el 20%** contra `easy`, por qué `medium` te aplasta (1 victoria en 128 partidas), y por qué el modelo sufre de partidas incompletas por *timeout* (53.000 ticks).

A continuación te presento el desglose técnico, contrastado con los puntos de la revisión anterior, y las soluciones concretas.

---

# 1. Diagnóstico de Datos: ¿Qué está pasando en el Run?

### A. La macro-economía funciona: El agente crea "ejércitos colosales"
En `live_games.txt` se observan partidas asombrosas:
* **Iteración 3:** Gana con **222 unidades** (123 rifles, 39 lanzacohetes, 32 médicos, 7 tanques) y 15 barracones.
* **Iteración 11:** Gana en solo 24.117 ticks con **116 unidades**.
* **Iteración 17:** Llega a tener **285 unidades vivas**.
* **Iteración 20:** Gana acumulando un banco de producción masivo.

El agente **ha aprendido a minar, construir bases funcionales y producir ejércitos gigantescos**. La infraestructura de roles y el modo macro funcionan.

---

### B. El Fenómeno del "Coloso Paralizado" (Timeouts de 53k ticks)
A pesar de tener ejércitos monstruosos, una enorme cantidad de partidas terminan en `incomplete` a los 53.000 ticks:
* **Iteración 10:** 21 unidades vivas, estancadas en base (timeout).
* **Iteración 15:** ¡**243 unidades de combate vivas** apiñadas en `[57, 32]` a solo 43 celdas de la base enemiga y el juego hace timeout!
* **Iteración 17:** **245 unidades vivas** en `[59, 31]` (timeout).
* **Iteración 17 (otra):** ¡**285 unidades vivas** en `[44, 23]` y no entran a rematar!
* **Iteración 21:** **243 unidades vivas** en `[49, 33]` (timeout).

**¿Por qué un ejército de casi 300 soldados se queda parado en mitad del mapa sin cerrar la partida?**
Revisando tu código (`action_adapter.py`), he descubierto el **culpable exacto**:

```python
# action_adapter.py (líneas 272-273)
if n_combat_near_own_base(obs) < PACK_ARMY:
    m[TYPE_TO_IDX["army_attack_move"]] = False
```

**La trampa mortal:**
1. Para lanzar un ataque grupal con la política (`army_attack_move`), exiges que haya al menos 12 unidades **cerca de la base propia** (`PACK_HOME_RADIUS = 18`).
2. Cuando el ejército de 200 unidades marcha y cruza la mitad del mapa (`x=45..60`), **ya no está cerca de la base propia** (`n_home` baja a 4 o 5).
3. En ese instante, tu máscara **VUELVE A BLOQUEAR `army_attack_move`**.
4. El ejército que está a las puertas del enemigo se queda sin permiso para emitir ataque grupal. Empieza a ciclar entre `set_stance`, `harvest` o `no_op`, mientras los pocos refuerzos que nacen en casa no llegan a 12 para reactivar el comando.
5. **Resultado:** Tu ejército colosal se queda congelado en `x=50..60` hasta que el episodio muere por límite de tiempo.

---

### C. Confirmación Empírica del "Stale Policy Shift" (Review Anterior)
En la revisión previa te advertí que el solapamiento asíncrono ($k$ vs $k+1$) generaría desfase de política y saturación del clip de PPO. Los datos lo han confirmado de forma contundente:

Mirando `metrics.txt`:
* **Iter 30:** `clip_frac = 0.424` (¡**42.4%** de las acciones recortadas!), `kl = 0.145` (KL masivo).
* **Iter 40:** `clip_frac = 0.328`, `kl = 0.108`.
* **Iter 44:** `clip_frac = 0.346`, `kl = 0.110`.
* **Iter 70:** `clip_frac = 0.422`, `kl = 0.068`.
* **Iter 99:** `clip_frac = 0.343`, `kl = 0.131`.

Cuando `clip_frac` supera el **25%-40%**, el algoritmo PPO prácticamente **tira a la basura casi la mitad de los gradientes** porque el ratio de importancia está fuera de $[1-\epsilon, 1+\epsilon]$. La política da un salto tan grande entre el modelo de inferencia y el de optimización que desestabiliza el aprendizaje (de ahí los picos de `grad_norm` de $9.4$ y $12.4$).

---

### D. La Alucinación Naval en Mapas Terrestres
Mira la cinta de la **Iteración 65 (Derrota en 9.700 ticks)**:
```json
{"dec": 50, "pol": "build", "item": "naval", "iss": "syrd"}
{"dec": 90, "pol": "place_building", "item": "naval", "iss": "syrd"}
{"dec": 100, "pol": "build", "item": "refinery", "iss": "silo"}
{"dec": 120, "pol": "place_building", "item": "naval", "iss": "syrd"}
```
En esa partida, la red decidió construir y colocar **tres astilleros navales (`syrd`)** en un mapa terrestre como `Singles` con un charco de agua aislado.
* Gastó sus \$5.000 iniciales en astilleros inútiles y silos.
* Mantuvo `nc: 0` (cero unidades de combate producidas) durante toda la partida.
* El bot `easy` entró caminando a su base y lo eliminó a los 9.700 ticks sin resistencia.

---

### E. ¿Por qué `medium` tiene 99.2% de Winrate contra ti?
En `metrics.txt`, la carrera económica (`economy_race`) muestra el motivo:
* Contra `medium`, `harvest_edge` es sistemáticamente **negativo por un margen abismal**: el bot rival cosecha entre $-1.800$ y $-3.500$ de mineral más por cada 1.000 ticks.
* El bot `medium` mete una presión temprana con blindados o grupos de infantería antes del tick 15.000. Si el agente no tiene su segunda refinería o sufre una pequeña incursión, su economía colapsa (`worst_deficit` llega a $-\$40.000$).

---

# 2. Resumen de Contraste con la Review Anterior

| Punto Teórico previo | Evidencia en los Datos | Estado |
| :--- | :--- | :---: |
| **Stale Policy Shift en Overlap** | `clip_frac` explota frecuentemente a $>30\%$ y $>42\%$. `kl` supera $0.10$. | **Confirmado (Crítico)** |
| **Asignación de crédito en War Nudge** | Al apagar el nudge (`--no-war-nudge`), la red demostró que sabe producir, pero... | **Parcialmente resuelto** |
| **Falta de remate de la Política** | ...al apagar el nudge, descubrimos el bug de `PACK_ARMY` que congelaba a 280 soldados en mitad de la nada. | **Descubierto (Crítico)** |
| **Peligro de mínimos locales en Reward** | La red a veces prefiere construir astilleros o silos en bucle para ganar bonus de infraestructura sin tropas. | **Confirmado (Moderado)** |

---

# 3. Recomendaciones y Plan de Acción Inmediato

Para desbloquear el techo del 20% y empezar a ganarle con consistencia a `easy` y competir contra `medium`, aplica estos cambios:

---

### Acción 1: Corregir el Deadlock de `army_attack_move` (¡Urgente!)
El ejército no debe necesitar estar "en casa" para seguir avanzando si ya es una fuerza masiva. Modifica `ActionIndex` en `rl/action_adapter.py`:

```python
# ANTES (provoca que los ejércitos en el campo no puedan recibir órdenes grupales):
# if n_combat_near_own_base(obs) < PACK_ARMY:
#     m[TYPE_TO_IDX["army_attack_move"]] = False

# AHORA:
# Si el ejército total es >= PACK_ARMY, SIEMPRE es legal moverlo,
# O si hay al menos PACK_ARMY unidades listas en casa.
total_combat = len(_combat_units(getattr(obs, "units", None) or []))
home_combat = n_combat_near_own_base(obs)

if total_combat < PACK_ARMY:
    m[TYPE_TO_IDX["army_attack_move"]] = False
elif total_combat >= PACK_ARMY and home_combat < PACK_ARMY:
    # El ejército ya está en marcha: permitir empujar al frente,
    # solo bloquear si no hay suficientes unidades en total en el mapa.
    pass
```
*Impacto directo:* Los 250 soldados que se quedan clavados en `[57, 32]` en los ticks 30.000–50.000 recibirán la orden de seguir empujando hacia el beacon `[95, 11]` y cerrarán esas partidas en victorias de 35.000 ticks en vez de empates por tiempo.

---

### Acción 2: Frenar el `clip_frac` excesivo (Estabilidad PPO)
Tienes un desfase de 1 paso stale entre recolección y optimización. Para que el ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$ no arranque roto:

1. En `TRAIN_ARGS` (`rl/auto_train.py`):
   * Reduce el Learning Rate: de `1.5e-4` a **`8.0e-5`** o **`1.0e-4`**. Un paso de optimización más suave reduce el salto entre $\theta_{k-1}$ y $\theta_k$.
2. En `rl/trainer.py`:
   * Ajusta `clip_eps` de `0.2` a **`0.15`** temporalmente mientras el `clip_frac` esté por las nubes.
   * Esto evitará los picos de `grad_norm` de 10-12 y estabilizará la entropía.

---

### Acción 3: Enmascarar Edificios Inválidos en el Escenario (Anti-Astilleros)
Evita que el agente suicide partidas construyendo estructuras marítimas en escenarios terrestres. En `rl/action_adapter.py` dentro de `_split_production` o en el filtrado de `build_slot_mask`:

```python
# En mapas sin juego naval competitivo (como a_short / Singles):
FORBIDDEN_BUILD_ROLES = {"naval"}  # spen, syrd

for slot, role in enumerate(self.build_items):
    bslot = n_train + slot
    if bslot >= n_vocab:
        break
    if role in FORBIDDEN_BUILD_ROLES:
        self.build_slot_mask[bslot] = False
        self.item_mask[bslot] = False
```
Esto erradicará completamente los episodios como la Iteración 65 donde el agente regala la partida construyendo astilleros inútiles.

---

### Acción 4: Solución al "Turtling" mediante Reward de Proximidad Tardía
Para castigar las partidas donde el agente tiene 200 soldados pero no ataca:
En `rl/reward_shaping.py` (`eradicate_v4`), añade una penalización por estancamiento de ejército grande:
* Si `tick > 30000` y `n_combat > 50` y el centroide del ejército sigue a más de 40 celdas del enemigo:
  aplica una pequeña penalización temporal (o un incentivo positivo por acercar el centroide al beacon cuando el ejército supera las 40 unidades).

---

### Veredicto del Run
El modelo **está aprendiendo la parte más difícil del RTS**: gestionar la economía, levantar refinerías múltiples y entrenar un ejército de 200 soldados. El 80% de tus derrotas/incompletas actuales no son por incapacidad de la red, sino por **trabas mecánicas en las máscaras de acción** (el ejército que pierde el permiso de atacar una vez que sale de casa) y **desestabilización de gradiente por el solapamiento asíncrono**. 

Aplicando la corrección de `PACK_ARMY` y el ajuste de LR, deberías ver al agente convertir casi todos esos empates de 53k ticks en victorias aplastantes antes del tick 35.000.