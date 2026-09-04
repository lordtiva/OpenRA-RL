Los nuevos datos de **`metrics2.txt`** (72 iteraciones) y **`live_games2.txt`** revelan el estado exacto del agente tras los cambios:

### El balance inicial:
1. **El cambio de `PACK_ARMY` funcionó inicialmente:** En las primeras 30 iteraciones el winrate contra `easy` subió hasta un pico de **24.6%** (Iter 29), logrando victorias contundentes cuando decide atacar en masa (ej. Iter 10 ganando en 24k ticks con 111 unidades; Iter 13 ganando con 152 unidades).
2. **Sin embargo, el aprendizaje volvió a colapsar:** A partir de la iteración 35, el winrate se desplomó de nuevo hacia el **16% - 18%**, y contra `medium` sigues en **0% absoluto** (0 victorias en 93 partidas).

Al cruzar los registros de las partidas con las métricas de gradiente y el detalle clave que mencionas (**"el mapa tiene un gran lago central que separa a ambos jugadores"**), emergen **cuatro causas fundamentales** de por qué el agente sigue estancado.

---

# 1. Causa 1: El "Reward-Hacking" del Turtling (La Trampa Matemática)

Revisando el desglose de recompensas en `metrics2.txt` de las partidas que terminan en derrota o timeout de 53k ticks:

* **Penalización por salir y perder:** `defense_loss` ($-3.8$ a $-4.4$) + `margin` ($-2.5$) = **$-6.5$ a $-7.0$ puntos de penalización**.
* **Recompensa por quedarse en casa sin hacer nada:**
  * `garrison` (bono por tener tropas cerca de casa): **$+4.0$ a $+6.5$**
  * `mining` (el camión que sigue entregando mineral): **$+1.5$ a $+2.0$**
  * `timeout` (penalización por agotarse el tiempo a los 53k ticks): **solo $-0.25$ o $-0.50$**

### La paradoja matemática:
Si el agente junta tropas, sale de base, bordea el lago y muere en el cuello de botella, el retorno del episodio es **$-4.0$ a $-6.0$**.
Si el agente se queda en su base 53.000 ticks acumulando infantería en guardia pasiva (`live_games2.txt`, Iter 10: 40.000 ticks quieto en `[7.8, 20.4]` con 23 soldados), el retorno neto del episodio es:
$$\text{Reward} = +5.5 \text{ (garrison)} + 1.8 \text{ (mining)} - 0.5 \text{ (timeout)} = \mathbf{+6.8}$$

**El agente descubrió que jugar al empate pasivo le da más recompensa que arriesgarse a atacar y perder.** Por eso en `metrics2.txt` ves episodios incompletos donde el agente obtiene $+10$ o $+14$ de reward sin haber disparado un solo tiro al rival.

---

# 2. Causa 2: La Geografía del Lago y la "Fila India" de Infantería

El dato del **lago central** explica las derrotas fulminantes del agente cuando decide salir (como en la Iter 4, donde pierde en 14k ticks teniendo 42 soldados):

1. **La cabeza de celda tira hacia el spawn enemigo:** La red emite `army_attack_move` apuntando directamente hacia el beacon `[95, 11]`.
2. **El buscador de caminos (A*) de OpenRA bordea el lago:** Al toparse con el agua, el motor desvía a las tropas por los estrechos pasos de tierra del norte o del sur ($y \approx 6..10$ o $y \approx 28..34$).
3. **El efecto embudo (Ley de Lanchester):**
   * El agente compone su ejército de **pura infantería lenta (`e1`, `e3`, `medi`)**. No construye vehículos ni aviación (`hist` muestra casi 0 tanques).
   * Al bordear el lago, los 60 soldados no avanzan en formación cerrada, sino en una **larga fila india de 1 o 2 unidades de ancho**.
   * El bot `easy` o `medium`, que espera atrincherado al otro lado del lago con defensas y unidades agrupadas, aniquila a tus soldados de dos en dos conforme van saliendo del cuello de botella.
   * En `live_games2.txt` (Iter 4, dec 200 $\to$ 250): el ejército pasa de 43 soldados a 28, luego a 18 y es barrido completamente sin llegar a tocar la base rival.

---

# 3. Causa 3: Monopolio Absurdo de la Acción `harvest`

Observa los histogramas de acción (`action_hist`) en `metrics2.txt`:
* **Iter 45:** `harvest: 1217`, `no_op: 471`, `set_stance: 346`, `train: 488`, `army_attack_move: 23`.
* **Iter 55:** `harvest: 1009`, `train: 287`, `set_stance: 98`.
* **Iter 61:** ¡`harvest: 1414`!
* **Iter 65:** `harvest: 1218`.

Entre el **40% y el 60% de todas las decisiones que toma la red neuronal son `harvest`**.
¿Por qué? Porque la recolección automática genera ganancias continuas y la red aprendió a "spamear" la tecla de cosechar para asegurar gradiente positivo sin peligro. Esto **asfixia el ancho de banda del agente**: de 1.000 decisiones por partida, gasta 800 en ordenar a los camiones que hagan lo que ya hacen solos, en lugar de emitir comandos militares o macro-estructurales.

---

# 4. Causa 4: La Inestabilidad PPO Sigue Presente (`clip_frac` y `grad_norm`)

Bajar el Learning Rate de $1.5 \times 10^{-4}$ a $1.0 \times 10^{-4}$ ayudó ligeramente, pero **no resolvió la desestabilización**:

* **Picos de `clip_frac`:**
  * Iter 7: **38.4%**
  * Iter 22: **40.6%**
  * Iter 29: **40.8%**
  * Iter 52: **42.8%**
* **Explosiones de gradiente (`grad_norm`):**
  * Iter 2: `11.17`
  * Iter 18: **`20.85`** (shock masivo)
  * Iter 48: **`13.61`**
  * Iter 67: **`12.27`**
* **Colapso de Entropía:** En las iteraciones 14, 36, 59 y 70, la entropía cayó por debajo de **$1.0$** (política híper-determinista que solo spamea `no_op` y `harvest`).

El solapamiento asíncrono ($k$ vs $k+1$) con 2 épocas completas de PPO sigue empujando a la política fuera de la región de confianza.

---

# 5. Plan de Acción Quirúrgico para Desbloquear el Run

Para quebrar el estancamiento y superar a `easy` y `medium`, aplica estos cuatro ajustes concretos:

---

### Paso 1: Eliminar el "Incentivo al Turtling" (Alinear Recompensas)
Debes hacer que el empate/timeout sea estrictamente peor que perder intentando atacar.
En `rl/reward_shaping.py` (preset `eradicate_v4`):

1. **Aumentar drásticamente el castigo por timeout:**
   ```python
   # Si la partida dura 53.000 ticks y no se resolvió, es un fracaso táctico.
   w_timeout = 6.0  # (Antes 1.0; ahora neutraliza por completo el bono acumulado de garrison)
   ```
2. **Capar `w_garrison` en el tiempo:**
   El bono por tener tropas en base solo debe pagar en los primeros 15.000 ticks (fase de apertura). Pasado ese tiempo, defender sin salir debe pagar $0$:
   ```python
   if obs.tick > 15000:
       r_defense_posture = 0.0  # Se acabó la fase de tortuga: a pelear
   ```

---

### Paso 2: Limitar el Spam de `harvest` en la Política
En OpenRA, los camiones cosechan solos; el comando manual es solo para redirigir.
En `rl/action_adapter.py`, evita que la política pueda spamear `harvest` en bucle si los camiones ya están cosechando activamente:
```python
# Si no hay cosechadoras ociosas (todas están trabajando), enmascarar 'harvest'
# para que la red no gaste decisiones redundantes en ello.
idle_harvesters = any(
    "harv" in getattr(u, "type", "").lower() and getattr(u, "is_idle", False)
    for u in obs.units
)
if not idle_harvesters:
    m[TYPE_TO_IDX["harvest"]] = False
```
*Efecto:* Liberarás entre 400 y 800 decisiones por partida que la red se verá **obligada** a utilizar en `army_attack_move`, `train` o `build`.

---

### Paso 3: Control del Cuello de Botella del Lago (Waypoints Tácticos)
Dado que hay un lago central, una orden directa a `[95, 11]` mete a las tropas en la fila india de la muerte.
En `rl/action_adapter.py`, cuando la celda objetivo de un `army_attack_move` caiga en agua o intente cruzar el centro del mapa a ciegas, asegúrate de que el snap passable elija el pasillo de tierra con mayor anchura (generalmente el flanco sur $y \approx 30$ o norte $y \approx 8$), o incentiva a la red a desbloquear la fábrica de armas (`weap`) para sacar tanques que resistan el paso del cuello de botella.

---

### Paso 4: Frenar Definitivamente el `clip_frac` (Ajuste de Hiperparámetros)
Para evitar que el 40% de las muestras se recorten:
1. En `rl/train.py` / `TRAIN_ARGS`:
   * Cambia `--epochs 2` a **`--epochs 1`**.
   * *Por qué:* Como tus rollouts ya vienen con 1 iteración de retraso por el solapamiento (`infer_net`), hacer 2 épocas sobre datos desfasados empuja el ratio $\pi / \pi_{\text{old}}$ a extremos grotescos. Con 1 época, el gradiente será limpio y el ratio se mantendrá dentro de $[0.85, 1.15]$.
2. Reduce `clip_eps` en `trainer.py` a **`0.15`** y sube el `max_grad_norm` a **`1.0`** (evitando picos de 20).
