# Sonda de horizonte — ¿cuánto tarda un resultado declarado en escenario A?

**Fecha**: 2026-08-24 · **Motivación**: plateau detectado (reward clavado +2.0/+2.7
desde iter ~111; 3948 episodios, 0 victorias, 100% `incomplete@16952t`) con
supremacía material creciente (+1756 → +2233, P(win) est 0.72). El agente domina
pero jamás convierte en victoria. Hipótesis del operador: el límite de tiempo
(104 decisiones × 160 ticks = 16.640 ticks) hace imposible que el engine declare.

## Método

`rl/probe_horizonte.py` — dos partidas deterministas SIN aprendizaje contra el
mismo mapa (`fase2_a.oramap`, mismo `map_name` estable que el trainer: cero
reescrituras por hash-compare), cap generoso de 320 decisiones (51.200 ticks,
3× el horizonte de entrenamiento):

| Guion | Acción | Qué acota |
|---|---|---|
| `rush` | `army_attack_move` hacia spawn del bot (95,11) CADA decisión desde t0 | mejor caso agresivo-ingenuo de VICTORIA |
| `pasiva` | NO_OP siempre | peor caso pasivo de DERROTA |

## Resultados medidos

| Métrica | Rush | Pasiva |
|---|---|---|
| Resultado | `incomplete` @ 51.216t | `incomplete` @ 51.232t |
| Primer contacto | t2635 (enemy_spotted) | **NUNCA** |
| Valor militar propio | 10400 → 5800 (t11412) → **PLANO hasta el final** | 10400 **intacto TODO el juego** |
| Wall-clock | 24.8s | 24.6s |

## Hallazgos

1. **El rush puro se estanca**: llega, pelea (pierde ~$4.600 de valor), y queda
   congelado en un empate militar que ninguno de los dos cierra. 8 rifles no
   quebran una base que se reconstruye (+762 $/kt de recaudación rival medida
   en entrenamiento).
2. **El bot nunca ataca** en 51.200 ticks de pasividad total. En este escenario
   no existe presión de derrota: el único resultado alcanzable es que NOSOTROS
   ganemos.
3. **Ninguna declaración en 3× el horizonte actual** bajo juego ingenuo. La
   pregunta del operador queda respondida CON DATOS: no es que falte un poco
   de tiempo — el tiempo-declaración bajo juego torpe excede cualquier
   escalera corta (104→156 no alcanzaría tampoco).

## Lectura para el diseño del régimen 2-B

- Un humano principiante le gana al beginner de RA en ~10–15 min de juego
  (~15.000–22.000 ticks ≈ 100–140 decisiones) **porque produce y masifica**,
  no porque rushée 8 unidades. El rush falló por composición, no por concepto:
  la vía a la victoria es construir ejército durante más decisiones y empujar.
- Horizonte 208 decisiones (33.280 ticks ≈ 22 min) da ese margen con holgura.
- Propiedad clave del costo: episodios que no terminan pagan ~55s (vs 27s),
  pero **cada victoria temprana recorta su propio episodio**. El costo extra se
  concentra exactamente donde falta señal y desaparece a medida que sube el
  winrate. No hay barrido de incrementos: un salto calibrado por esta sonda
  (50s de sonda vs horas de GPU).
- γ: a 208 pasos, 0.99^208 ≈ 0.12 (crédito terminal desvanecido) → subir a
  0.995 (0.995^208 ≈ 0.35), como ya estaba pactado para horizontes >104.

## Opciones (decisión del operador pendiente)

- **A. Régimen 2-B**: horizonte 208 dec + γ 0.995 + bonus terminal win +8 /
  loss −4 / truncamiento 0. Completa la prescripción del especialista ("el
  gradiente tiene que poder tocar `result=='win'`").
- **B. Solo horizonte, sin bonus**: extiende a 208 pero el reward sigue sin
  distinguir ganar de truncar → probablemente reproduce el plateau con
  episodios más caros. Descartado salvo como control experimental.
- **C. Cirugía de escenario (Plan B)**: base pre-construida ESTÁTICA para el
  bot + regla shortgame (destruir edificios = victoria). Declaraciones mucho
  más rápidas, pero cambia la semántica del objetivo final (enemigo real se
  expande). Reservado si A no produce victorias en ~2500 eps.

## Nota operativa

Correr sondas en paralelo con las 12 sesiones del trainer puede generar errores
transitorios de engine (RPC UNKNOWN) en las sesiones de entrenamiento — el
rollout los absorbe (umbral 5 consecutivos) y ninguna iteración se perdió
(iter 330 verificada limpia post-sonda). Preferir :8020 o aceptar el ruido.
