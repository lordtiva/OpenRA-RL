# Plan 4 capas — Siguiente nivel en tu hardware (2070 + 5600X + 32GB)

> **Fuente:** tercera revisión del revisor quisquilloso (2026-08-27), respuesta a "¿cuál es mi siguiente jugada con mi hardware?". Guardado como roadmap de 6-12 meses. Complementa a `10-benchmark-alphastar-openai-five.md` (mapa de familia) y `11-revision-quisquillosa.md` (termómetro).
> **Tesis:** el cuello no es el algoritmo (PPO ya hace su trabajo: maximiza `garrison`), son **muestras, crédito y que el juego se pueda ganar**. En 1 GPU el stack que rinde es `PPO + BC + self-imitation + self-play chico`.

---

## Hardware honesto

| Recurso | Lo que tenés | Presupuesto honesto |
|---|---|---|
| **GPU RTX 2070 8GB** | Red 2.8M usa fracción | Cabe red ~15-20M con BPTT 32. No cabe DreamerV3 reconstruyendo mapas + imaginación + política. |
| **CPU 5600X 6c/12t** | OpenRA es el cuello | 3 contenedores con límites `9+4+4 CPUs` es oversubscribe. Óptimo: **2 daemons**, 1-2 núcleos para trainer/OS. |
| **RAM 32GB** | 3×6GB Docker + Windows + WSL + PyTorch te deja sin aire | 2 daemons × 4-5GB. |

La 2070 no es el problema. El 5600X y los 32GB sí: generás `~10² partidas/h`, no `10⁵`.

---

## Lo que NO haría (aunque esté de moda)

*   **DreamerV3 / TD-MPC / MuZero.** Mejor sample-efficiency en Atari/dmc. En tu espacio (mapa 2D + 48 entidades + 4 cabezas + horizonte 600) es rewrite de 2-3 meses, inestable, y 8GB se quedan cortos.
*   **Transformer gordo, Mamba, MoE, 50M params.** Con cientos de partidas/día, más capacidad = más sobreajuste al `beginner` y al `beacon`.
*   **IMPALA/V-trace a lo AlphaStar.** Quiere decenas de actores. Con 6 núcleos perdés el motivo.
*   **Apagar shaping a `+1/−1` ahora.** Último paso, no siguiente. Hoy `win=0`.
*   **Liga de 12 agentes.** No caben en RAM ni CPU.
*   **Tirar PPO.** Sigue siendo correcto en 1 GPU. Five usaba PPO. El salto de Five no fue el optimizer.

El stack 2025-26 que sí cabe: **PPO + clonado (BC) + self-imitation + self-play chico.**

---

## Capa 0 — Hacer que ganar sea posible (1-2 semanas, software)

> La sonda de horizonte demostró el hecho más importante: un rush de 8 rifles no declara victoria en 51k ticks. El `beginner` reconstruye. Sin masa no hay `win`. Sin `win`, no hay AlphaStar-lite.

Tres cambios de **entorno**, no de red:

1.  **Opción de asalto sostenido, no más micro.** `army_attack_move` ya existe y casi no se usa. Tratarlo como `auto_support`: si `military_ratio` alto y hay `beacon`, el entorno mantiene un push (la red elige *cuándo* entrar en modo asalto; el adapter mueve ociosos cada bloque). Cierra la brecha APM que un GRU 416 no va a aprender.
2.  **Declaración de partida.** Si el engine no cierra con conyard enemigo abajo, es bug de MDP, no de PPO. Del lado C# / bridge: rendición del bot o `win` cuando no queda producción enemiga N ticks. Si el label `win` nunca existe, `w_win=8` es código muerto (ya te pasó con margen truncable).
3.  **Horizonte vs γ.** `624 steps × 80 ticks` con `incomplete` dominante. Subir `γ 0.995-0.997` (ya anotado) y cortar episodios que van ganando por `raze` en cuanto el engine pueda declarar. Cada victoria temprana es throughput gratis.

**Métrica:** `raze>0` en rolling 20, después `winrate>0`. Si en 50-80 iters de `eradicate_v4` sigue `raze=0`, `garrison ~+3.4` sigue siendo el trabajo más seguro → bajar `w_garrison / w_mining_rate` un escalón, no subir `w_raze` a 5 (reintroduce zerg).

**Regla:** no toques la red aquí. Un cambio de régimen por vez.

---

## Capa 1 — Bootstrap por imitación (el secreto real de AlphaStar, y te cabe)

Tabula rasa en RTS con `10² partidas/h` es masoquismo. AlphaStar no descubrió "hay que construir refinería": lo clonó.

Fuentes baratas, en orden:

**A. Clonar tu propio `examples/scripted_bot.py` (días, no meses).** Correrlo contra el mismo escenario, grabar `(obs → ActionIndex)` y hacer behavioral cloning sobre las 4 cabezas con las mismas máscaras. Loss = `CE(tipo) + CE(unidad) + CE(celda) + CE(ítem)` solo en cabezas activas. 20-50k decisiones de un bot que ya hace `proc → harv → barracks → push` te dejan un agente que no nace en 0.

**B. Grabar órdenes del bot C# hard (cambio de engine, alto ROI).** En `BotModules` loguear `IOrder` / comandos al mismo esquema que `CommandModel`. Es el experto de `ai.yaml`. Un finde de C# te ahorra un mes de PPO ciego.

**C. Replays `.orarep` humanos (opcional, más tarde).** OpenRA los tiene. Parsearlos es más sucio (cámara, grupos, APM). Dejarlo para cuando A/B funcionen.

**Entrenamiento híbrido, no "BC y después RL" rígido:**

```
L = L_PPO + λ_bc · L_BC + λ_sil · L_SIL
```

*   `λ_bc` arranca en `1.0` y baja a `0` en ~50-100 iters (kickstarting, receta OpenAI).
*   **SIL (self-imitation):** cuando un episodio tenga `raze>0` o `win`, meterlo en buffer de élite y clonar esas transiciones. En 8GB es un replay de 2-4k steps. El truco de sample-efficiency que más rinde sin world model.
*   **PPG (Phasic Policy Gradient):** 1-2 epochs de política + 4-6 de crítico. El crítico de RTS está crónico (te pasó el pesimismo `-2.5`). PPG es de 2020, cabe en la 2070, y es más útil que cambiar a SAC.

Esto sí es "algoritmo actual" adaptado a tu máquina. No es un paper nuevo; es el stack que sobrevive sin 1024 GPUs.

---

## Capa 2 — Red: un salto 2018→2019, no a 2026 (1 PR, ~3-4M params extra)

La 2070 aguanta esto sin sudar. No agrandes canales a lo loco; cambiá el sesgo inductivo.

| Pieza | Qué | Por qué | VRAM |
|---|---|---|---|
| **Transformer de entidades** (2 capas, 4 heads, d=64, 48 slots) | Reemplaza la media enmascarada | Focus fire, harv vs tanque, quién está herido. 48 tokens es ridículamente barato. | ~+2-4 ms/step |
| **Scatter** | Pintar cada unidad (equipo, HP, rol) en el fmap antes de `cell_head` | AlphaStar de verdad. Hoy Ch6/Ch8 son densidad anónima; la cabeza de celda no sabe qué hay en (x,y). | casi 0 |
| **Condicionar `dist_cell` a la unidad** | Concat del embedding del slot elegido | Ya diagnosticado en `11-revision-quisquillosa.md`. Sin esto el autoregresivo es de mentira. | 0 |
| **GRU 416 → 512** | Opcional, al final | Solo si el transformer ya está y el crítico sigue ciego a planes de 30+ steps. | irrelevante |

No pongas un ViT sobre el mapa. La U-Net lite de RF ~40 celdas está bien para mapas de RA. El agujero es el set de unidades, no el terreno.

Tamaño objetivo: `~6-8M` params, no 50M. Net2Net si querés conservar pesos; si venís de BC fresco, da igual.

`FP16 (torch.cuda.amp)` en el update: gratis, Turing lo soporta. No va a 2×, pero el backward del BPTT 32 con spatial sí se nota.

---

## Capa 3 — Sparring que se adapta (cuando `winrate>30%` vs `beginner`)

Hasta que no ganes al script, self-play es teatro: dos idiotas se enseñan el mismo vicio.

Cuando haya wins:

1.  Currículum de bots `beginner → easy → hard`, promoción por `winrate` rolling, no por reward. Ya está diseñado. Ejecutarlo.
2.  Self-play de 3 checkpoints (PFSP pobre): `50% vs bot, 50% vs latest / iter-20 / best_winrate`. Prioridad al que más te gana.
3.  El cambio de software gordo: sesiones **RL vs RL** en el bridge. Hoy el slot 2 es `ai.yaml`. Sin dos políticas en el mismo `World`, no hay liga. Eso es más "siguiente nivel de proyecto" que cualquier paper. CPU: una partida self-play ≈ `1.2×` una vs bot (segundo forward en GPU, barato; la sim sigue mandando).

No hace falta una league de exploiters. Con 3 snapshots y PFSP ya aparece el ciclo arma-contra-arma. Eso es el 80% de AlphaStar League a escala de sótano.

**Annealing del shaping:** solo cuando `winrate vs hard >50%` estable. Bajá `w_garrison`, `w_mining_rate`, `w_refinery` primero; dejá `w_raze` y `w_win/w_lose` hasta el final. Nunca apagues todo de golpe.

---

## Throughput: el único "algoritmo de sistemas" que te cambia la vida

El docstring de `train.py` ya confiesa que el overlap APPO nunca existió: el update bloquea el event loop.

En un 5600X esto vale más que un Transformer:

1.  **2 daemons, no 3.** Menos RAM, menos context-switch, más ticks/s reales.
2.  **Colecta async de verdad.** Proceso (o hilo) de rollout en CPU/GPU-inferencia mientras el trainer hace update. Inferencia es ~1 ms; el 95% del wall-clock es OpenRA. Solapá `update (~40-90s hoy)` con la siguiente tanda.
3.  **Capar WSL/Docker a ~16GB** para que Windows no swapee. El swap mata más el run que un batch chico.
4.  **No subas concurrency** por encima de sesiones que el daemon mueve a ~1 núcleo cada una. 4-6 sesiones totales en 2 procesos, no 12 en 1.

Estimación realista: de `~100 partidas/h → 150-200` con overlap + 2 daemons bien alimentados. No a 10.000. Por eso BC/SIL importan más que el paralelismo.

---

## Orden de ejecución (un régimen por vez)

| Cuándo | Qué | Criterio de promoción |
|---|---|---|
| **Ahora** | **Capa 0:** declaración de `win` + asalto sostenido + bajar `garrison` si `raze` sigue 0 | `raze>0` y `winrate>0` vs `beginner` |
| **2-3 sem** | **Capa 1:** BC del `scripted_bot` + PPO híbrido + SIL de episodios élite | El agente reproduce `proc→army→push` sin farmear `garrison` |
| **1 sem** | **Capa 2:** transformer entidades + scatter + `cell|unidad` (un solo PR de red) | smoke + 20 iters sin colapso de entropía |
| **Cuando gane** | **Capa 3:** `easy/hard`, después RL-vs-RL con 3 ckpts | `winrate>30%` vs `beginner` |
| **Nunca** | Dreamer, liga de 12, `+1/−1` prematuro, red 50M | — |

---

## Cómo se ve "el siguiente nivel" en tu hardware, no en el de DeepMind

Un proyecto de sótano de nivel investigación, 6-12 meses de wall-clock:

*   Gana al `hard` de OpenRA de forma estable (no un lucky `raze`).
*   Self-play 1v1 con 3 snapshots, sin colapsar a un único build.
*   Red ~8M con atención a unidades y scatter; PPO+BC+SIL.
*   Shaping ya en retirada, no en el asiento del piloto.
*   `~10⁷-10⁸` ticks de experiencia, no `10¹²`.

Eso no es AlphaStar. Es un agente RTS moderno, reproducible, en una 2070, con las ideas que realmente importaban de 2019 (imitación + self-play + acciones factorizadas) y sin la granja. Publicable como hobby/research serio. No como "hemos igualado a DeepMind".

> **Si hay que elegir una cosa esta semana:** no toques la red. Hacé que el MDP pueda emitir `win` y que un push de ejército se sostenga entre decisiones. Todo lo demás, incluido el Transformer, espera a que `raze` deje de ser 0.

---

*Guardado: 2026-08-27 — rama `exp/rl-2026-08-27-scalar19`. Roadmap operativo para después del Run3 `eradicate_v4`.*
