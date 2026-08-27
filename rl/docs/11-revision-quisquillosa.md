# Contra-benchmark — Revisión quisquillosa vs "AlphaStar-lite" (2026-08-27)

> **Fuente:** segunda revisión externa, auditor quisquilloso, sobre `network.py`, `trainer.py`, `reward_shaping.py`, `auto_support.py` y `metrics.jsonl` actuales. Se guarda como contrapeso de `10-benchmark-alphastar-openai-five.md` (reseña 1).
> **Tesis del revisor:** "No. No estás en nivel AlphaStar ni OpenAI Five. La reseña acierta en el linaje y se equivoca al tratar ese linaje como equivalencia."

---

## Frase inflada (según el revisor)

> «la matemática, los grafos de cómputo y la estructura de tu red son conceptualmente equivalentes» — **No es cierto.** Compartís familia (obs multimodal + política factorizada + núcleo recurrente + máscaras). No compartís grafo, capacidad, régimen ni, sobre todo, resultado.

---

## Qué sí es verdad (el revisor concede 3 aciertos de la reseña 1)

1. **Obs multimodal.** Mapa `9×H×W` (U-Net lite + CoordConv), set de unidades (MLP 10-D + media enmascarada, `MAX_UNITS=48`) y 21 escalares. Tesis correcta de un RTS: ni Atari-pixels ni vector plano.
2. **Política factorizada con máscaras.** `P(tipo) · P(unidad|tipo) · P(celda|tipo) · P(ítem|tipo)` con `train_slot_mask / build_slot_mask` es el mismo tipo de muestreo encadenado. El `log_prob` solo suma cabezas activas (F3). Eso es serio.
3. **`auto_support.py`.** Reparar/apagar energía fuera del presupuesto de PPO es la misma filosofía que el auto-courier/buyback de OpenAI Five. Con 1 decisión / 80 ticks, es obligatorio.

Las 3 brechas que cita la reseña 1 (self-play, imitación, annealing del shaping) también son reales. Ya estaban en el roadmap como Fase 4, no como estado actual.

---

## Dónde la tabla de la reseña 1 miente o infla

| Celda de la reseña | Realidad en tu código |
|---|---|
| «Tipo → Slot → Celda → Rol» siempre | Las 4 cabezas no forman una cadena completa. `dist_cell` no ve la unidad elegida (limitación documentada en `revision-externa-rl.md`). Un MCV y un rifle reciben el mismo mapa de destinos. AlphaStar condiciona el pointer al sujeto. |
| «MLP + Masked Average Pooling» ≈ Transformer de AlphaStar | No. Media enmascarada de 48 slots × 10 features no atiende. No hay relaciones unidad-unidad (quién cubre a quién, focus fire, surround). OpenAI Five tampoco usó Set Transformers: era MLP por entidad + pooling hacia un LSTM 4096. Esa fila está mal en los tres lados. |
| U-Net lite ≈ ResNet + Scatter de AlphaStar | Tu U-Net es 2 niveles, 96 canales, ~2.8M params. AlphaStar proyecta entidades sobre el mapa (scatter) y después hace ResNet. Vos tenés densidad en Ch6/Ch8, no identidad de unidad en el tensor espacial. |
| GRU 416 ≈ LSTM 4096 / Deep LSTM | Misma idea (POMDP), dos órdenes de magnitud menos de estado. BPTT 32 es razonable; no es el núcleo de Five. |
| «OpenAI-5» | Se llama OpenAI Five. Detalle chico, síntoma de reseña genérica. |
| Cómputo «3 workers, ~10² partidas/h» | Dirección correcta. Five: ~180 años de juego por día. AlphaStar: meses de TPU. La diferencia no es 10×; es 10⁴–10⁶×. |

> «AlphaStar-lite / OpenAI-5-lite real y riguroso» es un buen nombre de arquitectura (`AlphaLiteNet`). Como veredicto de nivel, es marketing.

---

## Las brechas que la reseña 1 no pone (y importan más)

### 1. El resultado, no el diagrama

Últimas iters de `ckpts/metrics.jsonl`: `winrate 0, raze 0, win 0`. Outcomes: `lose / incomplete`. El reward positivo lo carga `garrison (~+3.1 a +3.4)`, no la victoria. Run2 ya lo diagnosticó: coloso pacifista. Run3 (`eradicate_v4`) todavía no remata.

AlphaStar y Five se miden por ganar contra humanos de élite. Vos todavía no ganás contra el `beginner` de `ai.yaml`, que está capado a propósito (1 harvester, sin repair, rush delay enorme, solo infantería).

### 2. El rival no es comparable

Ganar al `hard` de OpenRA no es «lo que hace pocos años costaba millones». Ese bot es un script de campaña (`HarvesterBotModule + SquadManager`). Equivale más a vencer la IA integrada de SC2 Very Hard — cosa que bots amateur hacían años antes de AlphaStar — que a vencer a MaNa, TLO u OG.

El salto caro no es `beginner → hard`. Es `hard → humano → self-play` que no colapsa.

### 3. Ancho de banda de acción

1 decisión cada 80 ticks (~3 s de sim) es otro problema. AlphaStar actúa a APM humano (~22 efectivos) con cabeza de delay y cámara. Five observa a ~30 Hz. Tu `army_attack_move + auto_support` son parches correctos a ese cuello; no lo cierran.

### 4. El shaping no está «en camino a +1/−1»

`eradicate_v4` paga refinería, first ore, mining, garrison, naked base, `raze×2`, `defense_first −3`, `hold_zero`, `cancel`, `timeout`, margen, `win +8 / lose −2.5`. Eso es un programa de comportamiento, no un juego de suma cero. En el log, `garrison` ya es el término que manda. Five y AlphaStar annealearon shaping después de ser fuertes. Vos lo necesitás para que el agente no se muera de hambre; el riesgo es que aprenda a maximizar garrison y truncar.

### 5. Curriculum con trampa pedagógica

`BEACON_BY_MAP` enciende `Ch7/Ch8` sobre la base enemiga sin abrir niebla. Está bien para Escenario A. No es el POMDP de AlphaStar. Un agente que «ve el beacon» no ha resuelto scouting.

### 6. Un mapa, un matchup, tabula rasa, PPO síncrono

Sin liga, sin PFSP, sin BC, sin scatter, sin pointers, sin V-trace/UPGO. El propio `roadmap-agente.md` pone self-play en Fase 4, después de winrate vs beginner. Eso es el orden correcto. La reseña 1 lo vende como si el motor ya fuera el de DeepMind y solo faltara la granja.

---

## Veredicto recalibrado (del revisor)

| Capa | Dónde estás |
|---|---|
| **Diseño** | Agente RTS compacto inspirado en AlphaStar/Five, bien ejecutado para 1 GPU. Familia correcta, no equivalencia. |
| **Ingeniería de pipeline** | Por encima del prototipo típico: máscaras jerárquicas, BPTT real, coerción honesta, vocab persistente, shaper con exploits ya cazados. Eso sí es rigor. |
| **Régimen** | Laboratorio. PPO síncrono, bot scripted, shaping denso, 1 mapa. |
| **Capacidad** | ~2.8M params, GRU 416, pool de unidades. Lite de verdad, no lite de marketing. |
| **Juego** | Aún no cierra partidas. Economía a veces viva, remate ofensivo no. |

> La analogía honesta: tenés el chasis de un paper 2019 recortado a hardware de consumo. No tenés el motor de entrenamiento que convirtió ese chasis en un campeón.
> Si el pipeline llega a winrate sostenido vs beginner con `raze>0`, eso es un hito real de laboratorio (el que el Run3 declara). Si después sube `easy → hard` sin colapsar a un exploit del script, es un resultado publicable de hobby/research. Ninguno de los dos es AlphaStar.

---

## Qué más te acerca de verdad, en este hardware (orden del revisor)

1. Cerrar el juego vs `beginner` (`raze + win`, no `garrison`).
2. Currículum de rivales (`easy/hard`) para que el óptimo local de 8 rifles se rompa.
3. Self-play casero (clonar `latest.pt`) cuando ya gane a un bot que se defiende.
4. Recién entonces apagar shaping hacia `win/lose`.

> La reseña 1 sirve como mapa de familia. No la uses como termómetro de nivel.

---

*Guardado: 2026-08-27 — rama `exp/rl-2026-08-27-scalar19`. Complementa a `10-benchmark-alphastar-openai-five.md`.*
