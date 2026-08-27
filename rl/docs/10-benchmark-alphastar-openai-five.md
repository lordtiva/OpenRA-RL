# Benchmark — Dónde está tu agente vs AlphaStar y OpenAI Five

> **Fuente:** auditoría externa solicitada el 2026-08-27 sobre el full-stack Run3 (SCALAR 21 + auto_support + eradicate_v4). Transcripción fiel de la reseña del auditor, guardada como referencia de posicionamiento y roadmap futuro.
> **Estado al momento de la reseña:** Run2 coloso pacifista cerrado (`08-avance-run2.md`), Run3 full-stack listo (`09-fullstack-run3.md`, `SCALAR 21`, `eradicate_v4`).

---

## Resumen del auditor

> La respuesta honesta y técnica tiene dos caras:
> **A nivel de diseño conceptual y arquitectura: Sí, has implementado un "AlphaStar-lite / OpenAI-5-lite" real y riguroso.**
> **A nivel de régimen de entrenamiento y escala: Estás en la fase de laboratorio/prototipo individual.**

---

## Comparativa: Tu Agente vs OpenAI Five vs AlphaStar

| Dimensión | Tu Agente (`AlphaLiteNet`) | OpenAI Five (*Dota 2*, 2019) [2] | AlphaStar (*StarCraft II*, 2019) [1] |
| :--- | :--- | :--- | :--- |
| **Entrada Espacial** | U-Net Lite + CoordConv ($9\times H\times W$) | No usa mapa 2D (solo vectores de entidades) | ResNet espacial + Scatter ($2D \to$ espacio latente) |
| **Entrada de Unidades** | MLP por slot + Masked Average Pooling | Set Transformers / Entity embeddings | Multi-Head Attention (Transformer) sobre unidades |
| **Memoria Temporal** | GRUCell ($416$ dim) con BPTT truncado (32 steps) | LSTM ($4096$ dim) con BPTT ($16$ steps) | Deep LSTM ($1024$ dim) con BPTT ($64$ steps) |
| **Espacio de Acción** | Autorregresivo condicional (Tipo $\to$ Slot $\to$ Celda $\to$ Rol) | Autorregresivo por argumentos con máscaras | Autorregresivo con Pointer Networks |
| **Micro vs. Macro** | Autonomía de soporte (`auto_support.py`) | Automatizaciones de soporte (auto-courier, buyback) | Pseudo-cámara con límites de APM humanos |
| **Algoritmo** | PPO síncrono + GAE + Huber loss | Rapid PPO asíncrono distribuido | Actor-Critic off-policy (V-trace + UPGO + TD($\lambda$)) |
| **Inicialización** | *Tabula Rasa* (desde cero) | *Tabula Rasa* con cirugías de red | **Imitation Learning** (500k replays de humanos Grandmaster) |
| **Sparring / Oponentes** | Bots Scripted (Currículum `beginner` $\to$ `hard`) | **Self-Play masivo** (jugar contra copias de sí mismo) | **AlphaStar League** (Main Agents, Exploiters, League Exploiters) |
| **Poder de Cómputo** | 1 GPU de consumo, 3 workers Docker (~$10^2$ partidas/h) | 128,000 CPU cores + 1,024 GPUs (~$180$ años de juego por día) | Cientos de Google TPUs v3 durante meses |

---

## Lo que tu proyecto YA TIENE de AlphaStar y OpenAI Five

1. **La Representación Multimodal de Estado:**
   Al igual que DeepMind y OpenAI, entendiste que un RTS no se puede jugar solo con visión (píxeles puros estilo Atari) ni solo con números. Tu red combina tensores espaciales ($H\times W$), sets de entidades móviles (unidades) y escalares económicos.

2. **Espacio de Acción Autorregresivo con Máscaras Jerárquicas:**
   Tu cabeza de ítems filtrada por `train_slot_mask` y `build_slot_mask`, condicionada al tipo de acción elegido en la cabeza 1, es exactamente la arquitectura de muestreo encadenado $P(a) = P(\text{tipo}) \cdot P(\text{target}|\text{tipo}) \dots$ que usan ambos papers [1, 2].

3. **Capa de Confort / Autonomía:**
   OpenAI Five no obligaba a la red neuronal a aprender a comprar el burro de transporte (*courier*) ni a micro-gestionar el inventario básico [2]. Tu `auto_support.py` aplica exactamente la misma filosofía: el RL se reserva para la estrategia macro.

---

## Las 3 Grandes Brechas que te separan de ellos

### 1. El Régimen de Sparring: De Bots a Self-Play League

* **Lo que tienes hoy:** El agente juega contra un bot con reglas fijas (`ai.yaml`). Una vez que el agente aprende los puntos débiles del bot (ej. cómo flanquearlo), el bot no se adapta.
* **Lo que hicieron AlphaStar y OpenAI Five:**
  Entrenan en **Self-Play**. El agente juega contra versiones congeladas de sí mismo del pasado. Si el agente descubre una táctica rota, su "yo rival" aprende a defenderse de ella en la siguiente iteración. Así emergen estrategias complejas sin hardcodear recompensas [1, 2].

### 2. El Bootstrap: De *Tabula Rasa* a Imitación

* **Lo que tienes hoy:** La red nace sabiendo $0$ y tiene que descubrir por prueba y error que construir una refinería es bueno. Por eso requieres *Reward Shaping* denso (`eradicate_v4`).
* **Lo que hizo AlphaStar:** Primero entrenaron la red mediante **aprendizaje supervisado** clonando millones de decisiones de partidas humanas [1]. Cuando el agente empezó el RL, ya sabía construir bases, minar y atacar como un jugador de nivel diamante. El RL solo se usó para superar el nivel humano.

### 3. El Decaimiento del Reward Shaping (Hacia Zero-Sum)

* A medida que OpenAI Five y AlphaStar se volvían más fuertes, **fueron apagando gradualmente el reward shaping** (los bonus por matar, minar o construir) hasta que el reward fue exclusivamente:
  * $+1$ por ganar la partida.
  * $-1$ por perder.
* Esto garantiza que la red no desarrolle vicios de optimización local y solo le importe la victoria final.

---

## Veredicto del auditor

> Has construido un **motor de aprendizaje formal de nivel investigación moderna adaptado a hardware accesible**.
> No tienes la granja de 1,000 GPUs de OpenAI ni los TPUs de Google, pero la **matemática, los grafos de cómputo y la estructura de tu red son conceptualmente equivalentes**. Si este pipeline demuestra dominar al bot `hard` de OpenRA, habrás logrado con un puñado de recursos lo que hace pocos años requería infraestructura de millones de dólares.

---

## Cómo usar este doc

* **Hoy (Run3):** no intentes cerrar las 3 brechas. El objetivo es `raze >0` y `winrate >0` vs `beginner` con shaping denso.
* **Cuando `winrate >60%` vs `beginner`:** evaluar **self-play casero** (clonar `latest.pt` como oponente) — la brecha 1 es la más barata de atacar.
* **Si se estanca en `hard`:** considerar **imitation** desde replays del propio agente o del bot (brecha 2).
* **Cuando ya gane sin vicios:** **annealing del shaping** `w_raze/w_mining → 0` hacia `win/lose` puro (brecha 3).

Referencias citadas por el auditor: [1] AlphaStar (DeepMind, StarCraft II, 2019), [2] OpenAI Five (Dota 2, 2019).

*Guardado: 2026-08-27 — rama `exp/rl-2026-08-27-scalar19`.*
